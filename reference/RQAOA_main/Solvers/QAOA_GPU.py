"""QAOA для Max-Cut: один класс, GPU-ядра и оптимизация углов.

GPU: backend="cupy" (NVIDIA CUDA). CPU: backend="numpy".
Углы: [gamma_0, ..., gamma_(p-1), beta_0, ..., beta_(p-1)].
Стоимость — минус вес разреза. Сумма abs(state)**2 равна 1/2,
поскольку хранится половина симметричного вектора состояния.
Экземпляр использует общие рабочие буферы и не является потокобезопасным.
"""

import time
from types import SimpleNamespace
from Generators.GraphGenerator import DensRandGraph
import networkx as nx
import numpy as np
from scipy.optimize import minimize


class QAOA:
    """Подготовка гамильтониана, QAOA-эволюция и классическая оптимизация.

    Основные параметры и методы соответствуют исходной структуре класса.
    dtype: complex64 (по умолчанию) или complex128.
    cuda_graphs: повторное использование последовательности CUDA-ядер.
    batch_size: размер пакета при CMA-ES и подборе начальных углов.
    """

    def __init__(self, graph, n_layers, max_iter=300, samples=50, logs=False,
                 callback=False, method='L-BFGS-B', opt_method='layerwise',
                 initialization='ones', *, backend='cupy', dtype='complex64',
                 device=0, cuda_graphs=True, batch_size=16, seed=None, gtol=None):
        self.graph = nx.Graph(graph)
        self.nodes = list(self.graph.nodes())
        self.n_nodes = len(self.nodes)
        if not 1 <= self.n_nodes <= 40:
            raise ValueError('Require 1 <= number of nodes <= 40; memory limits are usually much lower.')
        if int(n_layers) != n_layers or n_layers < 1:
            raise ValueError('n_layers must be a positive integer.')
        if backend not in ('cupy', 'numpy') or dtype not in ('complex64', 'complex128'):
            raise ValueError('Use backend cupy/numpy and dtype complex64/complex128.')
        if opt_method not in ('layerwise', 'simultaneous') or initialization not in ('ones', 'grid', 'approximation'):
            raise ValueError('Invalid opt_method or initialization; use ones/grid/approximation.')
        if max_iter < 1 or samples < 2 or not 1 <= batch_size <= 32767:
            raise ValueError('Require max_iter>=1, samples>=2, 1<=batch_size<=32767.')
        self.n_layers, self.max_iter, self.samples = int(n_layers), int(max_iter), int(samples)
        self.logs, self.callback = logs, callback
        self.method, self.opt_method, self.initialization = method, opt_method, initialization
        self.backend, self.dtype, self.device = backend, dtype, device
        self.cuda_graphs, self.batch_size = cuda_graphs, int(batch_size)
        self.gtol = (1e-4 if dtype == 'complex64' else 1e-7) if gtol is None else gtol
        self.rng = np.random.default_rng(seed)
        mapping = {u: i for i, u in enumerate(self.nodes)}
        self._edges = [(mapping[u], mapping[v], float(d.get('weight', 1.)))
                       for u, v, d in self.graph.edges(data=True) if u != v]
        if not all(np.isfinite(e[2]) for e in self._edges):
            raise ValueError('Edge weights must be finite.')
        self._prepared = False
        self._workspace = None
        self._energies, self._borders = [], []
        self.iterations = self.result = 0
        self.c_solution = self.q_solution = self.optimal_angles = np.array([])
        self.q_solution_dct = {}
        self.c_energy = self.q_energy = self.cut_value = self.max_energy = self.ned = 0.
        self.c_time = self.q_time = 0.
        self.state = self.probs = self.hamiltonian = None
        self.optimizer_results = []

    def _init_approximation(self, optimal_angles):
        """Robust batched grid replacing the original singular two-sample formula.

        The old function used depth p for 2*(p+1) angles and could produce NaNs.
        A general weighted, multi-layer cost is not determined by two beta samples.
        Include beta=0 so the grid can preserve the previous layer's energy.
        """
        old = np.asarray(optimal_angles, dtype=float)
        p = len(old)//2
        gammas = np.linspace(-2*np.pi, 2*np.pi, self.samples)
        betas = np.linspace(0, np.pi, 8, endpoint=False)
        trials = np.empty((len(gammas)*len(betas), 2*(p+1)))
        k = 0
        for g in gammas:
            for b in betas:
                trials[k] = np.r_[old[:p], g, old[p:], b]
                k += 1
        return trials[np.argmin(self.energies_batch(trials))].copy()

    def _maxcut_classical_solver(self):
        """Строит диагональ стоимости и находит точный классический минимум."""
        start = time.perf_counter()
        self._init_backend()
        xp = self._xp
        if self.backend == 'cupy':
            with xp.cuda.Device(self.device), self._stream:
                stats = xp.stack((self.hamiltonian.min(), self.hamiltonian.max(),
                                  self.hamiltonian.argmin().astype(xp.float64)))
                lo, hi, arg = self._to_cpu(stats)
        else:
            lo, hi, arg = self.hamiltonian.min(), self.hamiltonian.max(), self.hamiltonian.argmin()
        self.c_energy, self.max_energy = float(lo), float(hi)
        self.c_solution = self._decode(int(arg))
        self.c_time = time.perf_counter()-start

    def _mix_operator_preset(self):
        """Диагональ X-миксера в базисе Уолша; на GPU вычисляется в ядре."""
        if self.backend == 'numpy':
            idx = np.arange(self._size, dtype=np.uint64)
            pop = np.zeros(self._size, dtype=np.int16)
            for k in range(self.n_nodes-1):
                pop += ((idx >> np.uint64(k)) & 1).astype(np.int16)
            self._mixer_diagonal = self.n_nodes-1-2*pop + np.where(pop % 2, -1, 1)

    def _cost_function(self, depth, final=False, mode='optimization'):
        def fun(angles):
            angles = np.asarray(angles)
            if angles.shape != (2*depth,):
                raise ValueError('Angle count does not match depth.')
            self._prepare()
            if final:
                values, states = self._evaluate_state(angles, state=True)
                value = float(values[0, 0])
                answer = states[0], abs(value)
            else:
                value = self.energy(angles)
                answer = value
            if mode == 'optimization':
                self._record(value)
            return answer
        return fun

    def _simultaneous_optimization(self):
        """Оптимизирует все слои сразу."""
        start = time.perf_counter()
        initial = np.ones(2*self.n_layers)
        if self.initialization != 'ones':
            initial = np.array([], dtype=float)
            for _ in range(self.n_layers):
                initial = self._init_approximation(initial)
        self.optimal_angles = self._optimize(initial, self.n_layers)
        self._save_state(self.optimal_angles)
        self._borders.append(self.iterations)
        self.q_time = time.perf_counter()-start

    def _layerwise_optimization(self):
        """Добавляет по одному слою, переоптимизируя все накопленные углы."""
        start = time.perf_counter()
        initial = (np.ones(2) if self.initialization == 'ones'
                   else self._init_approximation(np.array([], dtype=float)))
        for depth in range(1, self.n_layers+1):
            layer_start = time.perf_counter()
            self.optimal_angles = self._optimize(initial, depth)
            self._borders.append(self.iterations)
            if self.logs:
                print(f'Layer {depth}: E={self.energy(self.optimal_angles):.9g}, '
                      f'time={time.perf_counter()-layer_start:.3f}s')
            if depth < self.n_layers:
                if self.initialization == 'ones':
                    initial = np.r_[self.optimal_angles[:depth], 1., self.optimal_angles[depth:], 1.]
                else:
                    initial = self._init_approximation(self.optimal_angles)
        self._save_state(self.optimal_angles)
        self.q_time = time.perf_counter()-start

    def _adam_optimization(self, init_params, depth):
        angles = np.array(init_params, dtype=float, copy=True)
        m, v = np.zeros_like(angles), np.zeros_like(angles)
        for t in range(1, self.max_iter+1):
            e, grad = self.energy_and_grad(angles)
            self._record(e)
            m = .9*m+.1*grad
            v = .999*v+.001*grad**2
            angles -= .05*(m/(1-.9**t))/(np.sqrt(v/(1-.999**t))+1e-8)
        return angles

    def _cmaes_optimization(self, init_params, depth):

        n = len(init_params)
        sigma = 0.5
        lamb = 4 + int(3 * np.log(n))
        mu = lamb // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1 / np.sum(weights ** 2)

        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        chi = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        mean = init_params.copy()
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)
        B = np.eye(n)
        D = np.ones(n)
        invsqrtC = np.eye(n)
        cost_function = self._cost_function(depth)
        best_angles = mean.copy()
        best_energy = cost_function(mean)

        for g in range(max(1, self.max_iter // lamb)):
            arz = self.rng.standard_normal((lamb, n))
            ary = (arz * D) @ B.T
            arx = mean + sigma * ary
            fitness = self.energies_batch(arx, record=True)
            idx = np.argsort(fitness)

            old_mean = mean.copy()
            mean = weights @ arx[idx[:mu]]
            y_w = (mean - old_mean) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ y_w)
            hsig = int(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * g + 2)) / chi < 1.4 + 2 / (n + 1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y_w

            artmp = (arx[idx[:mu]] - old_mean) / sigma
            C = ((1 - c1 - cmu + (1 - hsig) * c1 * cc * (2 - cc)) * C +
                 c1 * np.outer(pc, pc) + cmu * ((weights[:, None] * artmp).T @ artmp))
            sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chi - 1))

            eig_values, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(eig_values, 1e-30))
            invsqrtC = B @ np.diag(1 / D) @ B.T

            if fitness[idx[0]] < best_energy:
                best_energy = fitness[idx[0]]
                best_angles = arx[idx[0]].copy()

        return best_angles

    def solve(self):
        """Подготовка задачи -> оптимизация углов -> извлечение решения."""
        self._prepare()
        self._energies.clear()
        self._borders.clear()
        self.iterations = 0
        self.optimizer_results.clear()

        if self.opt_method == 'layerwise':
            self._layerwise_optimization()
        elif self.opt_method == 'simultaneous':
            self._simultaneous_optimization()
        else:
            raise ValueError(f'Unknown optimization method: {self.opt_method}')

        xp = self._xp
        if self.backend == 'cupy':
            with xp.cuda.Device(self.device), self._stream:
                self.probs = xp.abs(self.state)**2
                index = int(self._to_cpu(self.probs.argmax()))
        else:
            self.probs = np.abs(self.state)**2
            index = int(self.probs.argmax())
        self.q_solution = self._decode(index)
        self.q_solution_dct = dict(zip(self.nodes, self.q_solution.tolist()))
        self._cut_value()
        self.result = int(np.isclose(self.cut_value, self.c_energy, atol=1e-7, rtol=1e-7))
        self.ned = self.NED(self.expectation)
        self._callback()
        return self

    def _cut_value(self):
        self.cut_value = -sum(w for u, v, w in self._edges
                             if self.q_solution[u] != self.q_solution[v])

    def _callback(self):
        if self.callback:
            print(f'Exact energy: {self.c_energy:.9g}; expectation: {self.expectation:.9g}; '
                  f'most-probable cut: {self.cut_value:.9g}; NED: {self.ned:.6g}')
            print(f'QAOA time: {self.q_time:.3f}s; preprocessing: {self.c_time:.3f}s')

    def plot_convergence(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.energies, lw=.8, label='Objective evaluations')
        for border in self.layers_borders:
            ax.axvline(border-.5, color='red', ls='--', alpha=.4)
        ax.axhline(self.c_energy, color='green', ls='--', label='Exact energy')
        ax.set(xlabel='Objective evaluation', ylabel='Signed energy', title='QAOA convergence')
        ax.grid(alpha=.3)
        ax.legend()
        plt.show()

    def NED(self, energy):
        span = self.max_energy-self.c_energy
        return float((self.max_energy-energy)/span) if span else float('nan')

    def energy(self, angles):
        """Signed expectation, to minimize. Does not add a convergence-log entry."""
        self._prepare()
        return float(self._evaluate_state(angles)[0, 0])

    def energy_and_grad(self, angles):
        """Analytic adjoint gradient; no finite differences or parameter shift."""
        self._prepare()
        out = self._evaluate_state(angles, gradient=True)[0]
        return float(out[0]), out[1:].copy()

    def energies_batch(self, angles, *, record=False):
        """Independent angle vectors run together; memory is bounded by batch_size."""
        self._prepare()
        angles = np.asarray(angles, dtype=np.float64)
        if angles.ndim != 2 or not angles.shape[0]:
            raise ValueError('Expected nonempty (batch, 2*depth).')
        step = min(self.batch_size, len(angles))
        if self.backend == 'cupy':
            with self._xp.cuda.Device(self.device):
                free, _ = self._xp.cuda.runtime.memGetInfo()
            # Conservative working-set bound; the allocator may also retain freed blocks.
            per_row = self._size*np.dtype(self.dtype).itemsize + 8192
            step = min(step, max(1, int(free*0.25)//per_row))
        chunks = []
        for i in range(0, len(angles), step):
            chunk = angles[i:i+step]
            # Padding the final batch lets all chunks reuse one captured execution graph.
            padded = chunk if len(chunk) == step else np.concatenate((chunk, np.repeat(chunk[-1:], step-len(chunk), axis=0)))
            chunks.append(self._evaluate_state(padded)[:len(chunk), 0])
        values = np.concatenate(chunks)
        if record:
            self._energies.extend(values.tolist())
            self.iterations += len(values)
        return values

    def state_numpy(self, full=False):
        """Explicit GPU->CPU copy. full=True reconstructs the original 2**n state."""
        if self.state is None:
            raise RuntimeError('Call solve() first.')
        state = self._to_cpu(self.state)
        return np.r_[state, state[::-1]] if full else state

    @property
    def energies(self):
        return np.asarray(self._energies)

    @property
    def layers_borders(self):
        return np.asarray(self._borders, dtype=int)

    def _record(self, energy):
        self._energies.append(float(energy))
        self.iterations += 1

    def _optimize(self, initial, depth):
        if self.method == 'ADAM':
            return self._adam_optimization(initial, depth)
        if self.method == 'CMA-ES':
            return self._cmaes_optimization(initial, depth)
        def fg(a):
            e, g = self.energy_and_grad(a)
            self._record(e)
            return e, g
        derivative_free = self.method.lower() in ('nelder-mead', 'powell', 'cobyla', 'cobyqa')
        options = {'maxiter': self.max_iter}
        if self.method.upper() in ('L-BFGS-B', 'BFGS', 'CG'):
            options['gtol'] = self.gtol
        if self.method.upper() == 'L-BFGS-B':
            options.update(maxfun=self.max_iter, ftol=1e-6 if self.dtype == 'complex64' else 1e-12)
        result = minimize(self._cost_function(depth) if derivative_free else fg,
                          initial, jac=None if derivative_free else True,
                          method=self.method, options=options)
        self.optimizer_results.append(result)
        if self.logs and not result.success:
            print('Optimizer:', result.message)
        return result.x

    def _save_state(self, angles):
        out, states = self._evaluate_state(angles, state=True)
        self.state = states[0]
        self.expectation = float(out[0, 0])
        self.q_energy = abs(self.expectation)

    def _decode(self, index):
        return 2*((index >> np.arange(self.n_nodes-1, -1, -1, dtype=np.int64)) & 1)-1

    def _prepare(self):
        if not self._prepared:
            self._maxcut_classical_solver()
            self._mix_operator_preset()
            self._prepared = True

    # Вычислительные методы используют буферы этого же экземпляра QAOA.

    @property
    def _blocks(self):
        return max(1, min(256, (self._size + 255) // 256))

    def _call(self, name, grid, args, threads=256):
        self._kernels[name](grid, (threads,), args, stream=self._stream)

    def _to_cpu(self, a):
        if self.backend == 'numpy':
            return np.asarray(a).copy()
        with self._xp.cuda.Device(self.device):
            self._stream.synchronize()
            return self._xp.asnumpy(a)

    def _fwht(self, w, both=False):
        rows = w.batch * (2 if both else 1)
        if self.backend == 'numpy':
            self._walsh_numpy(w.work[:rows])
            return
        T = min(256, self._size)
        self._call('walsh_tile', (self._size // T, rows),
                   (w.work, np.uint64(self._size), np.int32(T), np.float64(1/np.sqrt(self._size))), 128)
        stride = T
        while stride < self._size:
            group = min(16, self._size//stride)
            self._call('walsh_high', (self._size//(group*16), rows),
                       (w.work, np.uint64(self._size), np.uint64(stride), np.int32(group)), 128)
            stride *= group

    def _phase(self, w, col, mixer=False, inverse=False, both=False):
        rows = w.batch * (2 if both else 1)
        sign = 1 if inverse else -1
        if self.backend == 'numpy':
            d = self._mixer_diagonal if mixer else self.hamiltonian
            factor = np.exp(sign*1j*w.angles[:, col, None]*d).astype(self._state_dtype)
            w.work[:w.batch] *= factor
            if both:
                w.work[w.batch:rows] *= factor
            return
        self._call('diagonal', (self._blocks, rows), (w.work, np.uint64(self._size), self.hamiltonian,
                   w.angles, np.int32(2*w.depth), np.int32(col), np.int32(mixer),
                   np.int32(self.n_nodes-1), np.int32(w.batch), np.int32(sign)))

    def _reduce(self, w, mode, col):
        x = w.work[:w.batch]
        a = w.work[w.batch:] if w.gradient else x
        if self.backend == 'numpy':
            if mode == 0:
                w.out[:, col] = 2*np.sum((x.real.astype(float)**2+x.imag.astype(float)**2)*self.hamiltonian, axis=1)
                if w.gradient:
                    a[:] = self.hamiltonian*x
            else:
                d = self.hamiltonian if mode == 1 else self._mixer_diagonal
                w.out[:, col] = 4*np.sum(d*(a.real.astype(float)*x.imag-a.imag.astype(float)*x.real), axis=1)
            return
        self._call('partial_reduce', (self._blocks, w.batch),
                   (x, a, self.hamiltonian, np.uint64(self._size), np.int32(mode), np.int32(self.n_nodes-1),
                    np.int32(w.gradient), w.partial))
        self._call('finish_reduce', (w.batch,), (w.partial, w.out, np.int32(self._blocks),
                   np.int32(w.out.shape[1]), np.int32(col)))

    def _trajectory(self, w):
        if self.backend == 'numpy':
            w.work[:w.batch].fill(1/np.sqrt(2*self._size))
        else:
            self._call('init', (self._blocks, w.batch),
                       (w.work, np.uint64(self._size), np.float64(1/np.sqrt(2*self._size))))
        for k in range(w.depth):
            self._phase(w, k)
            self._fwht(w)
            self._phase(w, w.depth+k, mixer=True)
            self._fwht(w)
        self._reduce(w, 0, 0)
        if not w.gradient:
            return
        # Reverse both the state and adjoint; no O(depth*N) forward tape.
        for k in range(w.depth-1, -1, -1):
            self._fwht(w, both=True)
            self._reduce(w, 2, 1+w.depth+k)
            self._phase(w, w.depth+k, mixer=True, inverse=True, both=True)
            self._fwht(w, both=True)
            self._reduce(w, 1, 1+k)
            self._phase(w, k, inverse=True, both=True)

    def _allocate(self, depth, batch, gradient):
        xp = self._xp
        width = 1+2*depth if gradient else 1
        w = SimpleNamespace(
            depth=depth, batch=batch, gradient=gradient, graph=None,
            work=xp.empty((batch*(2 if gradient else 1), self._size), dtype=self._state_dtype),
            angles=xp.zeros((batch, 2*depth), dtype=xp.float64),
            partial=xp.empty((batch, self._blocks), dtype=xp.float64),
            out=xp.empty((batch, width), dtype=xp.float64),
        )
        if self.backend != 'numpy':
            def pinned(shape):
                count = int(np.prod(shape))
                mem = xp.cuda.alloc_pinned_memory(
                    count * np.dtype(np.float64).itemsize
                )
                return np.frombuffer(
                    mem, dtype=np.float64, count=count
                ).reshape(shape)
            w.host_angles = pinned((batch, 2*depth))
            w.host_out = pinned((batch, width))
            # Force JIT compilation before capture. No CPU transfers during capture.
            self._trajectory(w)
            self._stream.synchronize()
            if self.cuda_graphs:
                self._stream.begin_capture()
                try:
                    self._trajectory(w)
                finally:
                    captured = self._stream.end_capture()
                w.graph = captured
        return w

    def _evaluate_workspace(self, angles, gradient, state):
        batch, width = angles.shape
        depth = width//2
        key = (depth, batch, gradient)
        if self._workspace is None or (self._workspace.depth, self._workspace.batch, self._workspace.gradient) != key:
            self._workspace = None
            self._workspace = self._allocate(*key)
        w = self._workspace
        if self.backend == 'numpy':
            w.angles[:] = angles
            self._trajectory(w)
            return (w.out.copy(), w.work[:batch].copy()) if state else w.out.copy()
        np.copyto(w.host_angles, angles)
        w.angles.set(w.host_angles, stream=self._stream)
        if w.graph is None:
            self._trajectory(w)
        else:
            w.graph.launch(stream=self._stream)
        w.out.get(out=w.host_out, stream=self._stream, blocking=False)
        self._stream.synchronize()  # one host round trip for energy AND all derivatives
        result = w.host_out.copy()
        if state:
            saved = w.work[:batch].copy()
            self._stream.synchronize()  # safe even if caller uses another CUDA stream
            return result, saved
        return result

    def _evaluate_state(self, angles, gradient=False, state=False):
        angles = np.ascontiguousarray(np.atleast_2d(angles), dtype=np.float64)
        if angles.ndim != 2 or angles.shape[1] % 2 or not angles.shape[0]:
            raise ValueError('Expected a nonempty (batch, 2*depth) angle array.')
        if not np.isfinite(angles).all():
            raise ValueError('Angles must be finite.')
        if gradient and state:
            raise ValueError('The adjoint sweep overwrites the state; request a separate state evaluation.')
        if self.backend == 'numpy':
            return self._evaluate_workspace(angles, gradient, state)
        with self._xp.cuda.Device(self.device), self._stream:
            return self._evaluate_workspace(angles, gradient, state)

    def _init_backend(self):
        n, edges = self.n_nodes, self._edges
        dtype, backend = self.dtype, self.backend
        device, cuda_graphs = self.device, self.cuda_graphs
        self.n_nodes, self._size = n, 1 << (n - 1)
        self._state_dtype = np.dtype(dtype)
        self.backend, self.cuda_graphs = backend, cuda_graphs
        self._workspace = None  # one workspace only; old layer graphs do not retain state arrays
        self._edges = edges
        self.device = device
        if backend == 'numpy':
            self._xp = np
            idx = np.arange(self._size, dtype=np.uint64)
            self.hamiltonian = np.zeros(self._size, dtype=np.float64)
            for u, v, w in edges:
                self.hamiltonian -= w * (((idx >> np.uint64(n-1-u)) ^ (idx >> np.uint64(n-1-v))) & 1)
            return
        try:
            import cupy as cp
        except ImportError as exc:
            raise RuntimeError('GPU backend requires CuPy. See README_GPU.md. '
                               'For CPU execution use backend="numpy".') from exc
        self._xp = cp
        with cp.cuda.Device(device):
            if cp.cuda.runtime.is_hip:
                raise RuntimeError('This implementation targets NVIDIA CUDA, not ROCm.')
            self._stream = cp.cuda.Stream(non_blocking=True)
            with self._stream:
                code = self._CUDA_SOURCE.replace('typedef REAL R;',
                                           'typedef ' + ('float' if dtype == 'complex64' else 'double') + ' R;')
                self._module = cp.RawModule(code=code, options=('--std=c++11',))
                names = ('build_h', 'init', 'walsh_tile', 'walsh_high', 'diagonal',
                         'partial_reduce', 'finish_reduce')
                self._kernels = {name: self._module.get_function(name) for name in names}
                self.hamiltonian = cp.empty(self._size, dtype=cp.float64)
                eu = cp.asarray([n-1-e[0] for e in edges], dtype=cp.int32)
                ev = cp.asarray([n-1-e[1] for e in edges], dtype=cp.int32)
                ew = cp.asarray([e[2] for e in edges], dtype=cp.float64)
                self._call('build_h', (self._blocks,), (self.hamiltonian, np.uint64(self._size),
                           eu, ev, ew, np.int32(len(edges))))
            self._stream.synchronize()

    @staticmethod
    def _walsh_numpy(x):
        """Normalized, in-place Walsh-Hadamard transform of each row."""
        N = x.shape[-1]
        stride = 1
        while stride < N:
            y = x.reshape(-1, N // (2 * stride), 2, stride)
            a, b = y[:, :, 0].copy(), y[:, :, 1].copy()
            y[:, :, 0] = a + b
            y[:, :, 1] = a - b
            stride *= 2
        x *= 1 / np.sqrt(N)

    _CUDA_SOURCE = r'''

typedef REAL R;
struct C { R x, y; };
__device__ C phase(C z, double a) {
    double s, c; sincos(a, &s, &c);
    return {(R)(z.x*c-z.y*s), (R)(z.x*s+z.y*c)};
}
__device__ int mixer_value(unsigned long long i, int m) {
    int k=__popcll(i); return m-2*k+((k&1)?-1:1);
}
extern "C" __global__ void build_h(
    double* H, unsigned long long N, const int* u, const int* v,
    const double* w, int E) {
    for (unsigned long long i=blockIdx.x*blockDim.x+threadIdx.x;
         i<N; i+=(unsigned long long)blockDim.x*gridDim.x) {
        double h=0;
        for(int e=0;e<E;e++) if(((i>>u[e])^(i>>v[e]))&1ULL) h-=w[e];
        H[i]=h;
    }
}
extern "C" __global__ void init(C* x, unsigned long long N, double a) {
    for(unsigned long long i=blockIdx.x*blockDim.x+threadIdx.x;
        i<N; i+=(unsigned long long)blockDim.x*gridDim.x) {
        x[(unsigned long long)blockIdx.y*N+i]={(R)a,0};
    }
}
// A tile holds 256 amplitudes. All low-bit butterfly stages stay in shared memory.
// Normalization is done here once per full transform (including high-bit stages).
extern "C" __global__ void walsh_tile(C* x, unsigned long long N, int T, double scale) {
    __shared__ C s[256];
    unsigned long long base=(unsigned long long)blockIdx.y*N+(unsigned long long)blockIdx.x*T;
    for(int j=threadIdx.x;j<T;j+=blockDim.x) s[j]=x[base+j];
    __syncthreads();
    for(int stride=1;stride<T;stride*=2) {
        for(int j=threadIdx.x;j<T/2;j+=blockDim.x) {
            int a=(j/stride)*(2*stride)+j%stride, b=a+stride;
            C u=s[a],v=s[b]; s[a]={u.x+v.x,u.y+v.y}; s[b]={u.x-v.x,u.y-v.y};
        }
        __syncthreads();
    }
    for(int j=threadIdx.x;j<T;j+=blockDim.x)
        x[base+j]={(R)(s[j].x*scale),(R)(s[j].y*scale)};
}
// Up to four high-bit stages per launch. Sixteen adjacent columns preserve
// contiguous accesses while butterflies operate along the strided dimension.
extern "C" __global__ void walsh_high(C* x, unsigned long long N,
    unsigned long long stride, int rows) {
    __shared__ C s[256];
    unsigned long long tiles=stride/16;
    unsigned long long base=(unsigned long long)blockIdx.y*N+
        ((unsigned long long)blockIdx.x/tiles)*(stride*rows)+(blockIdx.x%tiles)*16;
    for(int j=threadIdx.x;j<rows*16;j+=blockDim.x) s[j]=x[base+(j/16)*stride+j%16];
    __syncthreads();
    for(int k=1;k<rows;k*=2) {
        for(int j=threadIdx.x;j<(rows/2)*16;j+=blockDim.x) {
            int pair=j/16,col=j%16,a=((pair/k)*(2*k)+pair%k)*16+col,b=a+k*16;
            C u=s[a],v=s[b]; s[a]={u.x+v.x,u.y+v.y}; s[b]={u.x-v.x,u.y-v.y};
        }
        __syncthreads();
    }
    for(int j=threadIdx.x;j<rows*16;j+=blockDim.x) x[base+(j/16)*stride+j%16]=s[j];
}
extern "C" __global__ void diagonal(C* x, unsigned long long N, const double* H,
    const double* angles, int width, int col, int mode, int m, int batch, int sign) {
    double a=sign*angles[(blockIdx.y%batch)*width+col];
    for(unsigned long long i=blockIdx.x*blockDim.x+threadIdx.x;
        i<N;i+=(unsigned long long)blockDim.x*gridDim.x) {
        double d=mode?mixer_value(i,m):H[i];
        unsigned long long j=(unsigned long long)blockIdx.y*N+i;
        x[j]=phase(x[j],a*d);
    }
}
// mode=0: energy plus adjoint initialization; mode=1/2: gamma/beta gradient.
extern "C" __global__ void partial_reduce(C* x, C* adj, const double* H,
    unsigned long long N, int mode, int m, int init_adj, double* partial) {
    __shared__ double s[256];
    double acc=0;
    unsigned long long base=(unsigned long long)blockIdx.y*N;
    for(unsigned long long i=blockIdx.x*blockDim.x+threadIdx.x;
        i<N;i+=(unsigned long long)blockDim.x*gridDim.x) {
        C z=x[base+i];
        if(mode==0) {
            double h=H[i];
            acc+=2*h*((double)z.x*z.x+(double)z.y*z.y);
            if(init_adj) adj[base+i]={(R)(h*z.x),(R)(h*z.y)};
        } else {
            C a=adj[base+i]; double d=(mode==1)?H[i]:mixer_value(i,m);
            acc+=4*d*((double)a.x*z.y-(double)a.y*z.x);
        }
    }
    s[threadIdx.x]=acc; __syncthreads();
    for(int k=blockDim.x/2;k;k/=2) {
        if(threadIdx.x<k) s[threadIdx.x]+=s[threadIdx.x+k]; __syncthreads();
    }
    if(threadIdx.x==0) partial[blockIdx.y*gridDim.x+blockIdx.x]=s[0];
}
extern "C" __global__ void finish_reduce(const double* part, double* out,
    int blocks, int width, int col) {
    __shared__ double s[256]; double v=0;
    for(int i=threadIdx.x;i<blocks;i+=blockDim.x) v+=part[blockIdx.x*blocks+i];
    s[threadIdx.x]=v; __syncthreads();
    for(int k=blockDim.x/2;k;k/=2) {
        if(threadIdx.x<k) s[threadIdx.x]+=s[threadIdx.x+k]; __syncthreads();
    }
    if(threadIdx.x==0) out[blockIdx.x*width+col]=s[0];
}
'''


if __name__ == "__main__":

    G = DensRandGraph(number_of_nodes=28,
                      density=0.7,
                      weights="pos_neg",
                      seed=1000)


    qaoa = QAOA(
        graph=G,
        n_layers=10,
        max_iter=100,
        method="L-BFGS-B",
        opt_method="layerwise",
        initialization="ones",
        backend="cupy",
        dtype="complex64",
        logs=True,
        callback=True,
    )

    qaoa.solve()

    print("Оптимальные углы:", qaoa.optimal_angles)
    print("Решение:", qaoa.q_solution_dct)
    print("Средний вес разреза:", -qaoa.expectation)
    print("Вес найденного разреза:", -qaoa.cut_value)