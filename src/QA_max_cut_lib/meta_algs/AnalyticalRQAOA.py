"""Standalone p=1 AnalyticalRQAOA; no inheritance or QAOA imports.

Requires numpy, networkx; matplotlib only for plotting.
C = offset + sum(w_uv * (1-Zu*Zv)/2), U = exp(-i beta sum X) exp(-i gamma C).
All angles use +C; steps_angs_qaoa changes gamma's sign for the supplied QAOA.py.
Only existing nonzero-weight edges participate in correlation selection.

Weighted identity (k runs over neighbors of u or v, excluding u,v;
missing weights are zero):
R_uv = -.5*sin(4b)*sin(g*w_uv)*(prod cos(g*w_uk)+prod cos(g*w_vk))
       + .5*sin(2b)**2*(prod cos(g*(w_uk-w_vk))-prod cos(g*(w_uk+w_vk))).
This supports signed real weights and remains valid after contraction.

Integer weights: gamma in [0, pi] is sufficient. Real weights require an
explicit finite gamma_bounds search window; it is not a global guarantee.
Beta is optimized exactly; gamma is searched on a grid. Correlation rounding
is heuristic and can discard the original optimum. Exact enumeration is only
performed for the final residual graph, never for the full original graph.
"""
from itertools import product
from numbers import Integral
from time import perf_counter

import networkx as nx
import numpy as np

from Generators.GraphGenerator import RegularGraph, DensRandGraph


class AnalyticalRQAOA:
    """One file implementation of optimization -> correlations -> contraction.

    treshold retains the spelling in the original package. n_layers must be 1.
    samples counts gamma grid points; gamma_batch limits temporary memory.
    store_maps=True saves each sparse edge map in history (can use much memory).
    logs prints steps, callback prints final summary, draw_process plots after
    solve(). max_exact_nodes is a guard on terminal exhaustive enumeration.

    Outputs:
      q_solution: +/-1 spins in original node order; q_solution_dct: label->spin.
      solution_cut: positive-sign original weighted cut; cut_value=-solution_cut
        follows the supplied RQAOA package's convention.
      q_energy: last QAOA expected original cut, NOT final discrete solution_cut.
      fixed_corrs: direct (removed, retained, sign) constraints, in elimination
        order; arbitrary hashable node labels are preserved.
      edge_correlations / m_map: sparse dict for the LAST optimized pre-reduction
        graph. correlation_nodes identifies its nodes. Not a full all-pairs map.
      reduced_graph: terminal residual graph; offset: accumulated cut constant.
      steps_energies: negative expected ORIGINAL cut at each recursive step.
      gamma_history, beta_history: Python lists of optimal angles per step (+C),
        in radians; reset on each solve(). Index 0 corresponds to step 1.
      steps_angs: [gamma,beta] per step; history: graph sizes, offset, selected
        edge, correlation, angles, mean cut and timing (maps optional).
      c_energy and ned remain None: the original optimum is not computed.

    solve() resets the run and returns self. Input graph is copied.
    """
    def __init__(self, graph, treshold=3, n_layers=1, samples=1000,
                 logs=False, callback=False, draw_process=False,
                 gamma_bounds=None, gamma_batch=256, store_maps=False,
                 max_exact_nodes=20):
        for name, value, minimum in [('treshold', treshold, 1),
                                      ('samples', samples, 2),
                                      ('gamma_batch', gamma_batch, 1),
                                      ('max_exact_nodes', max_exact_nodes, 1)]:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        if n_layers != 1:
            raise ValueError('AnalyticalRQAOA supports n_layers=1 only')
        if treshold > max_exact_nodes:
            raise ValueError('treshold exceeds max_exact_nodes enumeration guard')
        if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
            raise ValueError('A simple undirected graph without self-loops is required')
        self.graph = nx.Graph(graph)
        integer_weights = True
        for u, v, data in self.graph.edges(data=True):
            w = float(data.get('weight', 1))
            if not np.isfinite(w):
                raise ValueError('Weights must be finite real numbers')
            data['weight'] = w
            integer_weights &= w.is_integer()
        if gamma_bounds is None:
            if not integer_weights:
                raise ValueError('For real weights specify gamma_bounds=(lower, upper); no universal period exists')
            gamma_bounds = (0., np.pi)
        bounds = np.asarray(gamma_bounds, dtype=float)
        if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1]:
            raise ValueError('gamma_bounds must contain two finite increasing numbers')
        self.gamma_bounds = tuple(float(x) for x in bounds)
        self.treshold = int(treshold)
        self.n_layers = 1
        self.samples = int(samples)
        self.gamma_batch = int(gamma_batch)
        self.max_exact_nodes = int(max_exact_nodes)
        self.logs = self.rq_logs = bool(logs)
        self.callback = self.rq_callback = bool(callback)
        self.draw_process = bool(draw_process)
        self.store_maps = bool(store_maps)
        self.method = 'Gamma grid + exact beta; weighted edge correlations'
        self.node_order = list(self.graph)
        self.original_n_nodes = len(self.node_order)
        self._reset()

    def _reset(self):
        self.reduced_graph = nx.Graph(self.graph)
        self.reduced_graph.remove_edges_from([(u,v) for u,v,d in self.reduced_graph.edges(data=True) if d['weight'] == 0])
        self.offset = 0.
        self.n_nodes = self.reduced_graph.number_of_nodes()
        self.step = self.iterations = self.cost_evaluations = 0
        self.fixed_corrs = []
        self.free_spins = {}
        self.history = []
        self.steps_angs = []
        self.gamma_history = []
        self.beta_history = []
        self.steps_angs_qaoa = []
        self.steps_energies = []
        self.steps_borders = []
        self.energies = []
        self.edge_correlations = self.m_map = {}
        self.correlation_nodes = []
        self.optimal_angles = np.array([])
        self.optimal_angles_qaoa = np.array([])
        self.q_solution = np.array([], dtype=int)
        self.q_solution_dct = {}
        self.q_energy = None
        self.solution_cut = self.cut_value = None
        self.c_energy = self.ned = None
        self.q_time = self.c_time = self.rq_time = 0.
        self.result = None
        self.solved = False
        self.gammas = self.betas = self.profile = np.array([])
        self.coefficients = np.array([])

    def _edge_terms(self, u, v, gamma):
        """Return S,Q with R_uv=S*sin(4 beta)+Q*sin(2 beta)**2.

        Products are accumulated without division by cosines: safe when cos=0.
        Temporary arrays have the shape of gamma (no dense n*n or 2**n objects).
        """
        G = self.reduced_graph
        g = np.asarray(gamma, dtype=float)
        pu = np.ones_like(g); pv = np.ones_like(g)
        minus = np.ones_like(g); plus = np.ones_like(g)
        for k in (G[u].keys() | G[v].keys()) - {u, v}:
            a = G[u][k]['weight'] if k in G[u] else 0.
            b = G[v][k]['weight'] if k in G[v] else 0.
            pu *= np.cos(g*a)
            pv *= np.cos(g*b)
            minus *= np.cos(g*(a-b))
            plus *= np.cos(g*(a+b))
        return -.5*np.sin(g*G[u][v]['weight'])*(pu+pv), .5*(minus-plus)

    def cost_function(self, gamma, beta):
        """Expected ORIGINAL cut, including offset, on the current reduced graph."""
        g, b = np.broadcast_arrays(np.asarray(gamma, dtype=float), np.asarray(beta, dtype=float))
        value = np.full(g.shape, self.offset, dtype=float)
        for u,v,d in self.reduced_graph.edges(data=True):
            s,q = self._edge_terms(u,v,g)
            r = s*np.sin(4*b)+q*np.sin(2*b)**2
            value += d['weight']*(1-r)/2
        return float(value) if value.ndim == 0 else value

    def _optimize_angles(self):
        start = perf_counter()
        self.gammas = np.linspace(*self.gamma_bounds, self.samples)
        baseline = self.offset + .5*sum(d['weight'] for _,_,d in self.reduced_graph.edges(data=True))
        coeff = np.empty((self.samples,3))
        # Two beta settings, as in AnalyticalQAOA. Batch gamma for bounded memory.
        for left in range(0,self.samples,self.gamma_batch):
            right = min(left+self.gamma_batch,self.samples)
            g = self.gammas[left:right]
            y0 = self.cost_function(g,3*np.pi/4)
            y1 = self.cost_function(g,3*np.pi/8)
            a0 = (baseline+y0)/2
            coeff[left:right] = np.column_stack((a0,a0-y1,(baseline-y0)/2))
        amplitude = np.hypot(coeff[:,1],coeff[:,2])
        self.betas = (np.arctan2(coeff[:,1],coeff[:,2])/4) % (np.pi/2)
        self.betas[amplitude == 0] = 0.
        self.profile = coeff[:,0]+amplitude
        idx = int(np.argmax(self.profile))
        self.coefficients = coeff[idx].copy()
        self.optimal_angles = np.array([self.gammas[idx],self.betas[idx]])
        self.optimal_angles_qaoa = self.optimal_angles*np.array([-1,1])
        self.q_energy = float(self.profile[idx])
        self.iterations += self.samples
        self.cost_evaluations += 2*self.samples
        self.energies.extend((-self.profile).tolist())
        self.q_time += perf_counter()-start

    def _build_correlation_map(self):
        g,b = self.optimal_angles
        self.edge_correlations = {}
        for u,v in self.reduced_graph.edges():
            s,q = self._edge_terms(u,v,g)
            self.edge_correlations[(u,v)] = float(s*np.sin(4*b)+q*np.sin(2*b)**2)
        self.m_map = self.edge_correlations
        self.correlation_nodes = list(self.reduced_graph)

    def mean_map(self, dense=False):
        """Last optimized EDGE map. dense=True returns (matrix, node_order).

        Dense matrix is symmetric, diagonal/nonedges zero; nonedge zero means
        'not evaluated', not a claim that the actual correlation is zero.
        """
        if not dense:
            return dict(self.edge_correlations)
        nodes = list(self.correlation_nodes)
        idx = {u:i for i,u in enumerate(nodes)}
        matrix = np.zeros((len(nodes),len(nodes)))
        for (u,v),r in self.edge_correlations.items():
            matrix[idx[u],idx[v]] = matrix[idx[v],idx[u]] = r
        return matrix,nodes

    def _reduce_graph(self, removed, retained, sign):
        """Impose z_removed=sign*z_retained and preserve the restricted cost.

        For each other neighbor k: w_retained,k += sign*w_removed,k;
        add (1-sign)*w_removed,k/2 to the constant. The contracted edge
        contributes (1-sign)*w_removed,retained/2 to the same constant.
        """
        G = self.reduced_graph
        if sign not in (-1,1):
            raise ValueError('Correlation constraint must have sign +1 or -1')
        for k,d in list(G[removed].items()):
            w = d['weight']
            self.offset += (1-sign)*w/2
            if k == retained:
                continue
            new_weight = (G[retained][k]['weight'] if G.has_edge(retained,k) else 0.)+sign*w
            if new_weight == 0:
                if G.has_edge(retained,k): G.remove_edge(retained,k)
            else:
                G.add_edge(retained,k,weight=new_weight)
        G.remove_node(removed)
        self.fixed_corrs.append((removed,retained,int(sign)))
        self.n_nodes = G.number_of_nodes()

    def _remove_isolates(self):
        # Isolates carry no remaining cost; fixing +1 does not lose an optimum.
        for u in list(nx.isolates(self.reduced_graph)):
            self.free_spins[u] = 1
            self.reduced_graph.remove_node(u)
        self.n_nodes = self.reduced_graph.number_of_nodes()

    def _recursive_step(self):
        start = perf_counter()
        self._optimize_angles()
        self._build_correlation_map()
        # max keeps the first edge in insertion order when magnitudes tie.
        u,v = max(self.edge_correlations,key=lambda edge:abs(self.edge_correlations[edge]))
        corr = self.edge_correlations[(u,v)]
        # Never sign(0)=0: zero correlations use a deterministic +1 constraint.
        sign = 1 if corr >= 0 else -1
        row = dict(step=self.step+1,n_nodes=self.n_nodes,
                   n_edges=self.reduced_graph.number_of_edges(),offset=self.offset,
                   gamma=float(self.optimal_angles[0]),beta=float(self.optimal_angles[1]),
                   expected_cut=self.q_energy,selected_edge=(u,v),correlation=corr,sign=sign)
        if self.store_maps:
            row['edge_correlations'] = self.mean_map()
        self._reduce_graph(u,v,sign)
        self.step += 1
        self.steps_angs.append(self.optimal_angles.copy())
        self.gamma_history.append(float(self.optimal_angles[0]))
        self.beta_history.append(float(self.optimal_angles[1]))
        self.steps_angs_qaoa.append(self.optimal_angles_qaoa.copy())
        self.steps_energies.append(-self.q_energy)
        self.steps_borders.append(self.iterations)
        row['time'] = perf_counter()-start
        self.history.append(row)
        if self.logs:
            print(f"Step {self.step:3d} | nodes {row['n_nodes']:4d} | "
                  f"<C> {self.q_energy:.8f} | gamma {row['gamma']:.6f}, beta {row['beta']:.6f} | "
                  f"edge {u!r},{v!r}: R={corr:+.6f}, sign={sign:+d} | {row['time']:.3f} s")

    def _solve_residual(self):
        start = perf_counter()
        nodes = list(self.reduced_graph)
        if len(nodes) > self.max_exact_nodes:
            raise ValueError('Residual graph exceeds max_exact_nodes')
        best = -np.inf; best_spins = {}
        # Global spin-flip symmetry allows fixing the first residual spin to +1.
        for tail in product((-1,1),repeat=max(0,len(nodes)-1)):
            spins = dict(zip(nodes,(1,)+tail))
            cut = self.offset + sum(d['weight']*(1-spins[u]*spins[v])/2
                                     for u,v,d in self.reduced_graph.edges(data=True))
            if cut > best:
                best,best_spins = cut,spins
        self.residual_solution = best_spins
        self.residual_cut_with_offset = float(best)
        self.c_time = perf_counter()-start

    def _restore_solution(self):
        spins = dict(self.free_spins)
        spins.update(self.residual_solution)
        for u,v,sign in reversed(self.fixed_corrs):
            spins[u] = sign*spins[v]
        self.q_solution_dct = {u:int(spins[u]) for u in self.node_order}
        self.q_solution = np.array([spins[u] for u in self.node_order],dtype=int)
        self.solution_cut = float(sum(d['weight']*(1-spins[u]*spins[v])/2
                                      for u,v,d in self.graph.edges(data=True)))
        self.cut_value = -self.solution_cut
        if not np.isclose(self.solution_cut,self.residual_cut_with_offset,rtol=1e-10,atol=1e-10):
            raise RuntimeError('Restored original cut disagrees with reduced cut plus offset')

    def solve(self):
        self._reset()
        start = perf_counter()
        self._remove_isolates()
        while self.n_nodes > self.treshold:
            self._recursive_step()
            self._remove_isolates()
        self._solve_residual()
        self._restore_solution()
        self.steps_angs = np.asarray(self.steps_angs).reshape(-1,2)
        self.steps_angs_qaoa = np.asarray(self.steps_angs_qaoa).reshape(-1,2)
        self.steps_energies = np.asarray(self.steps_energies)
        self.steps_borders = np.asarray(self.steps_borders,dtype=int)
        self.energies = np.asarray(self.energies)
        self.rq_time = perf_counter()-start
        self.solved = True
        self.result = dict(solution=self.q_solution_dct.copy(),cut=self.solution_cut,
                           energy=self.cut_value,steps=self.step,time=self.rq_time)
        if self.callback: self._callback()
        if self.draw_process:
            self.draw_convegence()
            self.draw_param_evolution()
        return self

    def summary(self):
        return '\n'.join([
            'AnalyticalRQAOA | p=1 | weighted edge correlations',
            f'Graph          | original nodes: {self.original_n_nodes}, edges: {self.graph.number_of_edges()}',
            f'Search         | gamma window: {self.gamma_bounds}, points: {self.samples}; exact beta',
            f'Recursion      | steps: {self.step}, residual nodes: {self.n_nodes}, threshold: {self.treshold}',
            f'Solution cut   | {self.solution_cut}; package energy (-C): {self.cut_value}',
            f'Last QAOA <C>  | {self.q_energy} (not the final discrete cut)',
            f'Timing         | total: {self.rq_time:.4f} s; search: {self.q_time:.4f} s; residual: {self.c_time:.4f} s',
            'Optimality     | exact residual solve; original global optimum not certified',
            f'Status         | {"completed" if self.solved else "not solved"}',
        ])

    def _callback(self):
        print(self.summary())

    def __str__(self):
        return self.summary()

    def draw_convegence(self,show=True):
        """Original package spelling retained; negative expected cut by grid point."""
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots(figsize=(10,6))
        ax.plot(np.arange(len(self.energies)),self.energies,label='Expected -C (gamma grid scans)')
        for border in self.steps_borders[:-1]: ax.axvline(border,color='gray',alpha=.4)
        if self.cut_value is not None: ax.axhline(self.cut_value,ls='--',label='Final discrete -C')
        ax.set(xlabel='Gamma grid evaluations',ylabel='Original energy (-C)',title='AnalyticalRQAOA')
        ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
        if show: plt.show()
        return fig,ax

    def draw_param_evolution(self, show=True, annotate_steps=True,
                             figsize=(9, 6), cmap='viridis'):
        """Plot recorded (gamma, beta) points with directed step transitions.

        Uses gamma_history and beta_history (+C convention, radians).
        Colors and optional labels show step numbers. Angles are displayed
        exactly as recorded, without periodic unwrapping. Returns (fig, ax).
        Empty history and a single step are supported. show=False allows saving
        or further customization without opening a window.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.ticker import MaxNLocator

        fig, ax = plt.subplots(figsize=figsize)
        gammas = np.asarray(self.gamma_history, dtype=float)
        betas = np.asarray(self.beta_history, dtype=float)
        count = len(gammas)
        if count:
            steps = np.arange(1, count + 1)
            # For a single step, give the color scale a nonzero range.
            norm = Normalize(vmin=1 if count > 1 else 0.5,
                             vmax=count if count > 1 else 1.5)
            scatter = ax.scatter(gammas, betas, c=steps, cmap=cmap, norm=norm,
                                 s=65, edgecolors='black', linewidths=0.5, zorder=3)
            for i in range(count - 1):
                # Repeated points cannot have a visible transition arrow.
                if gammas[i] == gammas[i + 1] and betas[i] == betas[i + 1]:
                    continue
                ax.annotate('', xy=(gammas[i + 1], betas[i + 1]),
                            xytext=(gammas[i], betas[i]),
                            arrowprops=dict(arrowstyle='->', color='gray',
                                            alpha=0.65, lw=1.2,
                                            shrinkA=5, shrinkB=5), zorder=2)
            if annotate_steps:
                for step, gamma, beta in zip(steps, gammas, betas):
                    ax.annotate(str(step), (gamma, beta), xytext=(5, 6),
                                textcoords='offset points', fontsize=8)
            ax.scatter(gammas[0], betas[0], s=160, facecolors='none',
                       edgecolors='green', linewidths=1.6, label='First step', zorder=4)
            ax.scatter(gammas[-1], betas[-1], s=170, marker='s', facecolors='none',
                       edgecolors='red', linewidths=1.6, label='Last step', zorder=4)
            colorbar = fig.colorbar(scatter, ax=ax, label='Recursive step')
            if count == 1:
                colorbar.set_ticks([1])
            else:
                colorbar.locator = MaxNLocator(integer=True)
                colorbar.update_ticks()
            ax.legend()
            ax.margins(x=0.12, y=0.15)
        else:
            ax.text(0.5, 0.5, 'No recursive steps recorded',
                    ha='center', va='center', transform=ax.transAxes)
        ax.set(xlabel=r'$\gamma$ (+C), rad', ylabel=r'$\beta$, rad',
               title='AnalyticalRQAOA parameter evolution')
        ax.grid(alpha=0.3)
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax


if __name__ == '__main__':

    # G = RegularGraph(
    #     number_of_nodes=50,
    #     degree=10,
    #     weights="ones",
    #     # seed=1000,
    # )

    G = DensRandGraph(
        number_of_nodes=100,
        density=0.5,
        weights="ones",
        seed=1000,
    )

    rq = AnalyticalRQAOA(graph=G,
                         treshold=3,
                         n_layers=1,
                         samples=1000,
                         logs=True,
                         callback=True)
    rq.solve()
    rq.draw_param_evolution(annotate_steps=False)

