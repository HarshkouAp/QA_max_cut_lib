"""Analytical RQAOA for weighted MaxCut, p=1.

C = offset + sum(w_uv * (1 - Zu*Zv)/2).
State: exp(-i*beta*sum(X)) exp(-i*gamma*C) |+>^n.
Requires numpy, networkx; matplotlib for convergence, plotly for 3D HTML.
"""
from itertools import product
from pathlib import Path
from time import perf_counter

import networkx as nx
import numpy as np


class AnalyticalRQAOA:
    """Gamma grid + exact beta, correlation reduction, exact residual solve.

    samples: gamma grid size; gamma_batch: batch size; treshold: residual size.
    Set gamma_bounds for real weights; the default [0, pi] covers integer weights.
    n_layers=1 and max_exact_nodes are retained as compatibility arguments only.
    solution_cut is the final cut; q_energy is the last QAOA expectation.
    """
    def __init__(self, graph, treshold=3, n_layers=1, samples=1000,
                 logs=False, callback=False, draw_process=False,
                 gamma_bounds=None, gamma_batch=256, store_maps=False,
                 max_exact_nodes=20):
        self.graph = nx.Graph(graph)
        for _, _, data in self.graph.edges(data=True):
            data['weight'] = float(data.get('weight', 1))
        self.gamma_bounds = (0., np.pi) if gamma_bounds is None else tuple(map(float, gamma_bounds))
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
        self.param_evolution_html = None
        self.gammas = self.betas = self.profile = np.array([])
        self.coefficients = np.array([])

    def _edge_terms(self, u, v, gamma):
        """R_uv = S*sin(4*beta) + Q*sin(2*beta)**2."""
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
        """Last edge correlations; dense zeros outside edges mean not evaluated."""
        if not dense:
            return dict(self.edge_correlations)
        nodes = list(self.correlation_nodes)
        idx = {u:i for i,u in enumerate(nodes)}
        matrix = np.zeros((len(nodes),len(nodes)))
        for (u,v),r in self.edge_correlations.items():
            matrix[idx[u],idx[v]] = matrix[idx[v],idx[u]] = r
        return matrix,nodes

    def _reduce_graph(self, removed, retained, sign):
        """Fix z_removed = sign*z_retained; accumulate the constant in offset."""
        G = self.reduced_graph
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

        for u in list(nx.isolates(self.reduced_graph)):
            self.free_spins[u] = 1
            self.reduced_graph.remove_node(u)
        self.n_nodes = self.reduced_graph.number_of_nodes()

    def _recursive_step(self):
        start = perf_counter()
        self._optimize_angles()
        self._build_correlation_map()

        u,v = max(self.edge_correlations,key=lambda edge:abs(self.edge_correlations[edge]))
        corr = self.edge_correlations[(u,v)]

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
        best = -np.inf; best_spins = {}

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
                             figsize=(10, 8), cmap='viridis',
                             filename='rqaoa_parameter_trajectory.html'):
        """Save offline 3D HTML (gamma, beta, reduction step); return Plotly figure."""
        import plotly.graph_objects as go

        gammas = np.asarray(self.gamma_history, dtype=float)
        betas = np.asarray(self.beta_history, dtype=float)
        count = len(gammas)
        steps = np.arange(1, count + 1)
        fig = go.Figure()
        if count:
            fig.add_trace(go.Scatter3d(
                x=gammas, y=betas, z=steps,
                mode='lines+markers+text' if annotate_steps else 'lines+markers',
                text=[str(step) for step in steps] if annotate_steps else None,
                textposition='top center',
                line=dict(color='#64748b', width=4),
                marker=dict(size=5, color=steps, colorscale=cmap,
                            cmin=1 if count > 1 else 0.5,
                            cmax=count if count > 1 else 1.5,
                            colorbar=dict(title='Step', tickformat='d')),
                hovertemplate='Step %{z}<br>γ = %{x:.6f} rad<br>β = %{y:.6f} rad<extra></extra>',
                name='Trajectory', showlegend=False,
            ))
            for index, name, color, symbol in [
                (0, 'First step', '#16a34a', 'diamond'),
                (-1, 'Last step', '#dc2626', 'square'),
            ]:
                fig.add_trace(go.Scatter3d(
                    x=[gammas[index]], y=[betas[index]], z=[steps[index]],
                    mode='markers', name=name,
                    marker=dict(size=8, color=color, symbol=symbol),
                    hovertemplate='Step %{z}<br>γ = %{x:.6f} rad<br>β = %{y:.6f} rad<extra>%{fullData.name}</extra>',
                ))
        else:
            fig.add_annotation(text='No recursive steps recorded',
                               x=0.5, y=0.5, xref='paper', yref='paper', showarrow=False)

        fig.update_layout(
            title='AnalyticalRQAOA parameter trajectory', template='plotly_white',
            height=int(figsize[1] * 100), autosize=True,
            margin=dict(l=0, r=0, t=70, b=0),
            legend=dict(x=0.01, y=0.99),
            scene=dict(
                xaxis_title='γ (+C), rad', yaxis_title='β, rad',
                zaxis=dict(title='Reduction step', tickmode='linear',
                           tick0=1, dtick=max(1, (count + 9) // 10),
                           range=[0.5, count + 0.5] if count else [0, 1]),
                aspectmode='manual', aspectratio=dict(x=1, y=1, z=1.3),
                camera=dict(eye=dict(x=1.6, y=1.6, z=1.0)),
            ),
        )
        path = Path(filename).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(path), include_plotlyjs=True, full_html=True,
                       auto_open=show,
                       config=dict(responsive=True, scrollZoom=True, displaylogo=False))
        self.param_evolution_html = str(path)
        return fig


if __name__ == '__main__':
    G = nx.gnp_random_graph(100, 0.5, seed=1000)
    rq = AnalyticalRQAOA(G, treshold=3, samples=1000, logs=True, callback=True)
    rq.solve()
    rq.draw_param_evolution(annotate_steps=False)
