# Mathematical conventions

This page defines the quantities the library computes: the cut weight, the energy,
the Ising form used by reductions, the QAOA state and objectives, the estimators,
the reduction offsets, and the diagnostics built on top of them. Every formula
states what is substituted and what it returns, and how to read its sign. The same
symbols are used in `API.md`, in the docstrings, and in the acceptance records.

Read the first three sections before comparing numbers from different solvers.
The remaining sections are independent; each one starts from the working graph of
its method.

## Symbols

| Symbol | Meaning | Notes |
|---|---|---|
| `G=(V,E)`, `w_ij` | simple undirected graph, finite real edge weight | the unweighted case is written `w=1` |
| `s_i ∈ {-1,+1}` | spin of vertex `i` | `s_i=1-2b_i` for bit `b_i` |
| `C(s)` | cut weight | maximized |
| `E(s)` | energy, `E=-C` | minimized |
| `J_ij`, `h_i`, `c` | Ising pair coefficient, field, constant | internal representation |
| `b_t` | accumulated Max-Cut offset after `t` reductions | different from `c_t` |
| `σ` | constraint sign, `s_u=σ s_v` | `±1` |
| `ρ_t` | physical ZZ correlation of the selected pair | not a probability |
| `P_θ(z)` | Born probability of basis state `z` | `|ψ_θ(z)|²` |
| `F` | distribution objective | maximized; the optimizer minimizes `L=-F` |
| `ρ` | density matrix | only in the mutual-information section |

A correlation `ρ_t` and a density matrix `ρ` are different objects. The density
matrix appears only where the text says so.

## Weighted Max-Cut and energy

Let `G=(V,E)` be a simple undirected graph, `w_ij` a finite real weight, and
`s_i ∈ {-1,+1}`. For every algorithm the library defines

\[
C(s)=\sum_{(i,j)\in E}w_{ij}\frac{1-s_i s_j}{2},\qquad
E(s)=-C(s)=\sum_{(i,j)\in E}\frac{w_{ij}}2(s_i s_j-1).
\]

Minimizing `E` is equivalent to maximizing `C`. The constant part `-Σw/2` is
retained. The absolute value of the energy is not used. A negative edge inside the
cut contributes negatively to `C` and positively to `E`. The unweighted case is
written explicitly as `w=1`.

Array indices are defined by `node_order`, not by the numeric value of a vertex
label. For a basis index `z`, bit `i` belongs to `node_order[i]` and
`s_i = 1 - 2 bit_i(z)`. This is little-endian index order. In a probability
diagram the bit strings are reversed so that the **leftmost character is the first
vertex**.

## Ising form and exact pair reduction

The internal problem is stored as

\[
E(s)=c+\sum_i h_i s_i+\sum_{i<j}J_{ij}s_i s_j,
\]

where each undirected pair appears once. For a Max-Cut graph without fields

\[
J_{ij}=\frac{w_{ij}}2,\qquad h_i=0,\qquad
c=-\frac12\sum_{(i,j)\in E}w_{ij}.
\]

Therefore `IsingProblem.energy(s) = -C_G(s)` for every assignment. Zero pairs are
not stored, but the input graph may contain explicit zero edges.

Under the constraint \(s_u=\sigma s_v\), \(\sigma\in\{-1,+1\}\), for \(k\ne u,v\)

\[
J'_{vk}=J_{vk}+\sigma J_{uk},\qquad
h'_v=h_v+\sigma h_u,\qquad
c'=c+\sigma J_{uv}.
\]

Vertex \(u\) and all its old interactions are removed. The term \(J_{uv}s_us_v\)
becomes the constant \(\sigma J_{uv}\), because \(s_u s_v=\sigma s_v^2=\sigma\).
The field \(h_us_u\) becomes \(\sigma h_us_v\), so the retained vertex receives
\(\sigma h_u\). The other terms are unchanged. For **every** assignment of the
remaining spins

\[
E_{\text{before}}(s_u=\sigma s_v,s')=E_{\text{after}}(s').
\]

This equality holds on the constraint subspace. It does not claim that an
arbitrary selected pair preserves the global optimum of the original problem.

For a Max-Cut origin it is convenient to keep the working graph \(G_t\) and the
offset \(b_t\):

\[
C_{G_0}(s_{\text{recovered}})=b_t+C_{G_t}(s'),\qquad
E_{G_0}(s_{\text{recovered}})=-b_t-C_{G_t}(s').
\]

The initial value is \(b_0=0\). For the same reduction

\[
w'_{vk}=w_{vk}+\sigma w_{uk},\qquad
\Delta b=\frac{1-\sigma}{2}\sum_{k\in N_t(u)}w_{uk}.
\]

The sum runs over the current working graph before the transformation and
includes the edge \(u\)-\(v\) when it exists. The full constant of the internal
Ising problem is related to the offset by

\[
c_t=-b_t-\frac12\sum_{e\in E(G_t)}w_e.
\]

Therefore `cut_offset` and `ising_constant` are different quantities. For the
selected edge

\[
e_t^{\text{fixed}}=-\frac{w_t}{2}(1-\sigma_t),\qquad
\overline e_t=-\frac{w_t}{2}(1-\rho_t),
\]

where \(\rho_t=\langle Z_uZ_v\rangle\) is the physical correlation, not a
probability. The probability of the constraint is \((1+\sigma_t \rho_t)/2\). The
value \(\overline e_t\) is not multiplied by that probability a second time. In
the control example `fixed_energy`, `cut_offset_delta`, and
`ising_constant_delta` equal -2, 5, and -1 respectively.

Constraints are stored in application order and restored in reverse order:
\(s_{\text{removed}}=\sigma s_{\text{retained}}\). If the retained vertex was
removed later, the reverse pass already knows its spin. Isolated vertices stay in
the remainder and receive a spin from the remainder solution. For an empty input
graph the restoration returns an empty dictionary.

## Exact enumeration (`ExactCPU`)

The public problem has no local fields. Therefore `E(s)=E(-s)`, and for `N>0` the
first spin can be fixed to `+1`. The algorithm enumerates `2^(N-1)`
representatives, finds the minimum and maximum of `C`, then makes a second pass to
count the number of maxima relative to the final `C*`. Doubling that count
restores the degeneracy caused by the global spin flip. An empty graph has a
single empty solution.

`tie_tolerance` is used only when counting degeneracy; the maximum is selected by
the largest computed `C`. Time is `O(|E|2^N)`, and the working-vector memory is
`O(batch_size)`. "Exact" means complete enumeration, not symbolic arithmetic:
weights and sums are `float64`. Strong cancellation of large positive and
negative weights can affect the comparison of close values. The final `C` of the
solution is additionally verified by direct summation on the original graph
snapshot.

## Statevector QAOA (`QAOACPU`)

The diagonal Hamiltonian `H_E` has eigenvalues `E(z)=-C(z)`. The mixer is
`B=Σ_i X_i`, and the initial state is `|+>^N`. For layer `ℓ` the phase operator
`U_E(γℓ)=exp(-iγℓH_E)=exp(+iγℓC)` is applied first, then the mixer
`U_B(βℓ)=exp(-iβℓB)`. The first layer has index 0. The parameter vector is stored
as `[γ₀,…,γₚ₋₁,β₀,…,βₚ₋₁]`.

The mixer application is split into independent two-element transformations
because the operators `X_i` commute. Both new elements are computed from the old
pair, which prevents use of an already modified amplitude. The full vector holds
`2^N` complex numbers; compression by spin inversion is not applied. The norm is
checked, then only the accumulated normalization error is removed. The scope is
small problems and future subproblems of composite algorithms. The ability to
create a 1000-vertex graph does not imply the ability to simulate its full quantum
statevector.

The circuit follows [Farhi, Goldstone, Gutmann, 2014](https://arxiv.org/abs/1411.4028).
Because `H_E=-C` is chosen here, the sign of `γ` is opposite to formulations that
use `exp(-iγC)`. For a single edge with `w=1`, `p=1`:

\[
\langle C\rangle=\frac{1-\sin(\gamma)\sin(4\beta)}2.
\]

With `γ=-π/2` and `β=π/8` this gives `⟨C⟩=1`, `⟨E⟩=-1`, `⟨Z₀Z₁⟩=-1`. This case and
an independent dense matrix product for `p=1,2` are part of the checks.

## Distribution objectives

Let `Pθ(z)=|ψθ(z)|²`, let the values of `C` be sorted in ascending order as `cᵣ`,
and let their masses be `pᵣ`. Every objective `F` below is **maximized**. The
optimizer receives `L=-F`. For Mean Cut `L=⟨E⟩`; for the other objectives `L` is
not the ordinary expectation of the physical energy.

| `type` | `F` and parameters | Units |
|---|---|---|
| `mean_cut` | `Σ pᵣcᵣ` | cut weight |
| `cvar` | mean `C` over the best `α` fraction of probability mass; `0<α≤1` | cut weight |
| `gibbs` | `log(Σ pᵣ exp(ηcᵣ))/η`; `η>0` | cut weight; `η` has inverse units |
| `best_of_k` | `E[max(C₁,…,C_K)]` for `K` independent samples; integer `K≥1` | cut weight |
| `threshold_probability` | `P(C≥A)`, parameter `threshold=A` | dimensionless probability |

CVaR takes exactly the mass `α`, including the fractional part of the boundary
state, not an integer number of best states. In energy notation it is the negative
of the lower-tail CVaR of the energy. The justification for applying CVaR to
variational optimization is [Barkoutsos et al., 2020](https://quantum-journal.org/papers/q-2020-04-20-256/).

Best-of-K uses the CDF: `P(max≤cᵣ)=F_C(cᵣ)^K`, so the mass of the maximum is the
difference of adjacent values of that power. Repeated values of `c` are allowed.
Gibbs is evaluated with `logsumexp` so that large values are not exponentiated
directly. Threshold Probability uses an absolute threshold on `C`, which is
equivalent to `E≤-A`; it is not a threshold on normalized quality.

## Estimation, gradient, readout, and shot budget

In `exact` mode `F` is computed from the full `Pθ`. In `shots` mode each estimate
draws `M` independent samples, builds an empirical mass, and applies the same `F`
to it. For nonlinear objectives this is a plugin estimate; it need not be
unbiased. The gradient uses central differences
`[L(θ+heᵢ)-L(θ-heᵢ)]/(2h)`, with independent samples for each probe vector. With
`d=2p`, one full step requires `2d+1` evaluations, including the center. A small
`h` under shot estimation can amplify noise.

ADAM is then applied with bias correction of the first and second moments,
following [Kingma, Ba, 2014](https://arxiv.org/abs/1412.6980). The update of the
last step is not necessarily evaluated: readout uses the best **already
evaluated** vector, including gradient probe points. `optimizer_history` stores
the step centers; `objective_history`, when enabled, stores every evaluation.
Their best elements can differ.

`most_probable` selects the index with the largest exact probability; on a tie the
first index is selected. It spends no measurement shots and is available here
because a classical simulator is used. `best_of_shots` performs exactly `M` final
samples and selects the largest `C` among them. This is a separate operation from
the training shots.

The entire final readout is reserved before training starts. A new gradient step
starts only if the budget allows that step to be completed while the reserve is
kept. Therefore `shots_used≤limits.shots_total`. If only the readout fits, no
training is performed, `objective_value=None`, the initial angles are kept, and a
solution is still returned. An insufficient budget for the readout itself is
rejected before any computation.

`expected_cut` and `expected_energy` are exact expectations of the final simulated
state even when training used shots; the origin is recorded as
`expected_cut_source="exact_final_statevector"`. `objective_value` is the best
training estimate; in `shots` mode it is a random estimate selected among many
probes. It must not be presented as an independent exact measure of quality.

## Correlations and comparison

`correlation_matrix` returns the physical `Mᵢⱼ=⟨ZᵢZⱼ⟩=1-2P(bitᵢ≠bitⱼ)`, with
`Mᵢᵢ=1`. A correlation can be negative and is not a probability:
`P(sᵢ=sⱼ)=(1+Mᵢⱼ)/2`.

When `C*` and `C_min` are known for the same problem, `quality_score` computes
`q=(C-C_min)/(C*-C_min)`. If the denominator is zero it returns `None`. This is a
convenient normalization for signed weights; it is not the Goemans-Williamson
approximation ratio. An unknown exact optimum must not be replaced by this score
when comparing against a heuristic reference.

## RQAOA on CPU

RQAOA builds a sequence of constraints. At step \(t\) the solver trains QAOA on
the current working graph \(G_t\) and obtains the physical correlations

\[
c_{uv}=\langle Z_uZ_v\rangle
=\sum_x P_t(x)(1-2x_u)(1-2x_v),
\]

then selects a pair \((u,v)\) and fixes \(s_u=\sigma s_v\). In the base selector
the pair maximizes \(|c_{uv}|\); on a tie the lexicographically first pair of
external labels is used. By default the smaller label is retained and the larger
is removed. The sign is \(\sigma=+1\) for \(c>0\) and \(\sigma=-1\) for \(c<0\); an
exact zero uses `zero_sign`. The correlation is not a probability:
\(P(s_u=\sigma s_v)=(1+\sigma c_{uv})/2\).

The constraint is applied by the exact reduction core. For a Max-Cut origin

\[
C_{G_0}(s_{\text{recovered}})=b_t+C_{G_t}(s'),
\qquad
E_{G_0}(s_{\text{recovered}})=-b_t+E_{G_t}(s').
\]

The full constant of the internal Ising problem is

\[
c_t^{\text{Ising}}=-b_t-\frac12\sum_{e\in E(G_t)}w_e.
\]

`cut_offset`, `ising_constant`, and the contribution of the selected edge are
different quantities. For the selected edge

\[
e_t^{\text{fixed}}=-\frac{w_t}{2}(1-\sigma_t),
\qquad
\overline e_t=-\frac{w_t}{2}(1-\rho_t).
\]

Both values are unnormalized and are summed over steps as an event log. They are
not the full energy of the recovered solution. If the edge is absent, \(w_t=0\);
the mean modulus over an empty set is `None`, not zero.

QAOA training uses the same circuit as `QAOACPU`: \(H_E=-C\), phase
\(\exp(+i\gamma C)\), angle order `γ₀…γₚ₋₁,β₀…βₚ₋₁`. The source returns all pairs
of active vertices; with `correlations.mode="shots"` one sample of `M` shots is
drawn once and yields all ZZ correlations. The shared budget belongs to the root
run; the step's correlation reserve is protected before training, and the exact
remainder spends no shots.

`objective_reference="original_graph"` means that the objective refers to the
distribution of the original cut after the constraints are restored. For Mean
Cut, CVaR, Gibbs, and Best-of-K the offset \(b_t\) adds a constant and does not
change the optimizing angles. For Threshold Probability

\[
P(C_{\text{original}}\ge A)=P(C_t\ge A-b_t).
\]

Therefore both the requested and the actually used thresholds are stored.

When the run stops on `max_reductions` or on an insufficient shot budget, a full
feasible solution is returned: active spins are set to \(+1\), removed spins are
restored, and the cut value is computed on the original graph. This is
`all_plus_reduced_fallback`, not the result of exactly solving a large remainder.
An exact remainder does not prove global optimality of RQAOA; the control example
returns \(C=7\) against a global optimum of \(8\).

## Analytic depth one

For one layer, the standard `|+>` state, and the X mixer, the mean observables can
be computed without a statevector. The internal matrix uses `J=w/2`, while the
correlator formula contains `gamma*w`. For distinct `u,v`, define products over
all `k≠u,v`:

\[
P_{u\setminus v}=\prod_{k\ne u,v}\cos(\gamma w_{uk}),\qquad
P_{v\setminus u}=\prod_{k\ne u,v}\cos(\gamma w_{vk}),
\]

\[
P^-_{uv}=\prod_{k\ne u,v}\cos(\gamma(w_{uk}-w_{vk})),\qquad
P^+_{uv}=\prod_{k\ne u,v}\cos(\gamma(w_{uk}+w_{vk})).
\]

An empty product equals 1. Then

\[
a_{uv}=\frac12\sin(\gamma w_{uv})(P_{u\setminus v}+P_{v\setminus u}),\qquad
d_{uv}=\frac12(P^-_{uv}-P^+_{uv}),
\]

\[
c_{uv}=a_{uv}\sin(4\beta)+d_{uv}\sin^2(2\beta).
\]

The sign of the first term is positive under the adopted `E=-C`. In a historical
record that used the phase `+C` it is opposite; the conversion is
`gamma_old=-gamma_new`. The second term is even in `gamma`. For a non-adjacent
pair `w_uv=0` the first term is zero, but `d_uv` can be nonzero because of common
neighbours.

Wave derivation: the X mixer maps `Z_uZ_v` into a combination of `ZZ`, `ZY+YZ`,
and `YY`. Conjugation by the diagonal phase leaves products of cosines; the mixed
term yields `a_uv`, and common neighbours in `YY` yield `d_uv`. The global phase of
the Hamiltonian constant cancels. The formulas agree with Ozaeta, van Dam,
McMahon, Quantum Sci. Technol. 7, 045036 (2022), with `J=w/2` and the chosen signs
of the unitary operators.

The mean energy over the existing edges is

\[
E(\gamma,\beta)=E_0+A\sin(4\beta)+B\sin^2(2\beta),
\]

\[
A=\frac12\sum_{u<v}w_{uv}a_{uv},\qquad
B=\frac12\sum_{u<v}w_{uv}d_{uv},\qquad
E_0=-\frac12\sum_{u<v}w_{uv}.
\]

Write `F0=E0+B/2`, `Fs=A`, `Fc=-B/2`. Then
`E=F0+Fs sin(4β)+Fc cos(4β)` and

\[
\min_\beta E=F_0-\operatorname{hypot}(F_s,F_c),\qquad
\beta_*=\frac14\operatorname{atan2}(-F_s,-F_c)\bmod\frac{\pi}{2}.
\]

When `R=0` all beta values are equivalent; 0 is selected deterministically. The
search over gamma uses an explicit finite grid with both bounds included. The
accuracy of the analytic moments and of the beta minimum is limited by machine
arithmetic; the minimum over gamma refers only to the scanned grid.

The cost of one gamma pass with `K` requested pairs is `O(KN)` in the dense
representation, `O(N^3)` for all pairs; preparing the `N×N` weights costs `O(N^2)`.
Memory is bounded by the weight matrix and a batch of `O(pair_batch_size*N)`. No
`2**N` object is created. The analytic source does not provide a full
distribution, shots, CVaR, Gibbs, Best-of-K, or Threshold Probability; such
requests are rejected. The scalar offset `b_t` adds a constant to the energy and
does not change the angles or the physical correlations; for
`objective_reference="original_graph"` the energy profile is shifted by `-b_t`.

## Local tasks in LS-RQAOA

For the current graph `G_t` and a vertex set `S_a`, the local task is the induced
subgraph

\[
G_{t,a}=G_t[S_a],\qquad
E_{t,a}(s)=-\sum_{\{u,v\}\subseteq S_a}w^{(t)}_{uv}\frac{1-s_us_v}{2}.
\]

Boundary edges with one end outside `S_a` are excluded. This is the
`induced_no_fields` policy: a local heuristic, not an exact decomposition of the
full Hamiltonian. No local fields are added, unknown boundary spins are not
replaced by averages, and discarded edges are not converted into a constant.

Each task trains its own source and returns \(c^{(a)}_{uv}\). For a pair `u,v` the
base aggregator `mean` uses the observations of that pair:

\[
\overline c_{uv}=\frac{1}{|I_{uv}|}\sum_{a\in I_{uv}}c^{(a)}_{uv}.
\]

The sum runs in canonical task-ID order; true zeros enter both the numerator and
the denominator. A missing pair is not equal to zero. The base score is
`abs(c_bar)`, followed by lexicographic tie-breaking. The sign of the constraint
equals the sign of `c_bar`; for zero, `zero_sign` is used.

For the selected pair with weight `w_t` and sign `sigma_t`:

\[
e_t^{\mathrm{fixed}}=-\frac{w_t}{2}(1-\sigma_t),\qquad
\overline e_t^{\mathrm{local}}=-\frac{w_t}{2}(1-\overline c_t),
\]

and the probability of the local constraint is `(1+sigma*c_bar)/2`. This is the
expected local contribution of the selected pair, not the global energy of the
full graph. Different pairs can come from different mixtures of local states; the
sum of local expectations is not declared a global `expected_energy`. The final
discrete energy is always computed on the original graph.

The global constraint \(s_{\mathrm{removed}}=\sigma s_{\mathrm{retained}}\) is
applied to the whole `G_t`, not only to the selected subgraph. Therefore links from
the removed vertex to vertices outside the local task are updated as well, and the
energy invariant of the reduction core is preserved.

## GPU pair representation

For a standard Max-Cut without fields, the amplitudes of globally inverted rows
are equal. For `N>=1` and `D=2^(N-1)` the representative fixes the last bit of
`node_order[N-1]` to zero:

\[
\chi_r=\sqrt2\,\psi_r,\qquad q_r=|\chi_r|^2=P(r)+P(\bar r).
\]

Restoration is `concatenate([chi, chi[::-1]])/sqrt(2)`. For pair ZZ observables
and `C(r)` the masses `q` are used directly; the physical probability of one row
is `q/2`. The last physical qubit in the pair mixer maps `r` to `r xor (D-1)`,
and for `N=1` it contributes the global phase `exp(-i beta)`.

The exact Mean Cut adjoint computes the derivative of `-<C>` by a reverse pass
over the state. All objectives are evaluated on the device, including CVaR with
partial boundary mass, Gibbs with a stable exponential, Best-of-K through the CDF,
and Threshold Probability. Shot objectives use the empirical values of a real
sample, not a fictitious histogram of size `2**N`.

## Classical baselines and the conditional block

GW solves the SDP

\[
\max_{X\succeq0,\ \mathrm{diag}(X)=1}
\frac12\sum_{(u,v)}w_{uv}(1-X_{uv}).
\]

The optimal value of the relaxation is an upper bound on \(C^*\), but the numeric
`sdp_objective_raw` after an approximate SCS solve and spectral repair is not
declared a certificate. Rounding uses independent normal vectors: \(s_u=+1\) if
\((Vg)_u\ge0\), otherwise \(-1\).

SA uses the spin BQM

\[
E=c+\sum_{u<v}J_{uv}s_us_v,\quad J_{uv}=w_{uv}/2,\quad
c=-\frac12\sum_e w_e,
\]

and returns the best final read plus the all-plus incumbent. `num_reads` is not a
number of quantum shots.

For a block \(S\) with fixed external spins \(t\),

\[
h_u^{(S,t)}=\frac12\sum_{v\notin S}w_{uv}t_v,
\]

\[
c_S(t)=-C_{G[V\setminus S]}(t)
-\frac12\sum_{\{u,v\}\cap S\ne\varnothing}w_{uv}.
\]

The conditional energy is

\[
E_G(x,t)=c_S(t)+\sum_{u<v\in S}\frac{w_{uv}}2x_ux_v+
\sum_{u\in S}h_u^{(S,t)}x_u.
\]

All \(2^{|S|}\) internal assignments are enumerated; the external spins are fixed
until the end of the block. An update is accepted only on strict improvement

\[
\Delta C>\varepsilon_{\mathrm{abs}}+\varepsilon_{\mathrm{rel}}
\max(1,|C_{\mathrm{candidate}}|,|C_{\mathrm{current}}|).
\]

Blocks are applied sequentially; independent local gains must not be summed. The
accuracy of each fully enumerated block refers to its environment, not to the
global optimum.

## Block hierarchy

The current graph \(G_l\) is partitioned into disjoint blocks \(S_a\) that cover
all its vertices. A local pattern \(r_u\in\{-1,1\}\) fixes the internal products,
and the anchor \(a=\min S_a\) is normalized as \(r_a=+1\). One free variable
\(\tau_a\) per block defines \(s_u=r_u\tau_a\).

For inter-block edges

\[
W_{ab}=\sum_{u\in S_a,v\in S_b}w_{uv}r_ur_v,
\qquad
\Delta b_l=\sum_{(u,v)\in E_l}w_{uv}\frac{1-r_ur_v}{2}.
\]

Then

\[
C_{G_l}(r\tau)=\Delta b_l+C_{G_{l+1}}(\tau),\qquad
C_{G_0}=b_l+C_{G_l},\qquad b_{l+1}=b_l+\Delta b_l.
\]

`Delta b` includes both internal and boundary edges. A term that counts only local
cuts does not give the correct offset. For an internal edge \(a=b\) the second
term of the decomposition is zero.

The block projection is implemented with the same core constraints
\(s_u=r_u s_a\), not with the second reduction formula. `W`, `Delta b`, and
`c_{l+1}=-b_{l+1}-\frac12\sum W_{ab}` serve as an independent check. The full
Ising constant is not equal to `-b` and is not added a second time.

The local QAOA depth \(p\) and the number of hierarchy levels \(L\) are different
quantities. A correct projection does not guarantee a global optimum: the local
pattern restricts the solution space. `hierarchy_all_plus_fallback` returns a full
feasible answer at a normal limit, but it is not claimed to be optimal.

## Landscapes

`scan_landscape` computes the exact `p=1` grid: rows correspond to β, columns to γ.
By default it stores `optimizer_loss`, that is `⟨E⟩` for Mean Cut. The alternative
`quantity="objective_value"` stores `F`. The γ and β windows are given explicitly:
for arbitrary real weights no universal period in γ is assumed. Computing the map
does not start an optimization and does not change the solver result.

## Series metrics and the parallel profile

This section applies to the series runner and the parallelism profiler introduced
in `0.1.0a9`. For each trial the quality is recomputed on the full assignment of
the original graph:

.. math::

   C(s)=\sum_{(u,v)\in E} w_{uv}\frac{1-s_us_v}{2},
   \qquad E(s)=-C(s).

An exact reference provides `C_star`, so the absolute gap is `C_star-C`. For
GW/best-known the gap is `C-C_ref`; a positive difference is not an optimization
error. The ratio `C/C_star` is defined only for a positive proven `C_star`. The
range normalization

.. math::

   q_{\mathrm{range}}=\frac{C-C_{\min}}{C_{\max}-C_{\min}}

requires both proven extremes. The minimum is obtained by a separate exact run on
the graph with weights `-w`: `C_min(G)=-C_max(-G)`.

For one case `q` and a positive median `T_q(n)`:

.. math::

   S_q(n)=\frac{T_q(1)}{T_q(n)},\qquad
   \eta_q(n)=\frac{S_q(n)}n,\qquad
   f_{\mathrm{eff}}(n)=\frac{1/S_q(n)-1/n}{1-1/n}.

The fixed-work fit uses

.. math::

   T_q(n)\approx T_q(1)\left(f_s+\frac{1-f_s}{n}\right).

An estimate of `f_s` at the 0/1 boundary is not proof of perfect parallelism or of
strict sequentiality. The global recommendation minimizes

.. math::

   \mathcal T(n)=\sum_{q\in Q}m_qT_q(n)

only over candidates with full coverage of `Q`; missing, failed, and
`work_mismatch` cells are not replaced by zero. Paired-block bootstrap resampling
uses blocks of repeats, not independent timing rows.

## Checkpoint and resource ledger

An ADAM checkpoint stores `theta`, the full `m`, `v`, the best evaluation, the
iteration number, the RNG state, and the resource prefix. The boundary is a
completed full iteration: `4p+1` central differences, then the convergence check,
then the atomic update of `m/v/theta`. The values of
`optimizer_iterations_completed`, `adam_updates_completed`, and
`objective_evaluations` differ.

For the conservative ledger:

.. math::

   P_{\mathrm{charge}}=P_{\mathrm{known}}+U_{\mathrm{charge}},
   \qquad
   P_{\mathrm{charge}}+R_{\mathrm{held}}+A_{\mathrm{available}}=B.

`U_charge` is the upper bound of an unfinished operation, `R_held` is the set of
active reserves only. For a pause after two iterations with `B=704`, `M=64`, this
gives `P=320`, `R=64`, `A=320`. If an attempt is lost, the upper value `160`
remains an uncertain charge; repeated reconciliation does not add it twice.

The readout token is reserved once. A completed readout checkpoint returns the
stored assignment without new `M` measurements; the unknown result of a lost
attempt is not counted as zero cost.

## Warm-start product QAOA

For a seed `s_i^(0) = +/-1` the probability of bit `1` is `q_i=(1-s_i)/2`.
Regularization is `q~_i=clip(q_i, eps, 1-eps)`. For explicitly given fractional
`q_i` this is only a clip, without an affine transformation. The preparation
angles are `alpha_i=2 asin sqrt(q~_i)`. They are fixed and are not part of the ADAM
dimension.

```
U_aligned(beta)  = RY(alpha) RZ(-2 beta) RY(-alpha)
U_modified(beta) = RY(-alpha) RZ(-2 beta) RY(alpha)
U_x(beta)        = exp(-i beta X)
```

The cost layer is still `exp(+i gamma C)`, followed by the chosen mixer. With
`eps=1/2`, modified coincides with the ordinary X mixer at `beta`, and aligned at
`-beta`. With `eps=0`, aligned and modified keep the original basis state with a
phase; this does not hold for `x`.

The source cut and the product-state `E[C]` are different quantities:
`source_cut=C(s_seed)`, `initial_expected_cut=C^T P0`. A full budget with `p=1`,
`max_steps=3`, `M_train=32`, and `M_readout=64` equals 544 shots.

## Warm-start RQAOA ensemble

On each current graph `G_t` a finite batch of roundings or explicit assignments is
selected. After global-flip deduplication the unique starts are sorted by `C_t`;
multiplicity does not change the weight. For `K_t` children:

```
P_bar(b) = 1/K_t sum_l |<b|psi_l>|^2
c_bar_uv = 1/K_t sum_l <Z_u Z_v>_l
```

This is a statistical mixture; `rho_t` is not materialized and there is no global
pure statevector. Physical correlations are mixed before pair selection. One
reducer applies `s_u = sigma s_v`; for `b_t`:

```
Delta b = (1-sigma)/2 * sum_{k in N(u)} w_uk
E_original = -b_t - C_t
```

For the budget `R_corr=K_t M_corr`, `a=(4p+1)M_train`: training quotas are
distributed round-robin in canonical child order; `K_t` is not reduced because of
a shortage of training resources. Control A: A=(+,+,-,+), B=(+,-,+,+), U=2, A has
multiplicity 3; the averaged correlations select pair (30,80) with `sigma=-1`,
`b=8`, `C=8`, `E_Ising=-7.5`.

## Conditional fields and environment LS

For a current graph `G_t`, an active block `S`, and fixed external spins `r_T`:

\[
A_S(x;r_T)=C_{SS}(x)+C_{ST}(x,r_T),
\qquad
E_{\mathrm{aff}}=-A_S=c_{\mathrm{aff}}+\sum_{i\in S}h_i x_i+
\sum_{\{i,k\}\subset S}J_{ik}x_ix_k,
\]

with `J_ik=w_ik/2` and `h_i=1/2 sum_{j in T} w_ij r_j`. The three constants
`c_aff`, `C_TT(r_T)`, and the accumulated reduction offset `b_t` are distinct. The
active state is a full `2^{|S|}` CPU state; external spins are classical data, not
an X-mixed subsystem. Local fields never enter the global reducer.

## Block mean-field exchange

For a disjoint block cover and compact first and internal moments, the product
model has

\[
\mathcal C(m,q)=\frac12\sum_{a(i)=a(j)}w_{ij}(1-q_{ij})+
\frac12\sum_{a(i)\ne a(j)}w_{ij}(1-m_im_j).
\]

For one block, `h_i=1/2 sum_{j outside} w_ij m_j`, `J_ik=w_ik/2`, and
`c_aff=-1/2(sum internal w + sum cross w)`. A local full CPU QAOA state is trained
with the shared trainer. After angle damping, one moment operation returns all `Z`
and internal `ZZ`; both `m` and `q` then use the same convex damping. Gauss-Seidel
commits each block immediately; Jacobi commits one complete sweep at a barrier.
The public answer is sign-rounded messages and is checked by a full cut on the
original graph.

## Mutual information

For a pair `(i,j)` and reduced density matrix `ρ_ij`:

\[
I_Q(i:j)=S(\rho_i)+S(\rho_j)-S(\rho_{ij}),\qquad
S(\rho)=-\sum_{\lambda>0}\lambda\log_b\lambda.
\]

Here `ρ` is a density matrix, not the ZZ correlation of the reduction sections.
By default `b=2` (bits); the pair matrix has rows `00,01,10,11` with the first
label as the most significant index of the small matrix. The partial trace is
built as `ρ_A=M M†`, where `M` is the coefficient matrix; the global `D×D` matrix
is not stored.

The classical Z-measure is

\[
I_Z(i:j)=\sum_{a,c}p_{ac}\log_b\frac{p_{ac}}{p_a p_c}.
\]

`legacy_half_nats = 0.5 I_Q(nats)` is a separate derived view. For a pure product
state `I_Q=I_Z=0`; for a Bell pair `I_Q=2`, `I_Z=1`, and the Φ and Ψ states have
the same mutual information and opposite ZZ sign. CZ-plus has `I_Q=2` but
`I_Z=0`: its `P_Z` contains no coherences.

## Incremental LS-RQAOA

An epoch on `G_t` performs a full barrier of local tasks. For a pair `e`:

\[
\widehat c^{new}_{t,e}
=\frac{\sum_{a\in\mathcal O_{t,e}}c^{(a)}_{t,e}}{d_{t,e}},
\qquad d_{t,e}=|\mathcal O_{t,e}|.
\]

The sums are cleared once before the epoch; rounds enter the same sum.
`age_t = t-observed_epoch`. `keep_unobserved` retains an unobserved pair only while
`age_t <= max_age_steps`; a new observation replaces the old one, including a true
zero. The score is computed separately from the current weight:

\[
S_{abs}=|c|,\quad S_{w1}=|w||c|,\quad S_{w2}=|w|^2|c|,\quad
S_{prod}=wc.
\]

For `min_signed_edge_product` the minimum of `S_prod` is selected, but
\(\sigma=\operatorname{sign}(c)\), not `sign(wc)`. The default reduction direction
is to remove the smaller label and keep the larger.

## Typed plans and coverage

For a full edge partition `{E_a}` and one global assignment `s`:

\[
C_G(s)=\sum_a C_{(S_a,E_a)}(s|_{S_a}).
\]

This is an identity of the edge decomposition of the objective, but independently
optimized local assignments on shared vertices need not agree. `edge_partition`
stores an explicit `E_a`; induced membership `S_a` does not reconstruct this task
when unassigned edges are added from `G[S_a]`.

Coverage of edges is counted over unique parent edges: \(|\bigcup_a E_a|/|E|\).
Absolute-weight coverage is \(\sum_{e\in covered}|w_e|/\sum_{e\in E}|w_e|\). The
signed total is published separately; it is not a valid denominator.

The hop metric uses BFS by edge count:

\[
L_{\rm reachable}=\frac{\sum_{\{u,v\}\in R}d(u,v)}{|R|},\quad
E_{\rm global}=\frac1T\sum_{\{u,v\}\in R}\frac1{d(u,v)}.
\]

For a disconnected graph the all-pairs mean is `null`; the reachable mean and the
reachability fraction remain separate fields.
