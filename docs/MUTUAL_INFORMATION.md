# Mutual information: quantum, and classical in the Z basis

Two different numbers are both called "mutual information" in this library, and
confusing them changes the conclusion. `QAOAWithMIObserverCPU`
(`type="laboratory.qaoa_mi_cpu"`) and `RQAOAWithMIObserverCPU`
(`type="laboratory.rqaoa_mi_cpu"`) wrap an ordinary run with a passive observer that
reports both. Given a final state or a set of final angles, the observer returns a
table of pair values, the one-qubit entropies behind them, and the snapshot the
values belong to. This page defines the two measures, shows a state where they
disagree completely, and fixes the ordering convention of the small density matrix.

Neither measure is a property of the graph. Both are properties of the state the
solver produced.

## The two measures

For a pair of qubits and the reduced density matrices of the state,

\[
I_Q(i:j)=S(\rho_i)+S(\rho_j)-S(\rho_{ij}),
\qquad S(\rho)=-\operatorname{Tr}(\rho\log_b\rho).
\]

The default base is `2`, so the unit is bits, and for two qubits
`0 <= I_Q <= 2`. This is the standard quantum mutual information: it counts
correlations that no local measurement can see.

The classical measure uses only the `Z`-basis distribution:

\[
I_Z(i:j)=\sum_{a,c}p_{ac}\log_b\frac{p_{ac}}{p_a p_c}.
\]

For binary outcomes `0 <= I_Z <= 1` bit. It is not the quantum mutual information and
it contains no coherences: a state can be maximally correlated in the quantum sense
and completely uncorrelated in the `Z` basis. `legacy_half_nats = 0.5 * I_Q(nats)`
exists only as an explicit derived view (`legacy_half_nats_view` in the
configuration, `legacy_half_nats` in the result). It is not a second definition and
should not be compared with either measure above.

## The example where the two disagree

```python
import math
import numpy as np
from QA_max_cut_lib.tools.mutual_information import (
    classical_from_probabilities,
    quantum_mutual_information,
)

bell = np.array([1, 0, 0, 1]) / math.sqrt(2)
cz = np.array([1, 1, 1, -1]) / 2

print(quantum_mutual_information(bell, [0, 1], (0, 1)))          # 2.0
print(classical_from_probabilities(abs(bell) ** 2, [0, 1], (0, 1)))  # 1.0
print(quantum_mutual_information(cz, [0, 1], (0, 1)))            # 2.0
print(classical_from_probabilities(abs(cz) ** 2, [0, 1], (0, 1)))    # 0.0
```

The Bell state is correlated in both senses: `2` bits of QMI and `1` bit of `Z`
information, because a `Z` measurement of one qubit determines the other. The state
`CZ|++>` is a different case: it is a maximally entangled state, `I_Q = 2`, yet its
`Z`-basis distribution is uniform over the four outcomes, so `I_Z = 0`. Any claim
about "correlation" built from `Z` counts alone would miss this state entirely.
These four numbers are asserted by
`tests/test_mutual_information.py::test_standard_qmi_table`.

A three-qubit GHZ state shows the same bookkeeping at a different scale: one qubit
has entropy `1`, the pair `(0, 1)` has `I_Q = 1` and `I_Z = 1`. Entropy is a property
of a subsystem; mutual information is a property of a pair.

## Partial trace and row order

The observer never builds a global `D x D` density matrix. From a statevector
`psi` of length `2^N` it forms the coefficient matrix `M` for the selected positions
and returns `rho_A = M M†`. For a pair `(i, j)` the rows are `00, 01, 10, 11`, with
the first label as the most significant bit of the small matrix. This is a
convention of the pair matrix only; it does not change the global little-endian
index order used by `node_order`. Reading a `2 x 2` block as if it were
little-endian swaps the off-diagonal entries and can turn a pure state into an
apparently mixed one.

## Snapshots

Two snapshot policies define which state is measured:

- `selected_final` — the state at the selected final angles, one frame.
- `final_angle_prefixes` — layer `0` and then the successive prefixes of one chosen
  circuit, `theta[:ell] + theta[p:p+ell]`. The two halves of the parameter vector are
  sliced together, so a prefix is a valid depth-`ell` circuit, not the first `ell`
  numbers of the flat list.

The observer is passive. It does not change the optimizer, the probes, the readout,
the RNG, or the shot budget; `tests/test_mutual_information.py` compares an observed
run against an unobserved one to confirm it. `I_Q` is an exact classical simulator
diagnostic. It is not reconstructed from `Z` shots, and a shots-only run does not
produce it.

## Limits

- No tomography: the observer reads the simulator state, it does not estimate one
  from measurements.
- No quantum mutual information from `Z`-only data, no MI-guided selector, and no
  reconstruction of removed nodes from their correlations.
- No dense global density matrix; only the selected small matrices are formed.
- In the RQAOA wrapper, frames are captured before the reducer commit. An old frame
  is not carried onto a new graph version, so a pair table always refers to the
  graph that produced it.

## Where next

- [API reference](API.md) — `diagnostics.mutual_information`, snapshot policies, and
  analysis limits.
- [Concepts](CONCEPTS.md) — `node_order` and the bit convention the pair matrix
  depends on.
- [Choosing an algorithm](CHOOSE_ALGORITHM.md) — when a diagnostic run is worth the
  extra bookkeeping.
