# Взаимная информация: квантовая и классическая Z

Версия `0.1.0a16` добавляет пассивную диагностику в `Tools` и две CPU
обёртки: `QAOAWithMIObserverCPU` и `RQAOAWithMIObserverCPU`.

## Две разные меры

Для пары qubits и reduced density matrix:

\[
I_Q(i:j)=S(\rho_i)+S(\rho_j)-S(\rho_{ij}),
\qquad S(\rho)=-\operatorname{Tr}(\rho\log_b\rho).
\]

Default: `base=2`, bits. Для двух qubits `0 <= I_Q <= 2`.

Классическая мера в Z basis:

\[
I_Z(i:j)=\sum_{a,c}p_{ac}\log_b\frac{p_{ac}}{p_a p_c}.
\]

Для binary outcomes `0 <= I_Z <= 1` bit. Она не является QMI и не
содержит coherences. `legacy_half_nats = 0.5 * I_Q(nats)` — только
явное derived view.

## Partial trace

Из `ψ` длины `2^N` строится coefficient matrix `M` для выбранных
позиций и `ρ_A=M M†`; глобальная `D×D` density matrix не создаётся.
Порядок строк pair matrix — `00,01,10,11`, первый label — старший
index маленькой matrix. Это отдельный convention, не смена global
little-endian.

## Snapshot-политики

- `selected_final`: состояние при выбранных final angles;
- `final_angle_prefixes`: слой 0 и последовательные prefixes одной
  выбранной схемы, `θ[:ℓ] + θ[p:p+ℓ]`.

Observer не меняет optimizer, probes, readout, RNG или shot budget.
QMI — exact classical simulator diagnostic; из Z-shots QMI не строится.

## Ограничения

Нет tomography, quantum MI из Z-only данных, MI-guided selector,
reconstruction удалённых узлов и dense global density matrix. RQAOA
frames берутся до reducer commit; старые frames не переносятся на новый
graph version.
