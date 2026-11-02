# Блочный mean-field QAOA (CPU)

Версия `0.1.0a14` добавляет `BlockMeanFieldQAOACPU`
(`laboratory.block_mean_field_qaoa_cpu`). Это полный CPU-солвер с
несколькими локальными QAOA-задачами и классическим обменом средними
спинами.

## Модель

Граф и partition фиксированы на run. Для каждого блока `S_a` хранятся
только компактные моменты:

\[
m_i=\langle Z_i\rangle,\qquad q_{ij}=\langle Z_iZ_j\rangle\quad(i,j\in S_a).
\]

Средний cut product-модели:

\[
\mathcal C(m,q)=
\frac12\sum_{\{i,j\}\in E,\ a(i)=a(j)}w_{ij}(1-q_{ij})
+\frac12\sum_{\{i,j\}\in E,\ a(i)\ne a(j)}w_{ij}(1-m_im_j).
\]

Для активного блока внешние поля:

\[
h_i=\frac12\sum_{j\notin S_a}w_{ij}m_j,\qquad
c_{\mathrm{aff}}=-\frac12\left(\sum_{\mathrm{internal}}w+
\sum_{\mathrm{cross}}w\right).
\]

`J_{ik}=w_{ik}/2`. Локальная диагональ считается из коэффициентов
`A_S=-c_aff-h·x-J·xx`, а не из одного индуцированного подграфа.

## Обновление

1. Строятся `h/J/c_aff` по текущему (GS) или замороженному (Jacobi)
   снимку сообщений.
2. Общий CPU QAOA/ADAM обучает полный `2^{|S|}` statevector.
3. Углы смешиваются: `theta_applied=(1-kappa)theta_old+kappa theta_candidate`.
4. При **applied** углах одна выборка даёт все `Z` и внутренние `ZZ`.
5. `m` и `q` блока смешиваются одним и тем же `lambda`.
6. Jacobi публикует весь sweep только после барьера; GS — после каждого
   блока.

Выход — `sign_means(m)` с явным `zero_sign`, затем независимый полный
cut на исходном графе. Это не global quantum readout.

## Ограничения

- `max_sweeps` конечен; checkpoint/resume реализован на границе полного
  sweep (initialized / sweep commit), без mid-sweep child codec.
- Нет overlapping blocks, adaptive damping, nonlinear/correlation-weighted
  fields, GPU/chi/analytic path, final sampling глобальной смеси.
- `local_surrogate` threshold probability не является вероятностью
  `C_G >= A`.
- Пустой граф возвращает `{}` без quantum работы.
