QA_max_cut_lib 0.1.0a18
========================

Четырнадцатый этап научной библиотеки взвешенного Max-Cut.
Реализованы CPU-солверы, RQAOA, LS-RQAOA, NVIDIA GPU QAOA, QAOA-in-QAOA,
GW/SA, точное улучшение блоков, ExperimentRunner, ParallelismProfiler и
checkpoint/resume `QAOACPU`, Laboratory-каталог и warm-start CPU, warm-start RQAOA, условный QAOA с фиксированными внешними спинами и environment LS-RQAOA.
Полный перенос архива продолжается; отсутствующие алгоритмы перечислены
в :download:`статусе переноса <MIGRATION_STATUS.md>`.

Используется энергетическое соглашение

.. math::

   E(s)=-C(s)=\sum_{(i,j)\in E}\frac{w_{ij}}2(s_i s_j-1).

Меньшая энергия означает лучший разрез. Постоянная часть сохранена.
Критерий оптимизации и физическое ожидание энергии различаются для
CVaR, Gibbs, Best-of-K и Threshold Probability.

.. toctree::
   :maxdepth: 2

   api

Руководства
-----------

* :download:`Все конфигурационные поля, аргументы и атрибуты <API.md>`
* :download:`Математические определения и численные ограничения <MATHEMATICS.md>`
* :download:`Установка CPU и NVIDIA-профиля <INSTALLATION.md>`
* :download:`Карта архитектуры и состояния переноса <ARCHITECTURE_MAP_RU.md>`
* :download:`Рабочее окружение проекта <PROJECT_SETUP_RU.md>`
* :download:`Локальная проверка Windows <VALIDATION_LOCAL.md>`
* :download:`Проверка этапа 02 <VALIDATION_0.1.0a2.md>`
* :download:`Проверка этапа 03 <VALIDATION_0.1.0a3.md>`
* :download:`Проверка этапа 04 <VALIDATION_0.1.0a4.md>`
* :download:`Проверка этапа 05 <VALIDATION_0.1.0a5.md>`
* :download:`Проверка этапа 06 <VALIDATION_0.1.0a6.md>`
* :download:`Проверка этапа 07 <VALIDATION_0.1.0a7.md>`
* :download:`Проверка этапа 08 <VALIDATION_0.1.0a8.md>`
* :download:`Проверка этапа 09 <VALIDATION_0.1.0a9.md>`
* :download:`Проверка этапа 10 <VALIDATION_0.1.0a10.md>`
* :download:`Проверка этапа 11 <VALIDATION_0.1.0a11.md>`
* :download:`Проверка этапа 12 <VALIDATION_0.1.0a12.md>`
* :download:`Проверка этапа 13 <VALIDATION_0.1.0a13.md>`
* :download:`Проверка этапа 14 <VALIDATION_0.1.0a14.md>`
* :download:`Проверка этапа 15 <VALIDATION_0.1.0a15.md>`
* :download:`Проверка этапа 16 <VALIDATION_0.1.0a16.md>`
* :download:`Проверка этапа 17 <VALIDATION_0.1.0a17.md>`
* :download:`Проверка этапа 18 <VALIDATION_0.1.0a18.md>`
* :download:`Warm-start RQAOA <WARM_START_RQAOA_RU.md>`
* :download:`Laboratory: каталог и warm-start <LABORATORY_RU.md>`
* :download:`Условные поля и environment LS <CONDITIONAL_RU.md>`
* :download:`Блочный mean-field QAOA <BLOCK_MEAN_FIELD_RU.md>`
* :download:`Distributed LS-RQAOA <DISTRIBUTED_LS_RU.md>`
* :download:`Взаимная информация <MUTUAL_INFORMATION_RU.md>`
* :download:`Incremental LS-RQAOA <INCREMENTAL_LS_RU.md>`
* :download:`Generators и typed plans <GENERATORS_PLANS_RU.md>`
* :download:`Контрольные точки и resume <CHECKPOINTS_RU.md>`
* :download:`Серии экспериментов и профилирование <EXPERIMENTS_RU.md>`
* :download:`NVIDIA GPU QAOA <GPU.md>`
* :download:`Проверка исходной сборки <VALIDATION.md>`
* :download:`Согласованная полная архитектура <reference/MaxCut_Library_Architecture_RU.md>`

Девятнадцать выполненных учебных ноутбуков находятся в папке notebooks исходного
проекта. Третий ноутбук проверяет ядро редукций, четвёртый — RQAOA на CPU,
пятый — аналитический p=1, шестой — LS-RQAOA, седьмой — NVIDIA GPU QAOA,
восьмой — QAOA-in-QAOA, девятый — GW/SA и refinement, десятый — серии,
эталоны и профилирование параллелизма, одиннадцатый — checkpoint/resume, двенадцатый — Laboratory-каталог и warm-start, тринадцатый — warm-start RQAOA ensemble, четырнадцатый — условные поля и environment LS, пятнадцатый — блочный mean-field обмен, шестнадцатый — distributed LS-RQAOA, семнадцатый — взаимная информация, восемнадцатый — incremental LS/MK стратегии, девятнадцатый — генераторы и coverage планов.
Все они используют только реализованный API.
