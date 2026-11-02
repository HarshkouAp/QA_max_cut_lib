QA_max_cut_lib 0.1.0a8
========================

Восьмой этап научной библиотеки взвешенного Max-Cut.
Реализованы CPU-солверы, RQAOA, LS-RQAOA, NVIDIA GPU QAOA, QAOA-in-QAOA,
GW/SA и точное улучшение блоков.
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
* :download:`NVIDIA GPU QAOA <GPU.md>`
* :download:`Проверка исходной сборки <VALIDATION.md>`
* :download:`Согласованная полная архитектура <reference/MaxCut_Library_Architecture_RU.md>`

Девять выполненных учебных ноутбуков находятся в папке notebooks исходного
проекта. Третий ноутбук проверяет ядро редукций, четвёртый — RQAOA на CPU,
пятый — аналитический p=1, шестой — LS-RQAOA, седьмой — NVIDIA GPU QAOA,
восьмой — QAOA-in-QAOA, девятый — GW/SA и refinement.
Все они используют только реализованный API.
