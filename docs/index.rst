QA_max_cut_lib 0.1.0a2
========================

Второй CPU-этап научной библиотеки взвешенного Max-Cut.
Реализованы ExactCPU, QAOACPU и точное ядро парных редукций Max-Cut → Изинг.
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
* :download:`Проверка исходной сборки <VALIDATION.md>`
* :download:`Согласованная полная архитектура <reference/MaxCut_Library_Architecture_RU.md>`

Два выполненных учебных ноутбука находятся в папке notebooks исходного
проекта. Третий ноутбук проверяет ядро редукций. Все они используют только
реализованный API.
