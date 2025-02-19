.. _kernel_reference:
========================
:code:`lol.kernel` namespace
========================
.. module:: linear_operator_learning.kernel

Table of Contents
-----------------

- :ref:`Regressors <kernel_regressors>`
- :ref:`Types <kernel_types>`
- :ref:`Linear Algebra <kernel_linalg>`
- :ref:`Utilities <kernel_utils>`


.. _kernel_regressors:
Regressors
----------

Common
~~~~~~

.. autofunction:: linear_operator_learning.kernel.predict

.. autofunction:: linear_operator_learning.kernel.eig

.. autofunction:: linear_operator_learning.kernel.evaluate_eigenfunction

.. _rrr:
Reduced Rank
~~~~~~~~~~~~
.. autofunction:: linear_operator_learning.kernel.reduced_rank

.. autofunction:: linear_operator_learning.kernel.nystroem_reduced_rank

.. autofunction:: linear_operator_learning.kernel.rand_reduced_rank

.. _pcr:
Principal Component Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: linear_operator_learning.kernel.pcr

.. autofunction:: linear_operator_learning.kernel.nystroem_pcr

.. _kernel_types:
Types
-----

.. autoclass:: linear_operator_learning.kernel.structs.FitResult
    :members:

.. autoclass:: linear_operator_learning.kernel.structs.EigResult
    :members:

.. _kernel_linalg:
Linear Algebra Utilities
------------------------

.. autofunction:: linear_operator_learning.kernel.linalg.weighted_norm

.. autofunction:: linear_operator_learning.kernel.linalg.stable_topk

.. autofunction:: linear_operator_learning.kernel.linalg.add_diagonal_

.. _kernel_utils:
General Utilities
-----------------

.. autofunction:: linear_operator_learning.kernel.utils.topk

.. autofunction:: linear_operator_learning.kernel.utils.sanitize_complex_conjugates


.. footbibliography::