.. _kernel_reference:
==============
Kernel Methods
==============
.. module:: linear_operator_learning.kernel

Base Functions
--------------

Basic Functions
~~~~~~~~~~~~~~~

.. autofunction:: linear_operator_learning.kernel.predict

.. autofunction:: linear_operator_learning.kernel.eig

.. autofunction:: linear_operator_learning.kernel.evaluate_eigenfunction

Types
~~~~~

.. autoclass:: linear_operator_learning.kernel.structs.FitResult
    :members:

.. autoclass:: linear_operator_learning.kernel.structs.EigResult
    :members:

.. _linalg:
Linear Algebra Utilities
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: linear_operator_learning.kernel.linalg.weighted_norm

.. autofunction:: linear_operator_learning.kernel.linalg.stable_topk

.. autofunction:: linear_operator_learning.kernel.linalg.add_diagonal_


Regressors
----------
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

.. footbibliography::