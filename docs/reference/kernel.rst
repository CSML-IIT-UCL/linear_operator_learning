.. _kernel_reference:
==============
:code:`kernel`
==============
.. module:: linear_operator_learning.kernel

Table of Contents
-----------------

- :ref:`Regressors <kernel_regressors>`
- :ref:`Types <kernel_types>`
- :ref:`Linear Algebra <kernel_linalg>`

.. _kernel_regressors:
Regressors
----------

Inference
~~~~~~~~~

.. autofunction:: linear_operator_learning.kernel.predict

.. autofunction:: linear_operator_learning.kernel.predict_physics_informed

.. autofunction:: linear_operator_learning.kernel.eig

.. autofunction:: linear_operator_learning.kernel.eig_physics_informed

.. autofunction:: linear_operator_learning.kernel.evaluate_eigenfunction

.. autofunction:: linear_operator_learning.kernle.evaluate_right_eigenfunction_physics_informed

.. _rrr:
Reduced Rank
~~~~~~~~~~~~
.. autofunction:: linear_operator_learning.kernel.reduced_rank

.. autofunction:: linear_operator_learning.kernel.nystroem_reduced_rank

.. autofunction:: linear_operator_learning.kernel.rand_reduced_rank

.. autofunction:: linear_operator_learning.kernel.reduced_rank_regression_physics_informed

.. _pcr:
Principal Component Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: linear_operator_learning.kernel.pcr

.. autofunction:: linear_operator_learning.kernel.nystroem_pcr

.. _kernel_types:
Types
~~~~~

.. autoclass:: linear_operator_learning.kernel.structs.FitResult
    :members:

.. autoclass:: linear_operator_learning.kernel.structs.EigResult
    :members:

.. _kernel_linalg:
Linear Algebra Utilities
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: linear_operator_learning.kernel.linalg.weighted_norm

.. autofunction:: linear_operator_learning.kernel.linalg.stable_topk

.. autofunction:: linear_operator_learning.kernel.linalg.add_diagonal_


.. _kernel_derivatives:
Kernel Derivatives
~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: linear_operator_learning.kernel.return_phi_dphi

.. autofunction:: linear_operator_learning.kernel.return_dphi_dphi

Bibliography
~~~~~~~~~~~~~~~~~~~~~~~~

.. footbibliography::