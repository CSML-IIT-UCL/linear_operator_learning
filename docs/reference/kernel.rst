.. _kernel_reference:
==============
:code:`kernel`
==============
.. module:: linear_operator_learning.kernel

Table of Contents
-----------------

- :ref:`Inference <kernel_inference_fns>`
- :ref:`Types <kernel_types>`
- :ref:`Linear Algebra <kernel_linalg>`
- :ref:`Regressors <kernel_regressors>`

.. _kernel_inference_fns:
Inference
~~~~~~~~~

.. autofunction:: linear_operator_learning.kernel.predict

.. autofunction:: linear_operator_learning.kernel.eig

.. autofunction:: linear_operator_learning.kernel.evaluate_eigenfunction

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

.. _kernel_regressors:
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