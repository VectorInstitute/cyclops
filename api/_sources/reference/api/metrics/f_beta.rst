#############
F-beta Score
#############

Module Interface
________________

FbetaScore
^^^^^^^^^^^
.. autoclass:: cyclops.evaluate.metrics.FbetaScore
    :class-doc-from: class

BinaryFbetaScore
^^^^^^^^^^^^^^^^^
.. autoclass:: cyclops.evaluate.metrics.BinaryFbetaScore
    :class-doc-from: class

MulticlassFbetaScore
^^^^^^^^^^^^^^^^^^^^
.. autoclass:: cyclops.evaluate.metrics.MulticlassFbetaScore
    :class-doc-from: class


MultilabelFbetaScore
^^^^^^^^^^^^^^^^^^^^
.. autoclass:: cyclops.evaluate.metrics.MultilabelFbetaScore
    :class-doc-from: class



Functional Interface
____________________

fbeta_score
^^^^^^^^^^^^
.. autofunction:: cyclops.evaluate.metrics.functional.f_beta.fbeta_score

binary_fbeta_score
^^^^^^^^^^^^^^^^^^
.. autofunction:: cyclops.evaluate.metrics.functional.f_beta.binary_fbeta_score

multiclass_fbeta_score
^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: cyclops.evaluate.metrics.functional.f_beta.multiclass_fbeta_score

multilabel_fbeta_score
^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: cyclops.evaluate.metrics.functional.f_beta.multilabel_fbeta_score
