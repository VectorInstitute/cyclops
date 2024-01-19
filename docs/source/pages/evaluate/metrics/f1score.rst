#########
F1-Score
#########

Module Interface
________________

F1Score
^^^^^^^
.. autoclass:: cyclops.evaluate.metrics.F1Score
    :class-doc-from: class

BinaryF1Score
^^^^^^^^^^^^^^
.. autoclass:: cyclops.evaluate.metrics.BinaryF1Score
    :class-doc-from: class

MulticlassF1Score
^^^^^^^^^^^^^^^^^^^
.. autoclass:: cyclops.evaluate.metrics.MulticlassF1Score
    :class-doc-from: class


MultilabelF1Score
^^^^^^^^^^^^^^^^^^^
.. autoclass:: cyclops.evaluate.metrics.MultilabelF1Score
    :class-doc-from: class



Functional Interface
____________________

f1_score
^^^^^^^^^^^^
.. autofunction:: cyclops.evaluate.metrics.functional.f_beta.f1_score

binary_f1_score
^^^^^^^^^^^^^^^^^^
.. autofunction:: cyclops.evaluate.metrics.functional.f_beta.binary_f1_score

multiclass_f1_score
^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: cyclops.evaluate.metrics.functional.f_beta.multiclass_f1_score


multilabel_f1_score
^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: cyclops.evaluate.metrics.functional.f_beta.multilabel_f1_score
