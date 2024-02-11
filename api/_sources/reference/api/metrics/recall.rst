######
Recall
######

Module Interface
________________

Recall
^^^^^^
.. autoclass:: cyclops.evaluate.metrics.Recall
    :class-doc-from: class

BinaryRecall
^^^^^^^^^^^^
.. autoclass:: cyclops.evaluate.metrics.BinaryRecall
    :class-doc-from: class

MulticlassRecall
^^^^^^^^^^^^^^^^
.. autoclass:: cyclops.evaluate.metrics.MulticlassRecall
    :class-doc-from: class

MultilabelRecall
^^^^^^^^^^^^^^^^
.. autoclass:: cyclops.evaluate.metrics.MultilabelRecall
    :class-doc-from: class


Functional Interface
____________________

recall
^^^^^^
.. autofunction:: cyclops.evaluate.metrics.functional.precision_recall.recall

binary_recall
.. autofunction:: cyclops.evaluate.metrics.functional.precision_recall.binary_recall

multiclass_recall
^^^^^^^^^^^^^^^^^
.. autofunction:: cyclops.evaluate.metrics.functional.precision_recall.multiclass_recall

multilabel_recall
^^^^^^^^^^^^^^^^^
.. autofunction:: cyclops.evaluate.metrics.functional.precision_recall.multilabel_recall
