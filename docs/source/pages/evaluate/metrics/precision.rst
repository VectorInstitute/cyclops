#########
Precision
#########

Module Interface
________________

Precision
^^^^^^^^^
.. autoclass:: cyclops.evaluate.metrics.Precision
    :class-doc-from: class

BinaryPrecision
^^^^^^^^^^^^^^^
.. autoclass:: cyclops.evaluate.metrics.BinaryPrecision
    :class-doc-from: class

MulticlassPrecision
^^^^^^^^^^^^^^^^^^^
.. autoclass:: cyclops.evaluate.metrics.MulticlassPrecision
    :class-doc-from: class

MultilabelPrecision
^^^^^^^^^^^^^^^^^^^
.. autoclass:: cyclops.evaluate.metrics.MultilabelPrecision
    :class-doc-from: class


Functional Interface
____________________

precision
^^^^^^^^^
.. autofunction:: cyclops.evaluate.metrics.functional.precision_recall.precision

binary_precision
^^^^^^^^^^^^^^^^
.. autofunction:: cyclops.evaluate.metrics.functional.precision_recall.binary_precision

multiclass_precision
^^^^^^^^^^^^^^^^^^^^
.. autofunction:: cyclops.evaluate.metrics.functional.precision_recall.multiclass_precision

multilabel_precision
^^^^^^^^^^^^^^^^^^^^
.. autofunction:: cyclops.evaluate.metrics.functional.precision_recall.multilabel_precision
