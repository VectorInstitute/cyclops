#########
ROC Curve
#########

Module Interface
________________

ROCCurve
^^^^^^^^
.. autoclass:: cyclops.evaluate.metrics.ROCCurve
    :class-doc-from: class

BinaryROCCurve
^^^^^^^^^^^^^^
.. autoclass:: cyclops.evaluate.metrics.BinaryROCCurve
    :class-doc-from: class

MulticlassROCCurve
^^^^^^^^^^^^^^^^^^
.. autoclass:: cyclops.evaluate.metrics.MulticlassROCCurve
    :class-doc-from: class

MultilabelROCCurve
^^^^^^^^^^^^^^^^^^
.. autoclass:: cyclops.evaluate.metrics.MultilabelROCCurve
    :class-doc-from: class

Functional Interface
____________________

roc_curve
^^^^^^^^^
.. autofunction:: cyclops.evaluate.metrics.functional.roc.roc_curve

binary_roc_curve
^^^^^^^^^^^^^^^^
.. autofunction:: cyclops.evaluate.metrics.functional.roc.binary_roc_curve

multiclass_roc_curve
^^^^^^^^^^^^^^^^^^^^
.. autofunction:: cyclops.evaluate.metrics.functional.roc.multiclass_roc_curve

multilabel_roc_curve
^^^^^^^^^^^^^^^^^^^^
.. autofunction:: cyclops.evaluate.metrics.functional.roc.multilabel_roc_curve
