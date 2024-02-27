####################
PrecisionRecallCurve
####################

Module Interface
________________

PrecisionRecallCurve
^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: cyclops.evaluate.metrics.PrecisionRecallCurve
    :class-doc-from: class

BinaryPrecisionRecallCurve
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: cyclops.evaluate.metrics.BinaryPrecisionRecallCurve
    :class-doc-from: class

MulticlassPrecisionRecallCurve
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: cyclops.evaluate.metrics.MulticlassPrecisionRecallCurve
    :class-doc-from: class

MultilabelPrecisionRecallCurve
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: cyclops.evaluate.metrics.MultilabelPrecisionRecallCurve
    :class-doc-from: class


Functional Interface
____________________

precision_recall_curve
^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: cyclops.evaluate.metrics.functional.precision_recall_curve.precision_recall_curve

binary_precision_recall_curve
.. autofunction:: cyclops.evaluate.metrics.functional.precision_recall_curve.binary_precision_recall_curve

multiclass_precision_recall_curve
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: cyclops.evaluate.metrics.functional.precision_recall_curve.multiclass_precision_recall_curve

multilabel_precision_recall_curve
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: cyclops.evaluate.metrics.functional.precision_recall_curve.multilabel_precision_recall_curve
