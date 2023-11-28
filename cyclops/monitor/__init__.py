"""Monitor package."""

from cyclops.monitor.clinical_applicator import ClinicalShiftApplicator
from cyclops.utils.optional import import_optional_module


torch = import_optional_module("torch", error="warn")
if torch is not None:
    from cyclops.monitor.detector import Detector
    from cyclops.monitor.reductor import Reductor
    from cyclops.monitor.synthetic_applicator import SyntheticShiftApplicator
    from cyclops.monitor.tester import DCTester, TSTester
