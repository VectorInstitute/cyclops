"""Packaged dataset loading scripts."""
__all__ = ["medical_imagefolder"]

import inspect

from datasets.packaged_modules import _PACKAGED_DATASETS_MODULES, _hash_python_lines

from cyclops.data.packaged_loading_scripts.medical_imagefolder import (
    medical_imagefolder,
)

# add the packaged loading scripts to huggingface datasets' registry
# NOTE: cyclops.data must be imported before this change takes effect
_PACKAGED_DATASETS_MODULES.update(
    {
        "medicalimagefolder": (
            medical_imagefolder.__name__,
            _hash_python_lines(inspect.getsource(medical_imagefolder).splitlines()),
        )
    }
)
