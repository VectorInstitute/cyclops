"""Medical image feature."""

import os
import tempfile
from dataclasses import dataclass, field
from io import BytesIO
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pyarrow as pa
from datasets import config
from datasets.download.streaming_download_manager import xopen
from datasets.features import Image, features
from datasets.utils.file_utils import is_local_path
from datasets.utils.py_utils import string_to_dict

from cyclops.utils.optional import import_optional_module


if TYPE_CHECKING:
    from monai.data.image_reader import ImageReader
    from monai.data.image_writer import ITKWriter
    from monai.transforms.compose import Compose
    from monai.transforms.io.array import LoadImage
    from monai.transforms.utility.array import ToNumpy
else:
    ImageReader = import_optional_module(
        "monai.data.image_reader",
        attribute="ImageReader",
        error="warn",
    )
    ITKWriter = import_optional_module(
        "monai.data.image_writer",
        attribute="ITKWriter",
        error="warn",
    )
    Compose = import_optional_module(
        "monai.transforms.compose",
        attribute="Compose",
        error="warn",
    )
    LoadImage = import_optional_module(
        "monai.transforms.io.array",
        attribute="LoadImage",
        error="warn",
    )
    ToNumpy = import_optional_module(
        "monai.transforms.utility.array",
        attribute="ToNumpy",
        error="warn",
    )
_monai_available = all(
    module is not None
    for module in (
        ImageReader,
        ITKWriter,
        Compose,
        LoadImage,
        ToNumpy,
    )
)
_monai_unavailable_message = (
    "The MONAI library is required to use the `MedicalImage` feature. "
    "Please install it with `pip install monai`."
)


@dataclass
class MedicalImage(Image):  # type: ignore
    """Medical image `Feature` to read medical image files.

    Parameters
    ----------
    reader : Union[str, ImageReader], optional, default="ITKReader"
        The MONAI image reader to use.
    suffix : str, optional, default=".jpg"
        The suffix to use when decoding bytes to image.
    decode : bool, optional, default=True
        Whether to decode the image. If False, the image will be returned as a
        dictionary in the format `{"path": image_path, "bytes": image_bytes}`.
    id : str, optional, default=None
        The id of the feature.

    """

    reader: Union[str, ImageReader] = "ITKReader"
    suffix: str = ".jpg"  # used when decoding/encoding bytes to image

    _loader = None
    if _monai_available:
        _loader = Compose(
            [
                LoadImage(
                    reader=reader,
                    simple_keys=True,
                    dtype=None,
                    image_only=False,
                ),
                ToNumpy(),
            ],
        )

    # Automatically constructed
    dtype: ClassVar[str] = "dict"
    pa_type: ClassVar[Any] = pa.struct({"bytes": pa.binary(), "path": pa.string()})
    _type: str = field(default="MedicalImage", init=False, repr=False)

    def encode_example(
        self,
        value: Union[str, Dict[str, Any], npt.NDArray[Any]],
    ) -> Dict[str, Any]:
        """Encode example into a format for Arrow.

        Parameters
        ----------
        value : Union[str, dict, np.ndarray]
            Data passed as input to MedicalImage feature.

        Returns
        -------
        dict
            The encoded example.

        """
        if isinstance(value, list):
            value = np.asarray(value)

        if isinstance(value, str):
            return {"path": value, "bytes": None}

        if isinstance(value, np.ndarray):
            return _encode_ndarray(value, image_format=self.suffix)

        if "array" in value and "metadata" in value:
            output_ext_ = self.suffix
            metadata_ = value["metadata"]
            filename = metadata_.get("filename_or_obj", None)
            if filename is not None and filename != "":
                output_ext_ = os.path.splitext(filename)[1]
            return _encode_ndarray(
                value["array"],
                metadata=metadata_,
                image_format=output_ext_,
            )
        if value.get("path") is not None and os.path.isfile(value["path"]):
            # we set "bytes": None to not duplicate the data
            # if they're already available locally
            return {"bytes": None, "path": value.get("path")}
        if value.get("bytes") is not None or value.get("path") is not None:
            # store the image bytes, and path is used to infer the image format
            # using the file extension
            return {"bytes": value.get("bytes"), "path": value.get("path")}

        raise ValueError(
            "An image sample should have one of 'path' or 'bytes' "
            f"but they are missing or None in {value}.",
        )

    def decode_example(
        self,
        value: Dict[str, Any],
        token_per_repo_id: Optional[Dict[str, Union[str, bool, None]]] = None,
    ) -> Dict[str, Any]:
        """Decode an example from the serialized version to the feature type version.

        Parameters
        ----------
        value : dict
            The serialized example.
        token_per_repo_id : dict, optional
            To access and decode image files from private repositories on the Hub,
            you can pass a dictionary repo_id (`str`) -> token (`bool` or `str`).

        Returns
        -------
        dict
            The deserialized example as a dictionary in the format:
            `{"array": np.ndarray, "metadata": dict}`.

        """
        if not self.decode:
            raise RuntimeError(
                "Decoding is disabled for this feature. "
                "Please use `MedicalImage(decode=True)` instead.",
            )

        if token_per_repo_id is None:
            token_per_repo_id = {}

        path, bytes_ = value["path"], value["bytes"]
        if bytes_ is None:
            if path is None:
                raise ValueError(
                    "An image should have one of 'path' or 'bytes' but both are "
                    f"None in {value}.",
                )

            if is_local_path(path):
                if self._loader is None:
                    raise RuntimeError(_monai_unavailable_message)
                image, metadata = self._loader(path)
            else:
                source_url = path.split("::")[-1]
                try:
                    repo_id = string_to_dict(source_url, config.HUB_DATASETS_URL)[
                        "repo_id"
                    ]
                    use_auth_token = token_per_repo_id.get(repo_id)
                except ValueError:
                    use_auth_token = None
                with xopen(
                    path,
                    "rb",
                    use_auth_token=use_auth_token,
                ) as file_obj, BytesIO(file_obj.read()) as buffer:
                    image, metadata = self._read_file_from_bytes(buffer)
                    metadata["filename_or_obj"] = path

        else:
            with BytesIO(bytes_) as buffer:
                image, metadata = self._read_file_from_bytes(buffer)

        return {"array": image, "metadata": metadata}

    def _read_file_from_bytes(
        self,
        buffer: BytesIO,
    ) -> Tuple[npt.NDArray[Any], Dict[str, Any]]:
        """Read an image from bytes.

        Parameters
        ----------
        buffer : BytesIO
            BytesIO object containing image data as bytes.

        Returns
        -------
        Tuple[np.ndarray, dict]
            Image as numpy array and metadata as dictionary.

        """
        if self._loader is None:
            raise RuntimeError(_monai_unavailable_message)

        # XXX: Can we avoid writing to disk?
        with tempfile.NamedTemporaryFile(mode="wb", suffix=self.suffix) as fp:
            fp.write(buffer.getvalue())
            fp.flush()
            image, metadata = self._loader(fp.name)
            metadata["filename_or_obj"] = ""
            return image, metadata


def _encode_ndarray(
    array: npt.NDArray[Any],
    metadata: Optional[Dict[str, Any]] = None,
    image_format: str = ".png",
) -> Dict[str, Any]:
    """Encode a numpy array or torch tensor as bytes.

    Parameters
    ----------
    array : numpy.ndarray
        Numpy array to encode.
    metadata : dict, optional, default=None
        Metadata dictionary.
    image_format : str, optional, default=".png"
        Output image format.

    Returns
    -------
    dict
        Dictionary containing the image bytes and path.

    """
    if not _monai_available:
        raise RuntimeError(_monai_unavailable_message)

    if not image_format.startswith("."):
        image_format = "." + image_format

    # TODO: find a way to avoid writing to disk
    # TODO: figure out output dtype

    with tempfile.NamedTemporaryFile(mode="wb", suffix=image_format) as temp_file:
        writer = ITKWriter(output_dtype=np.uint8)
        writer.set_data_array(data_array=array, channel_dim=-1, squeeze_end_dims=False)
        writer.set_metadata(meta_dict=metadata, resample=True)
        writer.write(temp_file.name)

        temp_file.flush()

        # read tmp file into bytes
        with open(temp_file.name, "rb") as f_obj:
            temp_file_bytes = f_obj.read()

        return {"path": None, "bytes": temp_file_bytes}


# add the `MedicalImage` feature to the `features` module namespace
features.MedicalImage = MedicalImage
