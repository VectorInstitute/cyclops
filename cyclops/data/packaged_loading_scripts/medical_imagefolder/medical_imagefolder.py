"""Loading script for medical images stored in a folder.

The expected folder structure is: root_folder/train/label1/image1.dcm
root_folder/train/label1/image2.dcm root_folder/train/label2/image3.dcm
root_folder/test/label2/image1.dcm root_folder/test/label1/image2.dcm

"""
import logging
from typing import List

from datasets.features import Features
from datasets.packaged_modules.folder_based_builder import folder_based_builder
from datasets.tasks import ImageClassification

from cyclops.data.features.medical_image import MedicalImage
from cyclops.utils.log import setup_logging


# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)


class MedicalImageFolderConfig(
    folder_based_builder.FolderBasedBuilderConfig,  # type: ignore
):
    """BuilderConfig for MedicalImageFolder."""

    drop_labels: bool = None  # type: ignore
    drop_metadata: bool = None  # type: ignore


class MedicalImageFolder(folder_based_builder.FolderBasedBuilder):  # type: ignore
    """MedicalImageFolder."""

    BASE_FEATURE = MedicalImage()
    BASE_COLUMN_NAME = "image"
    BUILDER_CONFIG_CLASS = MedicalImageFolderConfig
    EXTENSIONS: List[str]  # definition at the bottom of the script
    ImageClassification.input_schema = Features({"image": MedicalImage()})
    CLASSIFICATION_TASK = ImageClassification(
        image_column="image",
        label_column="label",
    )


# for more image formats, see:
# https://insightsoftwareconsortium.github.io/itk-wasm/docs/image_formats.html
# and
# https://simpleitk.readthedocs.io/en/master/IO.html
IMAGE_EXTENSIONS = [
    ".bmp",
    ".BMP",
    ".dib",
    ".dcm",
    ".gipl",
    ".gipl.gz",
    ".h5",
    ".hdf",
    ".hdr",
    ".img",
    ".img.gz",
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".lsm",
    ".LSM",
    ".mha",
    ".mnc",
    ".mrc",
    ".MNC",
    ".nia",
    ".nii",
    ".nii.gz",
    ".nhdr",
    ".nrrd",
    ".pic",
    ".PIC",
    ".png",
    ".PNG",
    ".rec",
    ".tif",
    ".TIF",
    ".tiff",
    ".TIFF",
    ".vtk",
]
MedicalImageFolder.EXTENSIONS = IMAGE_EXTENSIONS
