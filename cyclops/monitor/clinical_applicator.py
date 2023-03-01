"""Clinical Shift Applicator module."""

from typing import Callable, Dict

from datasets.arrow_dataset import Dataset

from cyclops.datasets.slicing import SlicingConfig


class ClinicalShiftApplicator:
    """The ClinicalShiftApplicator class is used induce synthetic clinical shifts.

    Parameters
    ----------
    shift_type: str
        Method used to induce shift in data.
        Options include: "seasonal", "hospital_type".

    """

    def __init__(self, shift_type: str, source, target, shift_id: str = None):
        self.shift_type = shift_type
        self.shift_id = shift_id
        self.source = source
        self.target = target

        self.shift_types: Dict[str, Callable[..., Dataset]] = {
            "time": self.time,
            "month": self.month,
            "hospital_type": self.hospital_type,
            "custom": self.custom,
        }

        if self.shift_type not in self.shift_types:
            raise ValueError(f"Shift type {self.shift_type} not supported. ")

    def apply_shift(self, dataset: Dataset):
        """apply_shift.

        Returns
        -------
        dataset: huggingface Dataset
            Dataset to apply shift to.

        """
        ds_source, ds_target = self.shift_types[self.shift_type](
            dataset, self.source, self.target, self.shift_id
        )
        return ds_source, ds_target

    def time(
        self,
        dataset: Dataset,
        source: list,
        target: list,
        shift_id: str,
    ):
        """Shift in time.

        Parameters
        ----------
        dataset: huggingface Dataset
            Dataset to apply shift to.
        shift_id: str
            Column name for shift id.
        source: list
            List of values for source data.
        target: list
            List of values for target data.

        Returns
        -------
        ds_source: huggingface Dataset
            Dataset with source data.
        ds_target: huggingface Dataset
            Dataset with target data.

        """
        source_slice = SlicingConfig(
            feature_values=[
                {
                    shift_id: {
                        "min_value": source[0],
                        "max_value": source[1],
                        "min_inclusive": True,
                        "max_inclusive": True,
                    }
                }
            ]
        )
        for _, shift_func in source_slice.get_slices().items():
            ds_source = dataset.filter(shift_func, batched=True)

        target_slice = SlicingConfig(
            feature_values=[
                {
                    shift_id: {
                        "min_value": target[0],
                        "max_value": target[1],
                        "min_inclusive": True,
                        "max_inclusive": True,
                    }
                }
            ]
        )
        for _, shift_func in target_slice.get_slices().items():
            ds_target = dataset.filter(shift_func, batched=True)
        return ds_source, ds_target

    def month(
        self,
        dataset: Dataset,
        source: list,
        target: list,
        shift_id: str,
    ):
        """Shift for selection of months.

        Parameters
        ----------
        dataset: huggingface Dataset
            Dataset to apply shift to.
        shift_id: str
            Column name for shift id.
        source: list
            List of values for source data.
        target: list
            List of values for target data.

        """
        source_slice = SlicingConfig(feature_values=[{shift_id: {"value": source}}])
        for _, shift_func in source_slice.get_slices().items():
            ds_source = dataset.filter(shift_func, batched=True)

        target_slice = SlicingConfig(feature_values=[{shift_id: {"value": target}}])
        for _, shift_func in target_slice.get_slices().items():
            ds_target = dataset.filter(shift_func, batched=True)
        return ds_source, ds_target

    def hospital_type(
        self, dataset: Dataset, source: list, target: list, shift_id: str
    ):
        """Shift against hospital type.

        Parameters
        ----------
        dataset: huggingface Dataset
            Dataset to apply shift to.
        shift_id: str
            Column name for shift id.
        source: list
            List of values for source data.
        target: list
            List of values for target data.

        """
        source_slice = SlicingConfig(feature_values=[{shift_id: {"value": source}}])
        for _, shift_func in source_slice.get_slices().items():
            ds_source = dataset.filter(shift_func, batched=True)

        target_slice = SlicingConfig(feature_values=[{shift_id: {"value": target}}])
        for _, shift_func in target_slice.get_slices().items():
            ds_target = dataset.filter(shift_func, batched=True)
        return ds_source, ds_target

    def custom(
        self,
        dataset: Dataset,
        source: SlicingConfig,
        target: SlicingConfig,
        shift_id: str = None,
    ):
        """Build custom shift.

        Build a custom shift by passing in a SlicingConfig for source and target data.

        Parameters
        ----------
        dataset: huggingface Dataset
            Dataset to apply shift to.
        shift_id: str
            Column name for shift id.
        source: SlicingConfig
            SlicingConfig for source data.
        target: SlicingConfig
            SlicingConfig for target data.

        """
        if shift_id:
            raise ValueError(
                "Shift id not required for custom shift. \
                Please remove shift_id from method call."
            )
        ds_source = None
        for _, shift_func in source.get_slices().items():
            if ds_source is None:
                ds_source = dataset.filter(shift_func, batched=True)
            else:
                ds_source = ds_source.filter(shift_func, batched=True)

        ds_target = None
        for _, shift_func in target.get_slices().items():
            if ds_target is None:
                ds_target = dataset.filter(shift_func, batched=True)
            else:
                ds_target = ds_target.filter(shift_func, batched=True)
        return ds_source, ds_target
