"""Clinical Shift Applicator module."""

from typing import Callable, Dict, List, Optional, Tuple, Union

from datasets.arrow_dataset import Dataset

from cyclops.datasets.slicer import SliceSpec
from cyclops.monitor.utils import set_decode, sync_transforms


class ClinicalShiftApplicator:
    """The ClinicalShiftApplicator class is used induce synthetic clinical shifts.

    Takes a dataset and generates a source and
    target dataset with a specified clinical shift.
    The shift is induced by splitting along categorical features in the dataset.
    The source and target datasets are then generated by splitting
    the original dataset along the categorical feature.

    Examples
    --------
    >>> from cyclops.monitor.clinical_applicator import ClinicalShiftApplicator
    >>> from cyclops.datasets.utils import load_nih
    >>> ds = load_nih(path="/mnt/data/nihcxr")
    >>> applicator = ClinicalShiftApplicator("hospital_type",
                    source = ["hospital_type_1", "hospital_type_2"]
                    target = ["hospital_type_3", "hospital_type_4", "hospital_type_5"]
                    )
    >>> ds_source, ds_target = applicator.apply_shift(ds)


    Parameters
    ----------
    shift_type: str
        method used to induce shift in data. Options include:
        "time", "month", "hospital_type", "custom"
    source: list
        List of values for source data.
    target: list
        List of values for target data.
    shift_id: str
        Column name for shift id. Default is None.

    """

    def __init__(
        self,
        shift_type: str,
        source: Union[str, SliceSpec],
        target: Union[str, SliceSpec],
        shift_id: Optional[str] = None,
    ):
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

    def apply_shift(
        self,
        dataset: Dataset,
        batched: bool = True,
        batch_size: int = 1000,
        num_proc: int = 1,
    ) -> Tuple[Dataset, Dataset]:
        """Apply shift to dataset using specified shift type.

        Returns
        -------
        ds_source: huggingface Dataset
            Dataset with source data.
        ds_target: huggingface Dataset
            Dataset with target data.

        """
        ds_source, ds_target = self.shift_types[self.shift_type](
            dataset,
            self.source,
            self.target,
            self.shift_id,
            batched,
            batch_size,
            num_proc,
        )
        return ds_source, ds_target

    def time(
        self,
        dataset: Dataset,
        source: List[str],
        target: List[str],
        shift_id: str,
        batched: bool = True,
        batch_size: int = 1000,
        num_proc: int = 1,
    ) -> Tuple[Dataset, Dataset]:
        """Apply time shift to dataset.

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
        shift_id: str
            Column name for shift id.
        batched: bool
            Whether to use batching or not. Default is True.
        batch_size: int
            Batch size. Default is 1000.
        num_proc: int
            Number of processes to use. Default is 1.

        Returns
        -------
        ds_source: huggingface Dataset
            Dataset with source data.
        ds_target: huggingface Dataset
            Dataset with target data.

        """
        source_slice = SliceSpec(
            spec_list=[
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
        set_decode(dataset, False)
        with dataset.formatted_as("numpy", output_all_columns=True):
            for _, shift_func in source_slice.get_slices().items():
                ds_source = dataset.filter(
                    shift_func,
                    batched=batched,
                    batch_size=batch_size,
                    num_proc=num_proc,
                )

            target_slice = SliceSpec(
                spec_list=[
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
                ds_target = dataset.filter(
                    shift_func,
                    batched=batched,
                    batch_size=batch_size,
                    num_proc=num_proc,
                )
        set_decode(dataset, True)
        set_decode(ds_source, True)
        set_decode(ds_target, True)
        ds_source = sync_transforms(dataset, ds_source)
        ds_target = sync_transforms(dataset, ds_target)
        return ds_source, ds_target

    def month(
        self,
        dataset: Dataset,
        source: List[str],
        target: List[str],
        shift_id: str,
        batched: bool = True,
        batch_size: int = 1000,
        num_proc: int = 1,
    ) -> Tuple[Dataset, Dataset]:
        """Apply shift for selection of months.

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
        shift_id: str
            Column name for shift id.
        batched: bool
            Whether to use batching or not. Default is True.
        batch_size: int
            Batch size. Default is 1000.
        num_proc: int
            Number of processes to use. Default is 1.

        Returns
        -------
        ds_source: huggingface Dataset
            Dataset with source data.
        ds_target: huggingface Dataset
            Dataset with target data.

        """
        set_decode(dataset, False)
        with dataset.formatted_as("numpy", output_all_columns=True):
            source_slice = SliceSpec(spec_list=[{shift_id: {"value": source}}])
            for _, shift_func in source_slice.get_slices().items():
                ds_source = dataset.filter(
                    shift_func,
                    batched=batched,
                    batch_size=batch_size,
                    num_proc=num_proc,
                )

            target_slice = SliceSpec(spec_list=[{shift_id: {"value": target}}])
            for _, shift_func in target_slice.get_slices().items():
                ds_target = dataset.filter(
                    shift_func,
                    batched=batched,
                    batch_size=batch_size,
                    num_proc=num_proc,
                )
        set_decode(dataset, True)
        set_decode(ds_source, True)
        set_decode(ds_target, True)
        ds_source = sync_transforms(dataset, ds_source)
        ds_target = sync_transforms(dataset, ds_target)
        return ds_source, ds_target

    def hospital_type(
        self,
        dataset: Dataset,
        source: List[str],
        target: List[str],
        shift_id: str,
        batched: bool = True,
        batch_size: int = 1000,
        num_proc: int = 1,
    ) -> Tuple[Dataset, Dataset]:
        """Apply shift for selection of hospital types.

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
        shift_id: str
            Column name for shift id.
        batched: bool
            Whether to use batching or not. Default is True.
        batch_size: int
            Batch size. Default is 1000.
        num_proc: int
            Number of processes to use. Default is 1.

        Returns
        -------
        ds_source: huggingface Dataset
            Dataset with source data.
        ds_target: huggingface Dataset
            Dataset with target data.

        """
        set_decode(dataset, False)
        with dataset.formatted_as("numpy", output_all_columns=True):
            source_slice = SliceSpec(spec_list=[{shift_id: {"value": source}}])
            for _, shift_func in source_slice.get_slices().items():
                ds_source = dataset.filter(
                    shift_func,
                    batched=batched,
                    batch_size=batch_size,
                    num_proc=num_proc,
                )

            target_slice = SliceSpec(spec_list=[{shift_id: {"value": target}}])
            for _, shift_func in target_slice.get_slices().items():
                ds_target = dataset.filter(
                    shift_func,
                    batched=batched,
                    batch_size=batch_size,
                    num_proc=num_proc,
                )
        set_decode(dataset, True)
        set_decode(ds_source, True)
        set_decode(ds_target, True)
        ds_source = sync_transforms(dataset, ds_source)
        ds_target = sync_transforms(dataset, ds_target)
        return ds_source, ds_target

    def custom(
        self,
        dataset: Dataset,
        source: SliceSpec,
        target: SliceSpec,
        shift_id: Optional[str] = None,
        batched: bool = True,
        batch_size: int = 1000,
        num_proc: int = 1,
    ) -> Tuple[Dataset, Dataset]:
        """Build custom shift.

        Build a custom shift by passing in a SliceSpec for source and target data.

        Parameters
        ----------
        dataset: huggingface Dataset
            Dataset to apply shift to.
        source: SliceSpec
            SliceSpec for source data.
        target: SliceSpec
            SliceSpec for target data.
        shift_id: str
            Column name for shift id.
        batched: bool
            Whether to use batching or not. Default is True.
        batch_size: int
            Batch size. Default is 1000.
        num_proc: int
            Number of processes to use. Default is 1.

        Returns
        -------
        ds_source: huggingface Dataset
            Dataset with source data.
        ds_target: huggingface Dataset
            Dataset with target data.

        """
        if shift_id:
            raise ValueError(
                "Shift id not required for custom shift. \
                Please remove shift_id from method call."
            )
        set_decode(dataset, False)
        with dataset.formatted_as("numpy", output_all_columns=True):
            ds_source = None
            for _, shift_func in source.get_slices().items():
                if ds_source is None:
                    ds_source = dataset.filter(
                        shift_func,
                        batched=batched,
                        batch_size=batch_size,
                        num_proc=num_proc,
                    )
                else:
                    ds_source = ds_source.filter(
                        shift_func,
                        batched=batched,
                        batch_size=batch_size,
                        num_proc=num_proc,
                    )

            ds_target = None
            for _, shift_func in target.get_slices().items():
                if ds_target is None:
                    ds_target = dataset.filter(
                        shift_func,
                        batched=batched,
                        batch_size=batch_size,
                        num_proc=num_proc,
                    )
                else:
                    ds_target = ds_target.filter(
                        shift_func,
                        batched=batched,
                        batch_size=batch_size,
                        num_proc=num_proc,
                    )

        set_decode(dataset, True)

        # ds_source = sync_transforms(dataset, ds_source)
        # ds_target = sync_transforms(dataset, ds_target)
        set_decode(ds_source, True)
        set_decode(ds_target, True)

        return ds_source, ds_target
