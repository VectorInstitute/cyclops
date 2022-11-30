"""Data Processor class to process data per use-case."""

from use_cases.constants import DATA_TYPES, DATASETS, USE_CASES
from use_cases.data_processors.mimiciv import MimicivProcessor


class DataProcessor:
    """Data processor class."""

    def __init__(
        self,
        dataset_name: str,
        use_case: str,
        data_type: str,
    ) -> None:
        """Initialize processor.

        Parameters
        ----------
        dataset_name : str
            dataset name to process its data
        use_case : str
            use-case to process the data for
        data_type : str
            type of data (tabular, temporal, or combined)

        """
        self.dataset_name = dataset_name.lower()
        self.use_case = use_case.lower()
        self.data_type = data_type.lower()

        self._validate()
        self._init_processor()

    def _validate(self) -> None:
        """Validate the input arguments."""
        assert self.dataset_name in DATASETS, "[!] Invalid dataset name"
        assert (
            self.use_case in USE_CASES.keys()  # pylint: disable=C0201
        ), "[!] Invalid use case"
        assert (
            self.dataset_name in USE_CASES[self.use_case]
        ), "[!] Unsupported use case for this dataset"
        assert self.data_type in DATA_TYPES, "[!] Invalid data type"

    def _init_processor(self) -> None:
        """Initialize the specific processor based on dataset name and use-case."""
        if self.dataset_name == "mimiciv":
            self.processor = MimicivProcessor(self.use_case, self.data_type)

    def process_data(self):
        """Process the data based on its type."""
        if self.data_type == "tabular":
            self.processor.process_tabular()
        elif self.data_type == "temporal":
            self.processor.process_temporal()
        else:
            self.processor.process_combined()
