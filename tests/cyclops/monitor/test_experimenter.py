"""integration tests for Experimenter module."""

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import Dataset

from cyclops.monitor.clinical_applicator import ClinicalShiftApplicator
from cyclops.monitor.detector import Detector
from cyclops.monitor.experimenter import Experimenter
from cyclops.monitor.reductor import Reductor
from cyclops.monitor.synthetic_applicator import SyntheticShiftApplicator
from cyclops.monitor.tester import TSTester


def synthetic_gemini_dataset(size=1000):
    """Create a synthetic Gemini dataset."""
    gemini_columns = [
        "encounter_id",
        "subject_id",
        "city",
        "province",
        "country",
        "language",
        "total_direct_cost",
        "total_indirect_cost",
        "total_cost",
        "hospital_id",
        "sex",
        "age",
        "admit_timestamp",
        "discharge_timestamp",
        "admit_category",
        "discharge_disposition",
        "responsibility_for_payment",
        "province_territory_issuing_health_card_number",
        "number_of_alc_days",
        "institution_from",
        "institution_from_type",
        "institution_to",
        "institution_to_type",
        "readmission",
        "residence_code",
        "admitting_service_raw",
        "discharging_service_raw",
        "mrp_service",
        "cmg",
        "admitting_physician",
        "discharging_physician",
        "mrp",
        "admitting_physician_gim",
        "discharging_physician_gim",
        "mrp_gim",
        "admitting_service_mapped",
        "discharging_service_mapped",
        "from_nursing_home_mapped",
        "from_acute_care_institution_mapped",
        "los_derived",
        "del_present",
        "gemini_cohort",
        "discharge_description",
        "admit_via_ambulance",
        "triage_level",
        "physician_initial_assessment_date_time",
        "er_admit_timestamp",
        "disposition_date_time",
        "er_discharge_timestamp",
        "length_of_stay_in_er",
        "mortality",
    ]

    df = pd.DataFrame(columns=gemini_columns)

    df["encounter_id"] = np.random.randint(0, 100000, size=size)
    df["subject_id"] = np.random.randint(0, 100000, size=size)
    df["city"] = np.random.choice(["Toronto", "Ottawa", "Montreal"], size=size)
    df["province"] = np.random.choice(["Ontario", "Quebec", "Alberta"], size=size)
    df["country"] = np.random.choice(["Canada", "USA", "Mexico"], size=size)
    df["language"] = np.random.choice(["English", "French", "Spanish"], size=size)
    df["total_direct_cost"] = np.random.randint(0, 100000, size=size)
    df["total_indirect_cost"] = np.random.randint(0, 100000, size=size)
    df["total_cost"] = np.random.randint(0, 100000, size=size)
    df["hospital_id"] = np.random.choice(["SMH", "MSH", "THPC", "THPM", "UHNTG", "UHNTW", "PMH", "SBK"], size=size)
    df["sex"] = np.random.choice(["M", "F"], size=size)
    df["age"] = np.random.randint(0, 100, size=size)
    df["admit_timestamp"] = pd.date_range(
        start="1/1/2015", end="8/1/2020", periods=size
    )
    df["discharge_timestamp"] = pd.date_range(
        start="1/1/2015", end="8/1/2020", periods=size
    )
    df["mortality"] = np.random.randint(0, 2, size=size)

    X = np.random.rand(size, 64, 7)

    # huggingface arrowtable dataset with dataframe and numpy array
    

    return df, X


@pytest.fixture(name="gemini_dataset")
def fixture_gemini_dataset():
    """Create a test input for GEMINI use-case."""
    dataset = synthetic_gemini_dataset()
    return dataset


@pytest.fixture(name="txrv_dataset")
def fixture_txrv_dataset():
    """Create a test input for NIH use-case."""

    class TXRVDataset(Dataset):
        """TXRV Dummy Dataset."""

        def __init__(self, num_samples, channels, height, width, num_labels=14):
            self.len = num_samples
            self.data = torch.rand(num_samples, channels, height, width)
            self.labels = torch.rand(num_samples, num_labels)

        def __getitem__(self, index):
            item = {"img": self.data[index], "lab": self.labels[index]}
            return item

        def __len__(self):
            return self.len

    dataset = TXRVDataset(8, 1, 224, 224)
    metadata = pd.DataFrame(np.random.randint(0, 2, size=(8, 2)), columns=list("AB"))
    metadata_mapping = {"A": "A", "B": "B"}
    return dataset, metadata, metadata_mapping


# test gemini use-case with pca reductor and mmd tester and
# clinical shift applicator for hospital_type w/ synthetic_gemini_dataset
@pytest.mark.integration_test
def test_experimenter_gemini_pca_mmd(gemini_dataset):
    """Test Experimenter."""
    X = gemini_dataset
    reductor = Reductor("PCA", n_components=2)
    tester = TSTester("mmd")
    detector = Detector(reductor, tester)
    source = ["2015-01-01", "2017-06-01"]
    target = ["2017-06-01", "2020-01-01"]
    applicator = ClinicalShiftApplicator("time", source, target)
    experimenter = Experimenter("sensitivity_test", detector, applicator)
    results = experimenter.run(X)

    assert list(results.keys()) == [
        "samples",
        "mean_p_vals",
        "std_p_vals",
        "mean_dist",
        "std_dist",
    ]


# test nih use-case with tae_trxv_cnn reductor and mmd tester and
# synthetic shift applicator for gaussian noise w/ txrv_dataset
@pytest.mark.integration_test
def test_experimenter_nih_tae_trxv_cnn_mmd(txrv_dataset):
    """Test Experimenter."""
    reductor = Reductor("TAE_txrv_CNN")
    tester = TSTester("mmd")
    dataset, metadata, metadata_mapping = txrv_dataset
    detector = Detector(reductor, tester)
    applicator = SyntheticShiftApplicator("gn_shift")
    experimenter = Experimenter("sensitivity_test", detector, applicator)
    results = experimenter.run(dataset)

    assert list(results.keys()) == [
        "samples",
        "mean_p_vals",
        "std_p_vals",
        "mean_dist",
        "std_dist",
    ]
