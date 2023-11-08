"""Test MortalityPredictionTask."""

from unittest import TestCase

from cyclops.models.catalog import create_model
from cyclops.tasks import CXRClassificationTask


class TestCXRClassificationTask(TestCase):
    """Test CXRClassificationTask class."""

    def setUp(self):
        """Set up for testing."""
        self.model_name = "densenet"
        self.model = create_model(self.model_name)
        self.test_task = CXRClassificationTask(
            {self.model_name: self.model},
        )

    def test_init(self):
        """Test initialization of CXRClassificationTask."""
        models_list = self.test_task.list_models()
        assert models_list == [self.model_name]

    def test_add_model(self):
        """Test adding a model to CXRClassificationTask."""
        self.test_task.add_model("resnet")
        models_list = self.test_task.list_models()
        assert models_list == [self.model_name, "resnet"]

    def test_get_model(self):
        """Test getting a model from CXRClassificationTask."""
        model_name, model = self.test_task.get_model(self.model_name)
        assert model_name == self.model_name
        assert model == self.model
