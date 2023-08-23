"""Test MortalityPredictionTask."""

from unittest import TestCase

from cyclops.models.catalog import create_model
from cyclops.tasks.mortality_prediction import MortalityPredictionTask


class TestMortalityPredictionTask(TestCase):
    """Test MortalityPredictionTask class."""

    def setUp(self):
        """Set up for testing."""
        self.model_name = "mlp"
        self.model = create_model(self.model_name)
        self.test_task = MortalityPredictionTask(
            {self.model_name: self.model},
        )

    def test_init(self):
        """Test initialization of MortalityPredictionTask."""
        models_list = self.test_task.list_models()
        self.assertEqual(models_list, [self.model_name])

    def test_add_model(self):
        """Test adding a model to MortalityPredictionTask."""
        self.test_task.add_model("rf_classifier")
        models_list = self.test_task.list_models()
        self.assertEqual(models_list, [self.model_name, "rf_classifier"])

    def test_get_model(self):
        """Test getting a model from MortalityPredictionTask."""
        model_name, model = self.test_task.get_model(self.model_name)
        self.assertEqual(model_name, self.model_name)
        self.assertEqual(model, self.model)
