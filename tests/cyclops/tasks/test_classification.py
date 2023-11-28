"""Test classification tasks classes."""

import os
import shutil
from unittest import TestCase

from cyclops.models.catalog import create_model
from cyclops.tasks import (
    BinaryTabularClassificationTask,
    MultilabelImageClassificationTask,
)


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class TestBinaryTabularClassificationTask(TestCase):
    """Test BinaryTabularClassificationTask class."""

    def setUp(self):
        """Set up for testing."""
        self.model_name = "mlp"
        self.model = create_model(self.model_name)
        self.test_task = BinaryTabularClassificationTask(
            {self.model_name: self.model},
            ["feature_1", "feature_2"],
            "target",
        )

    def test_init(self):
        """Test initialization of BinaryTabularClassificationTask."""
        assert self.test_task.task_type == "binary"
        assert self.test_task.data_type == "tabular"
        models_list = self.test_task.list_models()
        assert models_list == [self.model_name]

    def test_add_model(self):
        """Test adding a model to BinaryTabularClassificationTask."""
        self.test_task.add_model("rf_classifier")
        models_list = self.test_task.list_models()
        assert models_list == [self.model_name, "rf_classifier"]

    def test_get_model(self):
        """Test getting a model from BinaryTabularClassificationTask."""
        model_name, model = self.test_task.get_model(self.model_name)
        assert model_name == self.model_name
        assert model == self.model

    def test_save_and_load_model(self):
        """Test saving and loading model from BinaryTabularClassificationTask."""
        self.test_task.save_model(
            f"{CURRENT_DIR}/test_model",
            self.model_name,
        )
        model_name = self.model.model.__name__
        model = self.test_task.load_model(
            f"{CURRENT_DIR}/test_model/{model_name}/model.pkl",
        )
        assert model == self.model
        assert os.path.isfile(f"{CURRENT_DIR}/test_model/{model_name}/model.pkl")
        shutil.rmtree(f"{CURRENT_DIR}/test_model")


class TestMultilabelImageClassificationTask(TestCase):
    """Test MultilabelImageClassificationTask class."""

    def setUp(self):
        """Set up for testing."""
        self.model_name = "resnet"
        self.model = create_model(self.model_name)
        self.test_task = MultilabelImageClassificationTask(
            {self.model_name: self.model},
            ["image"],
            ["target_1", "target_2"],
        )

    def test_init(self):
        """Test initialization of MultilabelImageClassificationTask."""
        assert self.test_task.task_type == "multilabel"
        assert self.test_task.data_type == "image"
        models_list = self.test_task.list_models()
        assert models_list == [self.model_name]

    def test_add_model(self):
        """Test adding a model to MultilabelImageClassificationTask."""
        self.test_task.add_model("densenet")
        models_list = self.test_task.list_models()
        assert models_list == [self.model_name, "densenet"]

    def test_get_model(self):
        """Test getting a model from MultilabelImageClassificationTask."""
        model_name, model = self.test_task.get_model(self.model_name)
        assert model_name == self.model_name
        assert model == self.model

    def test_save_and_load_model(self):
        """Test saving and loading model from MultilabelImageClassificationTask."""
        self.test_task.save_model(f"{CURRENT_DIR}/test_model", self.model_name)
        model = self.test_task.load_model(
            f"{CURRENT_DIR}/test_model/model.pt",
        )
        assert model == self.model
        assert os.path.isfile(f"{CURRENT_DIR}/test_model/model.pt")
        shutil.rmtree(f"{CURRENT_DIR}/test_model")
