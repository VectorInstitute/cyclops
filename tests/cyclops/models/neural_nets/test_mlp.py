"""Tests for the MLPModel."""

import torch

from cyclops.models.catalog import create_model
from cyclops.models.neural_nets.mlp import MLPModel


def test_mlp_model_forward_pass():
    """MLPModel must be constructible and runnable with default arguments.

    Regression test: get_module("activation", activation) returns the
    activation *class* (e.g. torch.nn.ReLU), not an instance, and the
    first hidden layer was wrapped in an extra list
    (`layers = [self._layer(...)]` instead of `self._layer(...)`), both
    of which made `nn.Sequential(*layers)` raise a TypeError.
    """
    model = MLPModel(input_dim=10)
    output = model(torch.randn(4, 10))
    assert output.shape == (4, 1)


def test_mlp_model_multiple_hidden_layers():
    """Hidden-to-hidden layer dimensions must chain correctly.

    Regression test: the loop connecting hidden layers used `input_dim`
    instead of `hidden_dims[i]` for the first hidden-to-hidden
    connection, causing a shape mismatch whenever hidden_dims[0] !=
    input_dim.
    """
    model = MLPModel(input_dim=10, hidden_dims=(32, 16, 8), output_dim=2)
    linear_layers = [m for m in model.model if isinstance(m, torch.nn.Linear)]
    dims = [(layer.in_features, layer.out_features) for layer in linear_layers]
    assert dims == [(10, 32), (32, 16), (16, 8), (8, 2)]

    output = model(torch.randn(4, 10))
    assert output.shape == (4, 2)


def test_mlp_model_accepts_module_instance_as_activation():
    """Activation may be passed as an already-instantiated nn.Module."""
    model = MLPModel(input_dim=10, activation=torch.nn.Tanh())
    assert isinstance(model.activation, torch.nn.Tanh)
    output = model(torch.randn(4, 10))
    assert output.shape == (4, 1)


def test_mlp_pt_config_initializes():
    """The packaged mlp_pt config must initialize without error.

    Regression test: configs/mlp_pt.yaml set model__layer_dim, a
    leftover from the RNN/GRU/LSTM configs, which MLPModel.__init__
    doesn't accept.
    """
    wrapped_model = create_model("mlp_pt", model__input_dim=10)
    wrapped_model.initialize()
    output = wrapped_model.model_(torch.randn(4, 10))
    assert output.shape == (4, 1)
