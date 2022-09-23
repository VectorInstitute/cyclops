import inspect
from alibi_detect.cd import ContextMMDDrift, LearnedKernelDrift
from alibi_detect.utils.pytorch.kernels import DeepKernel
import torch
import torch.nn as nn
import pickle
from scipy.special import softmax
from drift_detection.baseline_models.temporal.pytorch.utils import (
    get_temporal_model,
    get_device,
)
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from alibi_detect.utils.pytorch.kernels import GaussianRBF
import importlib


def get_args(obj, kwargs):
    """
    Get valid arguments from kwargs to pass to object.

    Parameters
    ----------
    obj
        object to get arguments from.
    kwargs
        Dictionary of arguments to pass to object.

    Returns
    -------
    args
        Dictionary of valid arguments to pass to class object.
    """
    args = {}
    for key in kwargs:
        if inspect.isclass(obj):
            if key in obj.__init__.__code__.co_varnames:
                args[key] = kwargs[key]
        elif inspect.ismethod(obj) or inspect.isfunction(obj):
            if key in obj.__code__.co_varnames:
                args[key] = kwargs[key]
    return args


def get_obj_from_str(obj_str: str, **kwargs):
    """
    Get object from string.

    Parameters
    ----------
    obj_str
        String of object to get.
    kwargs
        Arguments to pass to object.

    Returns
    -------
    obj
        Object from string.
    """
    module_name, class_name = obj_str.rsplit(".", 1)
    module = importlib.import_module(module_name)
    obj = getattr(module, class_name)
    args = get_args(obj, kwargs)
    return obj(**args)


def load_model(self, model_path: str):
    """Load pre-trained model from path.

    For scikit-learn models, a pickle is loaded from disk.

    For the torch models, the "state_dict" is loaded from disk.

    """
    file_type = self.model_path.split(".")[-1]
    if file_type == "pkl" or file_type == "pickle":
        model = pickle.load(open(model_path, "rb"))
    elif file_type == "pt":
        model = torch.load(self.model_path)
    return model


def save_model(self, model, output_path: str):
    """Saves the model to disk.

    For scikit-learn models, a pickle is saved to disk.

    For the torch models, the "state_dict" is saved to disk.

    Parameters
    ----------
    output_path: String
        path to save the model to
    """
    file_type = output_path.split(".")[-1]
    if file_type == "pkl" or file_type == "pickle":
        pickle.dump(model, open(output_path, "wb"))
    elif file_type == "pt":
        torch.save(model.state_dict(), output_path)


class ContextMMDWrapper:
    """
    Wrapper for ContextMMDDrift
    """

    def __init__(
        self,
        X_s,
        backend="tensorflow",
        p_val=0.05,
        preprocess_x_ref=True,
        update_ref=None,
        preprocess_fn=None,
        x_kernel=None,
        c_kernel=None,
        n_permutations=1000,
        prop_c_held=0.25,
        n_folds=5,
        batch_size=256,
        device=None,
        input_shape=None,
        data_type=None,
        verbose=False,
        context_type="rnn",
        model_path=None,
    ):

        self.context_type = context_type
        self.model_path = model_path
        C_s = context(X_s, self.context_type, self.model_path)
        self.tester = ContextMMDDrift(X_s, C_s)

    def predict(self, X_t, **kwargs):
        C_t = context(X_t, self.context_type, self.model_path)
        return self.tester.predict(X_t, C_t, **get_args(self.tester.predict, kwargs))


class LKWrapper:
    def __init__(
        self,
        X_s,
        *,
        backend="tensorflow",
        p_val=0.05,
        preprocess_x_ref=True,
        update_x_ref=None,
        preprocess_fn=None,
        n_permutations=100,
        var_reg=0.00001,
        reg_loss_fn=lambda kernel: 0,
        train_size=0.75,
        retrain_from_scratch=True,
        optimizer=None,
        learning_rate=0.001,
        batch_size=32,
        preprocess_batch=None,
        epochs=3,
        verbose=0,
        train_kwargs=None,
        device=None,
        dataset=None,
        dataloader=None,
        data_type=None,
        kernel_a=GaussianRBF(trainable=True),
        kernel_b=GaussianRBF(trainable=True),
        eps="trainable",
        proj_type="ffnn"
    ):

        self.proj = self.choose_proj(X_s, proj_type)

        kernel = DeepKernel(self.proj, kernel_a, kernel_b, eps)

        kwargs = locals()
        args = [
            kwargs["backend"],
            kwargs["p_val"],
            kwargs["preprocess_x_ref"],
            kwargs["update_x_ref"],
            kwargs["preprocess_fn"],
            kwargs["n_permutations"],
            kwargs["var_reg"],
            kwargs["reg_loss_fn"],
            kwargs["train_size"],
            kwargs["retrain_from_scratch"],
            kwargs["optimizer"],
            kwargs["learning_rate"],
            kwargs["batch_size"],
            kwargs["preprocess_batch"],
            kwargs["epochs"],
            kwargs["verbose"],
            kwargs["train_kwargs"],
            kwargs["device"],
            kwargs["dataset"],
            kwargs["dataloader"],
            kwargs["data_type"],
        ]
        self.tester = LearnedKernelDrift(X_s, kernel, *args)

    def predict(self, X_t, **kwargs):
        return self.tester.predict(X_t, **get_args(self.tester.predict, kwargs))

    def choose_proj(self, X_s, proj_type):
        if proj_type == "rnn":
            return recurrent_neural_network("lstm", X_s.shape[-1])
        elif proj_type == "ffnn":
            return feed_forward_neural_network(X_s.shape[-1])
        elif proj_type == "cnn":
            return convolutional_neural_network(X_s.shape[-1])
        else:
            raise ValueError("Invalid projection type.")


def context(x, context_type="rnn", model_path=None):
    """
    Get context for context mmd drift detection.
    """
    device = get_device()

    if context_type == "rnn":
        model = recurrent_neural_network("lstm", x.shape[-1])
        model.load_state_dict(load_model(model_path))
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(x).to(device)).cpu().numpy()
        return softmax(logits, -1)
    elif context_type == "gmm":
        gmm = load_model(model_path)
        c_gmm_proba = gmm.predict_proba(x)
        return c_gmm_proba


def recurrent_neural_network(
    model_name,
    input_dim,
    hidden_dim=64,
    layer_dim=2,
    dropout=0.2,
    output_dim=1,
    last_timestep_only=False,
):
    model_params = {
        "device": get_device(),
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "layer_dim": layer_dim,
        "output_dim": output_dim,
        "dropout_prob": dropout,
        "last_timestep_only": last_timestep_only,
    }
    model = get_temporal_model(model_name, model_params)
    return model


def feed_forward_neural_network(input_dim):
    """
    Creates a feed forward neural network model.

    Returns
    -------
    model: torch.nn.Module
        feed forward neural network model.
    """
    ffnn = nn.Sequential(
        nn.Linear(input_dim, 16),
        nn.SiLU(),
        nn.Linear(16, 8),
        nn.SiLU(),
        nn.Linear(8, 1),
    )
    return ffnn


def convolutional_neural_network(input_dim):
    """
    Creates a convolutional neural network model.

    Returns
    -------
    torch.nn.Module
        convolutional neural network for dimensionality reduction.
    """
    cnn = nn.Sequential(
        nn.Conv2d(input_dim, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(8, 16, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Flatten(),
    )
    return cnn
