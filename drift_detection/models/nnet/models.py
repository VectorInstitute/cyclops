import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import logging
import warnings
import sys

from utils_prediction.nn.metric_logging import MetricLogger
from utils_prediction.nn.layers import (
    SparseLinear,
    LinearLayer,
    SequentialLayers,
    FeedforwardNet,
    EmbeddingBagLinear,
)

from utils_prediction.nn.pytorch_metrics import weighted_cross_entropy_loss


class TorchModel:
    """
    This is the upper level class that provides training and logging code for a Pytorch model.
    To initialize the model, provide a config_dict with relevant parameters.
    The default model is logistic regression. Subclass and override init_model() for custom usage.
    The user is intended to interact with this class primarily through the train method.
    """

    def __init__(self, *args, model_override=None, **kwargs):
        self.config_dict = self.get_default_config()
        self.config_dict = self.override_config(**kwargs)
        self.check_config_dict()
        self.initialize_logging()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(self.device)
        if model_override is None:
            self.model = self.init_model()
        else:
            self.model = model_override

        self.model.apply(self.weights_init)
        self.model.to(self.device)
        self.optimizer = self.init_optimizer()
        self.scheduler = self.init_scheduler()
        self.criterion = self.init_loss()
        self.metric_comparator = self.init_metric_comparator()

    def get_default_config(self):
        """
        Defines default hyperparameters
        """
        return {
            "input_dim": None,
            "lr": 1e-4,
            "num_epochs": 10,
            "selection_metric": "loss",
            "batch_size": 256,
            "output_dim": 2,
            "iters_per_epoch": 100,
            "gamma": None,
            "early_stopping": False,
            "early_stopping_patience": 5,
            "print_every": 1,
            "weighted_loss": False,
            "has_optimizers_aux": False,
            "print_grads": False,
            "weight_decay": 0.0,
            "verbose": True,
            "weighted_evaluation": False,
            "logging_evaluate_by_group": False,
            "logging_metrics": ["auc", "auprc", "brier", "loss_bce"],
            "logging_threshold_metrics": None,
            "logging_thresholds": [0.5],
            "logging_path": None,
            "disable_metric_logging": False,
            "compute_group_min_max": False,
        }

    def override_config(self, **override_dict):
        """
        Updates the config_dict with elements of override_dict
        """
        return {**self.config_dict, **override_dict}

    def initialize_logging(self):
        logging.basicConfig(stream=sys.stdout, level="INFO", format="%(message)s")

    def check_config_dict(self):
        if self.config_dict.get("input_dim") is None:
            raise ValueError("Must provide input_dim")

        if (
            ("_min" in self.config_dict.get("selection_metric"))
            or ("_max" in self.config_dict.get("selection_metric"))
        ) and (
            (
                not self.config_dict.get("compute_group_min_max")
                or (not self.config_dict.get("logging_evaluate_by_group"))
            )
        ):
            logging.warning('Warning: selection metric requires compute_group_min_max and logging_evaluate_by_group be set to True. Overwriting provided parameters')
            self.config_dict['compute_group_min_max'] = True
            self.config_dict['logging_evaluate_by_group'] = True

    # def get_logger(self, logging_path=None):
    #     """
    #     Currently unused
    #     """
    #     logger = logging.getLogger(__name__)
    #     logger.setLevel("INFO")

    #     if logging_path is not None:
    #         logger.addHandler(logging.FileHandler(logging_path))
    #     else:
    #         logger.addHandler(logging.StreamHandler(sys.stdout))

    #     return logger

    def transform_batch(self, batch, keys=None):
        """
        Sends a batch to the device
        Provide keys to only send a subset of batch keys to the device
        """
        if keys is None:
            keys = batch.keys()

        result = {}
        for key in batch.keys():
            if (isinstance(batch[key], torch.Tensor)) and (key in keys):
                result[key] = batch[key].to(self.device, non_blocking=True)
            elif isinstance(batch[key], dict):
                result[key] = self.transform_batch(batch[key])
            else:
                result[key] = batch[key]

        return result

    def get_transform_batch_keys(self):
        """
        Returns the names of the list of tensors that are sent to device
        """
        result = ["features", "labels"]
        if self.config_dict.get("weighted_loss"):
            result = result + ["weights"]
        return result

    def get_logging_keys(self):
        result = ["outputs", "pred_probs", "labels", "row_id"]
        if self.config_dict.get("weighted_evaluation"):
            result = result + ["weights"]
        if self.config_dict.get("logging_evaluate_by_group"):
            result = result + ["group"]
        return result

    @staticmethod
    def weights_init(m):
        """
        Initialize the weights with Xavier initialization
        By default, linear and EmbeddingBag layers are initialized
        """
        if (
            isinstance(m, nn.Linear)
            or isinstance(m, nn.EmbeddingBag)
            or isinstance(m, nn.Embedding)
            or isinstance(m, SparseLinear)
        ):
            nn.init.xavier_normal_(m.weight)

    def init_model(self):
        """
        Initializes the model with an instance of torch.nn.Module
        Override this to customize
        """
        return LinearLayer(
            self.config_dict["input_dim"], self.config_dict["output_dim"]
        )

    def init_optimizer(self):
        """
        Initialize an optimizer
        """
        params = [{"params": self.model.parameters()}]
        optimizer = torch.optim.Adam(
            params,
            lr=self.config_dict["lr"],
            weight_decay=self.config_dict["weight_decay"],
        )
        return optimizer

    def init_scheduler(self):
        """
        A learning rate scheduler
        """
        gamma = self.config_dict.get("gamma")
        if gamma is None:
            return None
        else:
            return torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)

    # @staticmethod
    # def weighted_cross_entropy_loss(input, target, sample_weight=None, **kwargs):
    #     """
    #     A method that computes a sample weighted cross entropy loss
    #     """
    #     if sample_weight is None:
    #         return F.cross_entropy(input, target, reduction="mean")
    #     else:
    #         result = F.cross_entropy(input, target, reduction="none", **kwargs)
    #         assert result.size()[0] == sample_weight.size()[0]
    #         return (sample_weight * result).sum() / sample_weight.sum()

    def compute_grad_norm(self, loss):
        if self.config_dict.get("skip_input_grad"):
            parameters = (
                p[1]
                for p in self.model.named_parameters()
                if "layers.0.linear" not in p[0]
            )
        else:
            parameters = (p[1] for p in self.model.named_parameters())

        grads = torch.autograd.grad(outputs=loss, inputs=parameters, create_graph=True)
        result = (
            torch.cat(tuple(torch.linalg.norm(grad).reshape(-1) ** 2 for grad in grads))
            .sum()
            .sqrt()
        )
        return result

    def init_loss(self):
        """
        Returns the loss function
        """
        if self.config_dict.get("weighted_loss"):
            return weighted_cross_entropy_loss
        else:
            return F.cross_entropy

    class MetricComparator:
        """
        A class that can be used to compare metrics
        """

        def __init__(self, metric_type="min"):
            self.metric_type = metric_type

        def is_better(self, value, other):
            if self.metric_type == "min":
                return value < other
            elif self.metric_type == "max":
                return value > other

        def get_inital_value(self):
            if self.metric_type == "min":
                return 1e18
            elif self.metric_type == "max":
                return -1e18

    def init_metric_comparator(self):
        """
        Initializes a metric comparator
        """
        if "loss" in self.config_dict["selection_metric"] or "supervised" in self.config_dict["selection_metric"]:
            comparator = self.MetricComparator("min")
        else:
            comparator = self.MetricComparator("max")

        return comparator

    def get_loss_names(self):
        """
        Defines the names of the losses that will be tracked
        """
        return ["loss"]

    def forward_on_batch(self, the_data):
        """
        Run the forward pass, returning a batch_loss_dict and outputs
        """
        outputs = self.model(the_data["features"])
        if self.config_dict.get("weighted_loss"):
            loss_dict_batch = {
                "loss": self.criterion(
                    outputs, the_data["labels"], sample_weight=the_data["weights"]
                )
            }
        else:
            loss_dict_batch = {"loss": self.criterion(outputs, the_data["labels"])}
        return loss_dict_batch, outputs

    def zero_optimizers_aux(self):
        """
        Zeros any auxiliary optimizers
        """
        raise NotImplementedError

    def update_models_aux(self, the_data):
        """
        Update any auxiliary models
        """
        raise NotImplementedError

    def print_grads(self):
        """
        Prints grads
        """
        raise NotImplementedError

    def train(self, loaders, **kwargs):
        """
        Method that trains the model.
            Args:
                loaders: A dictionary of DataLoaders with keys corresponding to phases
                kwargs: Additional arguments to override in the config_dict
            Returns:
                result_dict: A dictionary with metrics recorded every epoch
        """

        self.config_dict = self.override_config(**kwargs)
        best_performance = self.metric_comparator.get_inital_value()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        metric_logger = MetricLogger(
            metrics=self.config_dict.get("logging_metrics"),
            threshold_metrics=self.config_dict.get("logging_threshold_metrics"),
            thresholds=self.config_dict.get("logging_thresholds"),
            losses=self.get_loss_names(),
            output_dict_keys=self.get_logging_keys(),
            weighted_evaluation=self.config_dict.get("weighted_evaluation"),
            evaluate_by_group=self.config_dict.get("logging_evaluate_by_group"),
            disable_metric_logging=self.config_dict.get("disable_metric_logging"),
            compute_group_min_max=self.config_dict.get("compute_group_min_max"),
        )
        phases = kwargs.get("phases", ["train", "val"])
        epochs_since_improvement = 0
        best_epoch = 0
        for epoch in range(self.config_dict["num_epochs"]):
            if self.config_dict["early_stopping"] & (
                epochs_since_improvement >= self.config_dict["early_stopping_patience"]
            ):
                logging.info(
                    "Early stopping at epoch {epoch} with best epoch {best_epoch}".format(
                        epoch=epoch - 1, best_epoch=best_epoch
                    )
                )
                break
            if epoch % self.config_dict["print_every"] == 0:
                logging.info(
                    "Epoch {}/{}".format(epoch, self.config_dict["num_epochs"] - 1)
                )
                logging.info("-" * 10)
            for phase in phases:
                self.model.train(phase == "train")
                metric_logger.init_metric_dicts()
                for i, the_data in enumerate(loaders[phase]):
                    self.optimizer.zero_grad()
                    if self.config_dict.get("has_optimizers_aux"):
                        self.zero_optimizers_aux()

                    the_data = self.transform_batch(
                        the_data, keys=self.get_transform_batch_keys()
                    )
                    loss_dict_batch, outputs = self.forward_on_batch(the_data)

                    if phase == "train":
                        loss_dict_batch["loss"].backward()
                        if self.config_dict["print_grads"]:
                            self.print_grads()

                        self.optimizer.step()
                        if self.config_dict.get("has_optimizers_aux"):
                            self.optimizer.zero_grad()
                            self.zero_optimizers_aux()
                            self.update_models_aux(the_data)

                    metric_logger.update_loss_dict(
                        loss_dict_batch, batch_size=the_data["labels"].shape[0]
                    )
                    metric_logger.update_output_dict(outputs=outputs, **the_data)

                if phase == "train":
                    if self.scheduler is not None:
                        self.scheduler.step()

                epoch_performance = metric_logger.compute_metrics_epoch(phase=phase)
                if epoch % self.config_dict["print_every"] == 0:
                    logging.info("Phase: {}:".format(phase))
                    metric_logger.print_metrics()

                if phase == "val":
                    if self.metric_comparator.is_better(
                        epoch_performance[self.config_dict["selection_metric"]],
                        best_performance,
                    ):
                        if self.config_dict.get("verbose"):
                            logging.info("Best model updated")
                        best_epoch = epoch
                        best_performance = epoch_performance[
                            self.config_dict["selection_metric"]
                        ]
                        best_model_wts = copy.deepcopy(self.model.state_dict())
                        best_optimizer_state = copy.deepcopy(
                            self.optimizer.state_dict()
                        )
                        if self.scheduler is not None:
                            best_scheduler_state = copy.deepcopy(
                                self.scheduler.state_dict()
                            )
                        epochs_since_improvement = 0
                    else:
                        epochs_since_improvement += 1

        if "val" in phases:
            logging.info("Best performance: {:4f}".format(best_performance))
            self.model.load_state_dict(best_model_wts)
            self.optimizer.load_state_dict(best_optimizer_state)
            if self.scheduler is not None:
                self.scheduler.load_state_dict(best_scheduler_state)

        self.epoch = epoch
        return {
            "performance": metric_logger.get_evaluation_overall(),
        }

    def predict(self, loaders, phases=["test"], return_outputs=True):
        """
        Method that trains the model.
            Args:
                loaders: A dictionary of DataLoaders with keys corresponding to phases
                kwargs: Additional arguments to override in the config_dict
            Returns:
                result_dict: A dictionary with metrics recorded every epoch
        """
        metric_logger = MetricLogger(
            phases=phases,
            metrics=self.config_dict.get("logging_metrics"),
            threshold_metrics=self.config_dict.get("logging_threshold_metrics"),
            losses=self.get_loss_names(),
            output_dict_keys=self.get_logging_keys(),
            weighted_evaluation=self.config_dict.get("weighted_evaluation"),
            evaluate_by_group=self.config_dict.get("logging_evaluate_by_group"),
            compute_group_min_max=self.config_dict.get("compute_group_min_max"),
        )
        self.model.train(False)
        output_dict = {}
        for phase in phases:
            logging.info("Evaluating on phase: {phase}".format(phase=phase))
            metric_logger.init_metric_dicts()
            for i, the_data in enumerate(loaders[phase]):
                the_data = self.transform_batch(
                    the_data, keys=self.get_transform_batch_keys()
                )
                loss_dict_batch, outputs = self.forward_on_batch(the_data)
                metric_logger.update_loss_dict(
                    loss_dict_batch, batch_size=the_data["labels"].shape[0]
                )
                metric_logger.update_output_dict(outputs=outputs, **the_data)

            _ = metric_logger.compute_metrics_epoch(phase=phase)

            if return_outputs:
                output_dict[phase] = metric_logger.get_output_df()

        result_dict = {"performance": metric_logger.get_evaluation_overall()}
        if return_outputs:
            result_dict["outputs"] = (
                pd.concat(output_dict)
                .reset_index(level=-1, drop=True)
                .rename_axis("phase")
                .reset_index()
            )
        return result_dict

    def load_weights(self, the_path):
        """
        Load model weights from file
        """
        self.model.load_state_dict(torch.load(the_path, map_location=self.device))

    def save_weights(self, the_path):
        """
        Load model weights from a file
        """
        torch.save(self.model.state_dict(), the_path)

    def is_training(self):
        return self.model.training


class FeedforwardNetModel(TorchModel):
    """
    The primary class for a feedforward network.
    Has options for sparse inputs, residual connections, dropout, and layer normalization.
    """

    def get_default_config(self):
        """
        Defines the default hyperparameters
        """
        config_dict = super().get_default_config()
        update_dict = {
            "hidden_dim_list": [128],
            "drop_prob": 0.0,
            "normalize": False,
            "sparse": True,
            "sparse_mode": "csr",  # alternatively, "convert"
            "resnet": False,
        }

        return {**config_dict, **update_dict}

    def init_model(self):
        model = FeedforwardNet(
            in_features=self.config_dict["input_dim"],
            hidden_dim_list=self.config_dict["hidden_dim_list"],
            output_dim=self.config_dict["output_dim"],
            drop_prob=self.config_dict["drop_prob"],
            normalize=self.config_dict["normalize"],
            sparse=self.config_dict["sparse"],
            sparse_mode=self.config_dict["sparse_mode"],
            resnet=self.config_dict["resnet"],
        )
        return model


class FixedWidthModel(FeedforwardNetModel):
    """
    The primary class for a feedforward network with a fixed number of hidden layers of equal size.
    Has options for sparse inputs, residual connections, dropout, and layer normalization.
    """

    def get_default_config(self):
        """
        Default hyperparameters.
        Uses num_hidden and hidden_dim to construct a hidden_dim_list
        """
        config_dict = super().get_default_config()
        update_dict = {"num_hidden": 1, "hidden_dim": 128}
        update_dict["hidden_dim_list"] = update_dict["num_hidden"] * [
            update_dict["hidden_dim"]
        ]

        return {**config_dict, **update_dict}

    def init_model(self):
        """
        Initializes a FeedforwardNet
        """
        model = FeedforwardNet(
            in_features=self.config_dict["input_dim"],
            hidden_dim_list=self.config_dict["num_hidden"]
            * [self.config_dict["hidden_dim"]],
            output_dim=self.config_dict["output_dim"],
            drop_prob=self.config_dict["drop_prob"],
            normalize=self.config_dict["normalize"],
            sparse=self.config_dict["sparse"],
            sparse_mode=self.config_dict["sparse_mode"],
            resnet=self.config_dict["resnet"],
        )
        return model


class BottleneckModel(FeedforwardNetModel):
    """
    A feedforward net where the layers progressively decrease in size
    """

    def get_default_config(self):
        config_dict = super().get_default_config()
        update_dict = {"bottleneck_size": 128, "num_hidden": 2}
        return {**config_dict, **update_dict}

    def init_model(self):

        hidden_dim_list = [
            self.config_dict["bottleneck_size"] * (2 ** (i))
            for i in reversed(range(self.config_dict["num_hidden"]))
        ]

        model = FeedforwardNet(
            in_features=self.config_dict["input_dim"],
            hidden_dim_list=hidden_dim_list,
            output_dim=self.config_dict["output_dim"],
            drop_prob=self.config_dict["drop_prob"],
            normalize=self.config_dict["normalize"],
            sparse=self.config_dict["sparse"],
            sparse_mode=self.config_dict["sparse_mode"],
            resnet=self.config_dict["resnet"],
        )
        return model


class SparseLogisticRegression(TorchModel):
    """
    A model that perform sparse logistic regression
    """

    def init_model(self):
        layer = SparseLinear(
            self.config_dict["input_dim"], self.config_dict["output_dim"]
        )
        model = SequentialLayers([layer])
        return model


class SparseLogisticRegressionEmbed(TorchModel):
    """
    A model that performs sparse logistic regression with an EmbeddingBag encoder
    """

    def init_model(self):
        layer = EmbeddingBagLinear(
            self.config_dict["input_dim"], self.config_dict["output_dim"]
        )
        model = SequentialLayers([layer])
        return model


class BilevelModel(TorchModel):
    """
    A generic class to support auxiliary optimizers and models
    """

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.models_aux = self.init_models_aux()
        self.optimizers_aux = self.init_optimizers_aux()

    def get_default_config(self):
        config_dict = super().get_default_config()
        update_dict = {"has_optimizers_aux": True}
        return {**config_dict, **update_dict}

    def init_optimizers_aux(self):
        raise NotImplementedError

    def zero_optimizers_aux(self):
        for optimizer in self.optimizers_aux.values():
            optimizer.zero_grad()

    def init_models_aux(self):
        raise NotImplementedError

    def update_models_aux(self, the_data):
        raise NotImplementedError
