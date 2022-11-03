"""Explainer module that uses shap values to explain the drift detected across features.

This module is used to explain the drift detected across features. It uses the
difference in shap values for a chosen domain classifier to provide insight into the
most significant features in the drift detection.

"""
import shap


class Explainer:
    """ShiftExplainer Class.

    Attributes
    ----------
    model: sklearn, tf, pytorch
        Model for which to build explainer
    explainer_type:
        Type of shap explainer to use for model.

    """

    def __init__(self, model, data=None, explainer_type=None):
        self.model = model
        self.data = data
        self.explainer_type = explainer_type
        self.explainer = self.get_explainer()

    def get_explainer(self):
        """Get the explainer for the model."""
        if self.explainer_type == "tree":
            explainer = shap.TreeExplainer(self.model, self.data)
        elif self.explainer_type == "deep":
            explainer = shap.DeepExplainer(self.model, self.data)
        elif self.explainer_type == "gradient":
            explainer = shap.GradientExplainer(self.model, self.data)
        else:
            explainer = shap.Explainer(self.model)
        return explainer

    def get_shap_values(self, X):
        """Get the shap values for the model."""
        shap_values = self.explainer(X)
        return shap_values

    def plot_dependence(self, feat, shap_values, X):
        """Plot the dependence of a feature on the model output."""
        shap.dependence_plot(feat, shap_values, X)

    def plot_summary(self, shap_values, X):
        """Plot the summary of the shap values."""
        shap.summary_plot(shap_values, X)

    def plot_waterfall(self, shap_values, max_display=20):
        """Plot the waterfall plot of the shap values."""
        shap.plots.waterfall(shap_values, max_display=max_display)

    def plot_beeswarm(self, shap_values):
        """Plot the beeswarm plot of the shap values."""
        shap.plots.beeswarm(shap_values)

    def plot_heatmap(self, shap_values):
        """Plot the heatmap of the shap values."""
        shap.plots.heatmap(shap_values)
