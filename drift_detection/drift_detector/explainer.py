import numpy as np
import shap

class Explainer:
    
    """ShiftExplainer Class.

    Attributes
    ----------
    model: sklearn, tf, pytorch
        Model for which to build explainer
        
    """
    
    def __init__(self, model, explainer_type=None):
        self.model = model
        self.explainer_type = explainer_type
        self.explainer = get_explainer()

    def get_explainer(self):
        if self.explainer_type == "tree":
            self.explainer = shap.TreeExplainer(self.model)
        elif self.explainer_type == "deep":
            self.explainer = shap.DeepExplainer(self.model)
        elif self.explainer_type == "gradient":
            self.explainer = shap.GradientExplainer(self.model)
        else:
            self.explainer = shap.Explainer(self.model)

    def get_shap_values(self, X):
        shap_values = self.explainer(X, check_additivity=False)
        return shap_values

    def plot_dependence(self, feat, shap_values, X):
        shap.dependence_plot(feat, shap_values, X)

    def plot_summary(self, shap_values, X):
        shap.summary_plot(shap_values, X)

    def plot_waterfall(self, shap_values,max_display=20):
        shap.plots.waterfall(shap_values, max_display=max_display)

    def plot_beeswarm(self, shap_values):
        shap.plots.beeswarm(shap_values)
    
    def plot_heatmap(self, shap_values):
        shap.plots.heatmap(shap_values)
        
    def plot_dependence(self, variable, shap_values, X):
        shap.dependence_plot(variable, shap_values, X)