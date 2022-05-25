import shap
import numpy as np


class ShiftExplainer:
    def __init__(self, model):
        self.model = model
        self.explainer = None

    def get_explainer(self):
        explainer = shap.Explainer(self.model)
        # background = shap.maskers.Independent(X)
        # explainer = shap.Explainer(self.model, background)
        self.explainer = explainer

    def get_shap_values(self, X):
        shap_values = self.explainer(X, check_additivity=False)
        return shap_values

    def plot_dependence(self, feat, shap_values, X):
        shap.dependence_plot(feat, shap_values, X)

    def plot_summary(self, shap_values, X):
        shap.summary_plot(shap_values, X)

    def plot_waterfall(self, shap_values):
        shap.plots.waterfall(shap_values, max_display=20)
