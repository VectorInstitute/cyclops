"""Train a GaussianProcessClassifier model on the data.

Train a GaussianProcessClassifier model on the data for a series of different kernels
and return the best model using auroc metric.

"""
from sklearn.gaussian_process import GaussianProcessClassifier as GPC
from sklearn.gaussian_process.kernels import (
    RBF,
    DotProduct,
    Matern,
    RationalQuadratic,
    WhiteKernel,
)
from sklearn.metrics import roc_auc_score


def fit_gp(X, y, X_val, y_val):
    """Train a GaussianProcessClassifier model on the data and return best model."""
    best_kernel = None
    best_score = 0
    best_model = None

    for kernel in [
        1 * RBF(),
        1 * DotProduct(),
        1 * Matern(),
        1 * RationalQuadratic(),
        1 * WhiteKernel(),
    ]:
        model = GPC(kernel=kernel, n_jobs=-1)
        model.fit(X, y)
        pred_val = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, pred_val)
        print("Fitted GPC with K:", kernel, "AUC:", score)
        if score > best_score:
            best_score = score
            best_model = model
            best_kernel = kernel

    print("Best K:", best_kernel)
    return best_model
