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


def fit_gp(X, Y, Xv, Yv):
    """Train a GaussianProcessClassifier model on the data and return best model."""
    best_c = None
    best_score = 0
    best_model = None

    for K in [
        1 * RBF(),
        1 * DotProduct(),
        1 * Matern(),
        1 * RationalQuadratic(),
        1 * WhiteKernel(),
    ]:
        m = GPC(kernel=K, n_jobs=-1)
        m.fit(X, Y)
        Pv = m.predict_proba(Xv)[:, 1]
        score = roc_auc_score(Yv, Pv)
        print("Fitted GPC with K:", K, "AUC:", score)
        if score > best_score:
            best_score = score
            best_model = m
            best_c = K

    print("Best K:", best_c)
    return best_model
