"""Train a logistic regression model on the data.

Train a logistic regression model on the data for a series of different values of l2
strength and return the best model using auroc metric.

"""
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import roc_auc_score


def fit_lr(X, y, X_val, y_val):
    """Train a logistic regression model on the data and return best model."""
    best_l2_strength = None
    best_score = 0
    best_model = None
    for l2_strength in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
        model = LR(C=l2_strength)
        print("Fitting model with l2 strength:", l2_strength)
        model.fit(X, y)
        pred_val = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, pred_val)
        if score > best_score:
            best_score = score
            best_model = model
            best_l2_strength = l2_strength

    print("Best l2 strength:", best_l2_strength)
    return best_model
