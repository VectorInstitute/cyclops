"""Train a random forest model on the data.

Train a random forest model on the data for a series of different values of n_estimaters
and returns the best model using auroc metric.

"""
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import roc_auc_score


def fit_rf(X, y, X_val, y_val):
    """Train a random forest model on the data and return best model."""
    best_n_est = None
    best_score = 0
    best_model = None
    for n_est in [5, 10, 50, 100, 500]:
        model = RF(n_estimators=n_est, n_jobs=-1)
        print("Fitting model with n:", n_est)
        model.fit(X, y)
        pred_val = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, pred_val)
        if score > best_score:
            best_score = score
            best_model = model
            best_n_est = n_est

    print("Best n_est:", best_n_est)
    return best_model
