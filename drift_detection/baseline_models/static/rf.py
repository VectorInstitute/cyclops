"""Train a random forest model on the data.

Train a random forest model on the data for a series of different values of n_estimaters
and returns the best model using auroc metric.

"""
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import roc_auc_score


def fit_rf(X, Y, Xv, Yv):
    """Train a random forest model on the data and return best model."""
    best_n = None
    best_score = 0
    best_model = None
    for n in [5, 10, 50, 100, 500]:
        m = RF(n_estimators=n, n_jobs=-1)
        print("Fitting model with n:", n)
        m.fit(X, Y)
        Pv = m.predict_proba(Xv)[:, 1]
        score = roc_auc_score(Yv, Pv)
        if score > best_score:
            best_score = score
            best_model = m
            best_n = n

    print("Best n:", best_n)
    return best_model
