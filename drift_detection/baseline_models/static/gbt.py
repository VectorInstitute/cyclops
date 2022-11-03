"""Train a xgboost model on the data.

Train a xgboost model on the data for a series of different values of max_depth and
gamma and returns the best model using auroc metric.

"""
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


def fit_gbt(X, y, X_val, y_val):
    """Train a xgboost model on the data and return best model."""
    best_max_depth = None
    best_gamma = None
    best_score = 0
    best_model = None
    for max_depth in [3, 5, 7, 9, 11]:
        for gamma in [0.5, 1, 1.5, 2, 5]:
            model = XGBClassifier(
                max_depth=max_depth,
                gamma=gamma,
                objective="binary:logistic",
                learning_rate=0.1,
                eval_metric="logloss",
                min_child_weight=1,
                seed=42,
                use_label_encoder=False,
            )
            # print("Fitting model with n: {} and g: {}".format(n, g))
            model.fit(X, y)
            pred_val = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, pred_val)
            if score > best_score:
                best_score = score
                best_model = model
                best_max_depth = max_depth
                best_gamma = gamma
    print("Best g:", best_gamma)
    print("Best n:", best_max_depth)
    return best_model
