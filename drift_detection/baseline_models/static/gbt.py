from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,
)


def fit_gbt(X, Y, Xv, Yv):
    best_n = None
    best_g = None
    best_score = 0
    best_model = None
    for n in [3, 5, 7, 9, 11]:
        for g in [0.5, 1, 1.5, 2, 5]:
            m = XGBClassifier(
                max_depth=n,
                gamma=g,
                objective="binary:logistic",
                learning_rate=0.1,
                eval_metric="logloss",
                min_child_weight=1,
                seed=42,
                use_label_encoder=False,
            )
            # print("Fitting model with n: {} and g: {}".format(n, g))
            m.fit(X, Y)
            Pv = m.predict_proba(Xv)[:, 1]
            score = roc_auc_score(Yv, Pv)
            if score > best_score:
                best_score = score
                best_model = m
                best_n = n
                best_g = g
    print("Best g:", best_g)
    print("Best n:", best_n)
    return best_model
