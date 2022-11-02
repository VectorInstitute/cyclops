from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import roc_auc_score


def fit_lr(X, Y, Xv, Yv):
    best_c = None
    best_score = 0
    best_model = None
    for c in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
        m = LR(C=c)
        print("Fitting model with C:", c)
        m.fit(X, Y)
        Pv = m.predict_proba(Xv)[:, 1]
        score = roc_auc_score(Yv, Pv)
        if score > best_score:
            best_score = score
            best_model = m
            best_c = c

    print("Best C:", best_c)
    return best_model
