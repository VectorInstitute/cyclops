from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import roc_auc_score

def fit(X, Y, Xv, Yv):
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
