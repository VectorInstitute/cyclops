#import all relevant libraries
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

params = {
    "objective":'binary:logistic',
    "nthread":4,
    "seed" : 1   
    "loss":["deviance"],
    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    "min_samples_split": np.linspace(0.1, 0.5, 12),
    "min_samples_leaf": np.linspace(0.1, 0.5, 12),
    "max_depth":[3,5,8],
    "max_features":["log2","sqrt"],
    "criterion": ["friedman_mse",  "mae"],
    "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
    "n_estimators":[10]
    }

#xgb1 = XGBClassifier()
#clf = GridSearchCV(xgbl, parameters,scoring=scoring,refit=False,cv=2, n_jobs=-1)
#clf.fit(trainX, trainY)

def fit(X, Y, Xv, Yv):
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