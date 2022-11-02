"""Train an mlp model on the data.

Train an mlp model on the data for a series of different values of c and returns the
best model using auroc metric.

"""
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier as MLP


def fit_mlp(X, Y, Xv, Yv):
    """Train an mlp model on the data and return best model."""
    best_c = None
    best_score = 0
    best_model = None
    for c in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
        m = MLP(
            hidden_layer_sizes=(256, 256, 256, 256),
            alpha=c,
            early_stopping=True,
            learning_rate="adaptive",
            batch_size=128,
        )
        m.fit(X.values, Y)
        Pv = m.predict_proba(Xv.values)[:, 1]
        score = roc_auc_score(Yv, Pv)
        print("Fitted model with C:", c, "AUC:", score)
        if score > best_score:
            best_score = score
            best_model = m
            best_c = c

    print("Best C:", best_c, "AUC:", best_score)
    return best_model
