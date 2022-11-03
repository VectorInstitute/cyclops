"""Train an mlp model on the data.

Train an mlp model on the data for a series of different values of c and returns the
best model using auroc metric.

"""
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier as MLP


def fit_mlp(X, y, X_val, y_val):
    """Train an mlp model on the data and return best model."""
    best_c = None
    best_score = 0
    best_model = None
    for l2_penalty in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
        model = MLP(
            hidden_layer_sizes=(256, 256, 256, 256),
            alpha=l2_penalty,
            early_stopping=True,
            learning_rate="adaptive",
            batch_size=128,
        )
        model.fit(X.values, y)
        pred_val = model.predict_proba(X_val.values)[:, 1]
        score = roc_auc_score(y_val, pred_val)
        print("Fitted model with C:", l2_penalty, "AUC:", score)
        if score > best_score:
            best_score = score
            best_model = model
            best_c = l2_penalty

    print("Best C:", best_c, "AUC:", best_score)
    return best_model
