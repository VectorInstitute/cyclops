from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.gaussian_process import GaussianProcessClassifier as GPC
from sklearn.gaussian_process.kernels import (
    RBF,
    DotProduct,
    Matern,
    RationalQuadratic,
    WhiteKernel,
)
from sklearn.metrics import (
    auc,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)


def fit_gp(X, Y, Xv, Yv):
    best_k = None
    best_score = 0
    best_model = None
    for K in [
        1 * RBF(),
        1 * DotProduct(),
        1 * Matern(),
        1 * RationalQuadratic(),
        1 * WhiteKernel(),
    ]:
        m = GPC(kernel=K, n_jobs=-1)
        m.fit(X, Y)
        Pv = m.predict_proba(Xv)[:, 1]
        score = roc_auc_score(Yv, Pv)
        print("Fitted GPC with K:", K, "AUC:", score)
        if score > best_score:
            best_score = score
            best_model = m
            best_k = K

    print("Best K:", best_k)
    return best_model


def fit_rf(X, Y, Xv, Yv):
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


def fit_xgb(X, Y, Xv, Yv):
    best_n = None
    best_g = None
    best_eta = None
    best_score = 0
    best_model = None
    for n in [3, 5, 7, 9, 11]:
        for g in [0.5, 1, 1.5, 2, 5]:
            for eta in [0.3, 0.2, 0.1, 0.05, 0.01, 0.005]:
                m = XGBClassifier(
                    max_depth=n,
                    gamma=g,
                    eta=eta,
                    objective="binary:logistic",
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
                    best_eta = eta

    print("Best eta:", best_eta, "g:", best_g, "n:", best_n)
    return best_model


def fit_mlp(X, Y, Xv, Yv):
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


def run_model(model_name, X, Y, Xv, Yv):
    if model_name == "mlp":
        best_model = fit_mlp(X, Y, Xv, Yv)
    elif model_name == "lr":
        best_model = fit_lr(X, Y, Xv, Yv)
    elif model_name == "rf":
        best_model = fit_rf(X, Y, Xv, Yv)
    elif model_name == "xgb":
        best_model = fit_xgb(X, Y, Xv, Yv)
    return best_model


def compute_threshold_metric(y_true, pred_prob, threshold, **kwargs):
    """
    Threshold metrics for binary prediction tasks
    """
    y_pred = 1 * (pred_prob > threshold)
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()

    metric_dict = {}

    # Sensitivity, hit rate, recall, or true positive rate
    metric_dict["sensitivity"] = TP / (TP + FN)
    # Specificity or true negative rate
    metric_dict["specificity"] = TN / (TN + FP)
    # Precision or positive predictive value
    metric_dict["ppv"] = TP / (TP + FP)
    # Negative predictive value
    metric_dict["npv"] = TN / (TN + FN)
    # Fall out or false positive rate
    metric_dict["fpr"] = FP / (FP + TN)
    # False negative rate
    metric_dict["fnr"] = FN / (TP + FN)
    # False discovery rate
    metric_dict["fdr"] = FP / (TP + FP)

    return metric_dict
