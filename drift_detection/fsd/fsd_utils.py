from textwrap import wrap as textwrap

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.optimize
import seaborn as sn
from scipy import stats
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from sklearn.utils import check_random_state


def marginal_attack(X, attack_set, random_state=None):
    """Perform marginal attack jointly on the features in attack_set."""
    rng = check_random_state(random_state)
    attack_set = np.array(attack_set)
    # just in case the original input is going to be used in later testing
    X = X.copy()
    # shuffle happens inplace
    X[:, attack_set] = rng.permutation(X[:, attack_set])
    return X


def create_graphical_model(
    sqrtn=2,
    kind="complete",
    alpha="auto",
    target_idx="auto",
    target_mutual_information=0.5,
    cond_thresh=100,
    random_seed=0,
    nx_kwargs=None,
):
    """Creates graphical dependence models based on a target MI, random_seed, and
    target_idx."""
    if nx_kwargs is None:
        nx_kwargs = {}
    n = sqrtn**2

    # Determine target index
    if target_idx == "auto":
        target_idx = int(np.floor(n / 2))

    # Create dependency graph
    def create_graph():
        if kind == "complete":
            G = nx.complete_graph(n, **nx_kwargs)
        elif kind == "grid":
            G = nx.grid_2d_graph(sqrtn, sqrtn, **nx_kwargs)
        elif kind == "cycle":
            G = nx.cycle_graph(n)
        elif kind == "random":
            kwargs = dict(p=0.1, seed=random_seed)
            kwargs.update(nx_kwargs)
            G = nx.erdos_renyi_graph(n, **kwargs)
        else:
            raise RuntimeError(f'Unknown graph kind="{kind}"')
        return G

    G = create_graph()
    edge_mat = nx.to_numpy_array(G)

    # Utility function for determining if a matrix is positive definite
    def is_positive_definite(A):
        try:
            # See if positive definite
            # (cholesky only works if PD)
            np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            return False
        else:
            return True

    def mutual_information(inv_cov):
        compromised_sel = np.zeros(n, dtype=bool)  # Create false array
        compromised_sel[target_idx] = True
        cov = np.linalg.inv(
            inv_cov
        )  # XXX could be faster by only computing inverse once
        cov_c = cov[np.ix_(compromised_sel, compromised_sel)]
        cov_notc = cov[np.ix_(~compromised_sel, ~compromised_sel)]
        # XXX could use cholesky for faster det computation
        return 0.5 * (
            np.linalg.slogdet(cov_c)[1]
            + np.linalg.slogdet(cov_notc)[1]
            - np.linalg.slogdet(cov)[1]
        )

    # Determine edge weights
    if alpha == "auto":
        # Automatically find a valid edge weight
        # that has a certain condition number
        def func_to_minimize(a):
            inv_cov = np.eye(n) + a * edge_mat
            if not is_positive_definite(inv_cov):
                return np.inf
            else:
                return mutual_information(inv_cov) - target_mutual_information

        a = scipy.optimize.brentq(func_to_minimize, 0, 1, maxiter=100)
    else:
        a = alpha

    # Form inverse covariance matrix
    inv_cov = np.eye(n) + a * edge_mat
    assert is_positive_definite(inv_cov), "Final matrix should be PD"
    # print(f'Condition number={np.linalg.cond(inv_cov)},
    # Mutual information={mutual_information(inv_cov)}')

    cov = np.linalg.inv(inv_cov)

    return dict(
        cov=cov,
        inv_cov=inv_cov,
        graph=G,
        kind=kind,
        target_idx=target_idx,
        mutual_information_of_attack=mutual_information(cov),
        condition_number=np.linalg.cond(cov),
    )


def sim_copula_data(p_size, q_size, mean, cov, a, b, rng=None):
    """Takes in a target Gaussian mean and covariance, then transforms to a copula."""
    if rng is None:
        rng = np.random.RandomState(np.random.randint(10000))
    X = rng.multivariate_normal(mean=mean, cov=cov, size=p_size + q_size)
    Z = (X - X.mean(axis=0)) / np.std(X, axis=0)  # z = (x - mu_x) / \sigma_x
    U = stats.norm.cdf(Z)  # fits to copula, j-dist r.v. with uniform marginals
    B = stats.beta.ppf(U, a=a, b=b)  # inverse CDF (percent point function)

    return B[:p_size], B[p_size:]  # returns samples p and q


def get_detection_metrics(true_labels, predicted_labels):
    """Calculates tp, fp, fn, and tn from a confusion matrix, then get precision,
    recall, and acc."""
    tn, fp, fn, tp = sklearn_confusion_matrix(
        true_labels, predicted_labels, labels=[0, 1]
    ).flatten()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    acc = (tp + tn) / (true_labels.shape[0])
    confusion_matrix = np.array([[tn, fp], [fn, tp]])
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "precision": precision,
        "recall": recall,
        "acc": acc,
        "confusion_matrix": confusion_matrix,
    }


def get_localization_metrics(true_labels_tensor, predicted_labels_tensor, n_dim):
    """Creates a confusion matrix for each feature as an array with shape (n_features,
    2, 2), then calculates the micro-precision and micro-recall and returns as a
    dict."""
    confusion_tensor = np.zeros(shape=(n_dim, 2, 2))
    for feature_idx in range(n_dim):
        confusion_tensor[feature_idx] = sklearn_confusion_matrix(
            true_labels_tensor[feature_idx],
            predicted_labels_tensor[feature_idx],
            labels=[0, 1],
        )
    # here we will sum along the feature axis of the confusion_tensor
    # to get the micro-precision and recall
    tn, fp, fn, tp = confusion_tensor.sum(axis=0).flatten()
    micro_precision = tp / (tp + fp)
    micro_recall = tp / (tp + fn)
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "micro-precision": micro_precision,
        "micro-recall": micro_recall,
        "confusion_tensor": confusion_tensor,
    }


def plot_confusion_matrix(
    confusion_matrix, plot=False, title=None, axis=None, filename=None
):
    """Plots as confusion matrix using seaborn heatmap."""
    if axis is None:
        fig, axis = plt.subplots()
    names = ["TN", "FP", "FN", "TP"]
    counts = confusion_matrix.flatten()
    labels = [f"{n}\n{c}" for n, c in zip(names, counts)]
    labels = np.array(labels).reshape(2, 2)
    sn.heatmap(
        confusion_matrix.astype(np.int),
        annot=labels,
        fmt="",
        xticklabels=False,
        yticklabels=False,
        linewidth=0.5,
        cbar=False,
        ax=axis,
        cmap="Blues",
    )
    if title:
        axis.set_title(wrap(title))
    if filename:
        plt.savefig(filename)
    if plot:
        plt.show()
    return None


def get_confusion_tensor(true_labels_tensor, predicted_labels_tensor, n_dim):
    """Creates a confusion matrix for each feature and returns it as an array with shape
    (n_features, 2, 2)"""
    confusion_tensor = np.zeros(shape=(n_dim, 2, 2))
    for feature_idx in range(n_dim):
        confusion_tensor[feature_idx] = sklearn_confusion_matrix(
            true_labels_tensor, predicted_labels_tensor, labels=[0, 1]
        )
    return confusion_tensor


def wrap(string):
    """Wraps strings of legn."""
    return "\n".join(textwrap(string, 60))
