from shift_tester import *
from shift_reductor import *


class ShiftDetector:

    """ShiftDetector Class.

    Attributes
    ----------
    dr_technique: String
        name of dimensionality reduction technique to use
    sign_level: float
        significance level
    red_model:
        shift reductor model
    md_test: String
        name of two sample statisticaal test
    sample: int
        number of samples in test set
    datset: String
        name of dataset

    """

    def __init__(self, dr_technique, md_test, sign_level, red_model, sample, datset):
        self.dr_technique = dr_technique
        self.sign_level = sign_level
        self.red_model = red_model
        self.md_test = md_test
        self.sample = sample
        self.datset = datset
        self.sign_level = sign_level

    def classify_data(self, X_s_tr, y_s_tr, X_s_val, y_s_val, X_t, y_t, orig_dims):
        shift_reductor = ShiftReductor(X_s_tr, y_s_tr, "BBSDh", orig_dims, self.datset)
        shift_reductor_model = shift_reductor.fit_reductor()
        X_t_red = shift_reductor.reduce(shift_reductor_model, X_t)
        return X_t_red

    def detect_data_shift(self, X_s_tr, y_s_tr, X_s_val, y_s_val, X_t, y_t, orig_dims):

        val_acc = None
        te_acc = None

        # Train or load reduction model.
        shift_reductor = ShiftReductor(
            X_s_tr, y_s_tr, self.dr_technique, orig_dims, self.datset
        )
        shift_reductor_model = shift_reductor.fit_reductor()

        # Reduce test sets.
        X_s_red = shift_reductor.reduce(shift_reductor_model, X_s_val)
        X_t_red = shift_reductor.reduce(shift_reductor_model, X_t)

        # Compute classification accuracy on both sets for BBSDh malignancy detection.
        if self.dr_technique == "BBSDh":
            val_acc = np.sum(np.equal(X_s_red, y_s_val).astype(int)) / X_s_red.shape[0]
            te_acc = np.sum(np.equal(X_t_red, y_t).astype(int)) / X_t_red.shape[0]

        # Perform statistical test
        shift_tester = ShiftTester(sign_level=self.sign_level, mt=self.md_test)
        p_val, dist = shift_tester.test_shift(X_s_red[: self.sample], X_t_red)

        if self.dr_technique != "BBSDh":
            # Lower the significance level for all tests (Bonferroni) besides BBSDh, which needs no correction.
            adjust_sign_level = self.sign_level / X_s_red.shape[1]
        else:
            adjust_sign_level = self.sign_level

        return p_val, dist, shift_reductor.dr_amount, self.red_model, val_acc, te_acc
