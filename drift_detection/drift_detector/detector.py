from .tester import ShiftTester

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

    def __init__(self, dr_technique, md_test, sign_level, shift_reductor, sample, datset, features, model_path):
        self.dr_technique = dr_technique
        self.sign_level = sign_level
        self.shift_reductor = shift_reductor
        self.md_test = md_test
        self.sample = sample
        self.dataset = datset
        self.sign_level = sign_level
        self.features = features
        self.model_path = model_path

    def classify_data(self, X_s_tr, y_s_tr, X_s_val, y_s_val, X_t, y_t, orig_dims):
        
        shift_reductor_model = self.shift_reductor.fit_reductor()
        X_t_red = self.shift_reductor.reduce(shift_reductor_model, X_t)
        return X_t_red

    def detect_data_shift(self, X_s_tr, y_s_tr, X_s_val, y_s_val, X_t, y_t, orig_dims, context_type):

        val_acc = None
        te_acc = None

        # Train or load reduction model.
        shift_reductor_model = self.shift_reductor.fit_reductor()

        # Reduce test sets.
        X_s_red = self.shift_reductor.reduce(shift_reductor_model, X_s_val)
        X_t_red = self.shift_reductor.reduce(shift_reductor_model, X_t)

        # Compute classification accuracy on both sets for BBSDh malignancy detection.
        if self.dr_technique == "BBSDh":
            val_acc = np.sum(np.equal(X_s_red, y_s_val).astype(int)) / X_s_red.shape[0]
            te_acc = np.sum(np.equal(X_t_red, y_t).astype(int)) / X_t_red.shape[0]

        # Perform statistical test
        shift_tester = ShiftTester(sign_level=self.sign_level, 
                                   mt=self.md_test, 
                                   model_path=self.model_path, 
                                   features=self.features, 
                                   dataset=self.dataset)
        
        p_val, dist = shift_tester.test_shift(X_s_red[: self.sample], X_t_red, context_type)

        if self.dr_technique != "BBSDh":
            # Lower the significance level for all tests (Bonferroni) besides BBSDh, which needs no correction.
            adjust_sign_level = self.sign_level / X_s_red.shape[1]
        else:
            adjust_sign_level = self.sign_level

        return p_val, dist, val_acc, te_acc
