import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.distributions.independent import Independent
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal


class SingleGaussianizeStep:
    def __init__(self, n_bins=10, alpha=10, lam_variance=0):
        self.n_bins = n_bins
        self.alpha = alpha
        self.lam_variance = lam_variance

    def fit(self, x):
        self.fit_transform(x)
        return self

    def fit_transform(self, x):
        all_latent = []
        # 1. PCA transform
        pca = PCA(random_state=0)
        pca.fit(x.detach().numpy())
        # assert np.isclose(np.abs(np.linalg.det(pca.components_)), 1), 'Should be close to one'
        Q_pca = torch.from_numpy(pca.components_)
        x = torch.mm(x, Q_pca.T)

        # 2. Independent normal cdf transform
        scale, loc = torch.std_mean(x, dim=0)
        ind_normal = Normal(loc, torch.sqrt(scale * scale + self.lam_variance))
        x = ind_normal.cdf(x)
        x = torch.clamp(x, 1e-10, 1 - 1e-10)

        # 3. Independent histogram transform
        if True:
            histograms = [
                TorchUnitHistogram(n_bins=self.n_bins, alpha=self.alpha).fit(x_col)
                for x_col in x.detach().T
            ]
            x = torch.cat(
                tuple(
                    hist.cdf(x_col).reshape(-1, 1)
                    for x_col, hist in zip(x.T, histograms)
                ),
                dim=1,
            )
            # all_latent.append(x.detach().numpy())
            self.histograms_ = histograms

        # 4. Independent inverse standard normal transform
        if True:
            standard_normal = Normal(
                loc=torch.zeros_like(loc), scale=torch.ones_like(scale)
            )
            x = standard_normal.icdf(x)
            self.standard_normal_ = standard_normal

        self.Q_pca_ = Q_pca
        self.ind_normal_ = ind_normal
        self._latent = x  # Just for debugging purposes
        return x

    def log_prob(self, x, return_latent=False):
        # 1. PCA
        log_prob = torch.zeros_like(x[:, 0])  # Orthogonal transform has logdet of 0
        x = torch.mm(x, self.Q_pca_.T)

        # 2. Ind normal
        log_prob += torch.sum(self.ind_normal_.log_prob(x), dim=1)  # Independent so sum
        x = self.ind_normal_.cdf(x)  # Transform
        x = torch.clamp(x, 1e-10, 1 - 1e-10)

        # 3. Histogram
        if True:
            log_prob += torch.sum(
                torch.cat(
                    tuple(
                        hist.log_prob(x_col).reshape(-1, 1)
                        for x_col, hist in zip(x.T, self.histograms_)
                    ),
                    dim=1,
                ),
                dim=1,
            )
            x = torch.cat(
                tuple(
                    hist.cdf(x_col).reshape(-1, 1)
                    for x_col, hist in zip(x.T, self.histograms_)
                ),
                dim=1,
            )

        # 4. Inverse standard normal
        if True:
            x = self.standard_normal_.icdf(
                x
            )  # For log prob of inverse cdf must do inverse cdf first
            log_prob -= torch.sum(
                self.standard_normal_.log_prob(x), dim=1
            )  # Independent so sum

        if return_latent:
            return log_prob, x
        else:
            return log_prob

    def inverse(self, x):
        # 4. Inverse standard normal
        if True:
            x = self.standard_normal_.cdf(
                x
            )  # For log prob of inverse cdf must do inverse cdf first
            x = torch.clamp(x, 1e-10, 1 - 1e-10)

        # 3. Histogram
        if True:
            x = torch.cat(
                tuple(
                    hist.icdf(x_col).reshape(-1, 1)
                    for x_col, hist in zip(x.T, self.histograms_)
                ),
                dim=1,
            )

        # 2. Ind normal
        x = self.ind_normal_.icdf(x)  # Transform

        # 1. PCA
        x = torch.mm(x, self.Q_pca_)
        return x


class TorchUnitHistogram:
    """Assumes all data is unit norm."""

    def __init__(self, n_bins, alpha):
        self.n_bins = n_bins
        self.alpha = alpha

    def fit(self, x):
        x = x.numpy()
        # Do numpy stuff
        hist, bin_edges = np.histogram(x, bins=self.n_bins, range=[0, 1])
        hist = np.array(hist, dtype=float)  # Make float so we can add non-integer alpha
        hist += self.alpha  # Smooth histogram by alpha so no areas have 0 probability
        cum_hist = np.cumsum(hist)
        cum_hist = cum_hist / cum_hist[-1]  # Normalize cumulative histogram

        # Make torch tensors
        bin_edges = torch.from_numpy(bin_edges)
        # Makes the same length as bin_edges
        cdf_on_edges = torch.from_numpy(np.concatenate(([0], cum_hist)))

        # Compute scale and shift for every bin
        # a = (y2-y1)/(x2-y1)
        bin_scale = (cdf_on_edges[1:] - cdf_on_edges[:-1]) / (
            bin_edges[1:] - bin_edges[:-1]
        )
        # b = -a*x2 + y2
        bin_shift = -bin_scale * bin_edges[1:] + cdf_on_edges[1:]

        # Normalize bins by bin_edges
        self.bin_edges_ = bin_edges
        self.cdf_on_edges_ = cdf_on_edges
        self.bin_scale_ = bin_scale
        self.bin_shift_ = bin_shift
        return self

    def cdf(self, x):
        assert torch.all(
            torch.logical_and(x >= 0, x <= 1)
        ), "All inputs should be between 0 and 1"
        bin_idx = self._get_bin_idx(x)
        # Linear interpolate within the selected bin
        return self.bin_scale_[bin_idx] * x + self.bin_shift_[bin_idx]

    def icdf(self, x):
        assert torch.all(
            torch.logical_and(x >= 0, x <= 1)
        ), "All inputs should be between 0 and 1"
        bin_idx = self._get_inverse_bin_idx(x)
        # Linear interpolate within the selected bin
        return (x - self.bin_shift_[bin_idx]) / self.bin_scale_[bin_idx]

    def log_prob(self, x):
        # Find closest bin
        bin_idx = self._get_bin_idx(x)
        return torch.log(self.bin_scale_[bin_idx])

    def _get_bin_idx(self, x):
        return (
            torch.floor(x.detach() * self.n_bins)
            .clamp(0, self.n_bins - 1)
            .type(torch.long)
        )

    def _get_inverse_bin_idx(self, x):
        bin_idx = -torch.ones_like(x, dtype=torch.long)
        for ii, (left_edge, right_edge) in enumerate(
            zip(self.cdf_on_edges_[:-1], self.cdf_on_edges_[1:])
        ):
            if ii == self.n_bins - 1:
                # Include right edge
                bin_idx[torch.logical_and(x >= left_edge, x <= right_edge)] = ii
            else:
                bin_idx[torch.logical_and(x >= left_edge, x < right_edge)] = ii
        assert torch.all(
            torch.logical_and(bin_idx >= 0, bin_idx < self.n_bins)
        ), "Bin indices incorrect"
        return bin_idx
