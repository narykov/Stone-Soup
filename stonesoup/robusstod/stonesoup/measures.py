import numpy as np

from ...measures import Measure


class GaussianKullbackLeiblerDivergence(Measure):
    """Kullback-Leibler Divergence

    This measure returns the Kullback-Leibler divergence between a pair of
    :class:`~.GaussianState` objects.

    """
    def __call__(self, state1, state2):
        """Calculate Kullback-Leibler Divergence between a pair of multivariate normal distributions

        Parameters
        ----------
        state1 : :class:`~.GaussianState`
        state2 : :class:`~.GaussianState`

        Returns
        -------
        float
            Kullback-Leibler Divergence between two input :class:`~.GaussianState` objects

        """

        state_vector1 = getattr(state1, 'mean', state1.state_vector)
        state_vector2 = getattr(state2, 'mean', state2.state_vector)

        if self.mapping is not None:
            mu1 = state_vector1[self.mapping, :]
            mu2 = state_vector2[self.mapping2, :]

            # extract the mapped covariance data
            rows = np.array(self.mapping, dtype=np.intp)
            columns = np.array(self.mapping, dtype=np.intp)
            sigma1 = state1.covar[rows[:, np.newaxis], columns]
            sigma2 = state2.covar[rows[:, np.newaxis], columns]
        else:
            mu1 = state_vector1
            mu2 = state_vector2
            sigma1 = state1.covar
            sigma2 = state2.covar

        n_dim = len(mu1)
        inv2 = np.linalg.inv(sigma2)
        diff = mu2 - mu1
        frac = np.linalg.det(sigma2) / np.linalg.det(sigma1)

        kld = 0.5 * (np.trace(inv2 @ sigma1) - n_dim + diff.T @ inv2 @ diff + np.log(frac))

        return kld
