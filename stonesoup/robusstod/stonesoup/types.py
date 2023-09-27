from ...base import Property
from ...types.array import CovarianceMatrix
from ...types.prediction import GaussianStatePrediction


class AugmentedGaussianStatePrediction(GaussianStatePrediction):
    """ Prediction type

    This is the base prediction class. """
    cross_covar: CovarianceMatrix = Property(
        default=None, doc='Cross-covariance for the SLR algorithm')
