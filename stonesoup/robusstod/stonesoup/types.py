from ...base import Property
from ...types.state import GaussianState
from ...types.array import CovarianceMatrix
from ...types.prediction import Prediction


class AugmentedGaussianState(GaussianState):
    """ This is a new GaussianState class that can also store information on cross-covariance
    between the two uncertain kinematic states. We need it report augmented predictions which is otherwise not
    possible."""
    cross_covar: CovarianceMatrix = Property(
        default=None, doc='Cross-covariance for the SLR algorithm')


class AugmentedGaussianStatePrediction(Prediction, AugmentedGaussianState):
    """ Prediction class for AugmentedGaussianState. The existence of this class harmonises how state predictions
    are reported, to be in line with measurement predictions, which also carry information on cross-covariance. """
