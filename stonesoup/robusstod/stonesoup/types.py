from ...types.base import Type
from ...base import Property
from ...models.transition.base import TransitionModel
from ...types.state import CreatableFromState
from ...types.array import CovarianceMatrix


class Prediction(Type, CreatableFromState):
    """ Prediction type

    This is the base prediction class. """
    transition_model: TransitionModel = Property(
        default=None, doc='The transition model used to make the prediction')
    cross_covar: CovarianceMatrix = Property(
        default=None, doc='Cross-covariance for the SLR algorithm')