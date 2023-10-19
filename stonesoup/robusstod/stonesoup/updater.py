import warnings
import numpy as np
from functools import lru_cache

from ...base import Property
from ...functions import gauss2sigma, unscented_transform
from ...types.prediction import Prediction, MeasurementPrediction
from ...updater.kalman import UnscentedKalmanUpdater
from ...measures import Measure

# ROBUSSTOD CLASSES
from .models.measurement import GeneralLinearGaussian
from .measures import GaussianKullbackLeiblerDivergence
from .functions import slr_definition


class IPLFKalmanUpdater(UnscentedKalmanUpdater):
    """
    Description goes here.
    """

    tolerance: float = Property(
        default=1e-1,
        doc="The value of the difference in the measure used as a stopping criterion.")
    measure: Measure = Property(
        default=GaussianKullbackLeiblerDivergence(),
        doc="The measure to use to test the iteration stopping criterion. Defaults to the "
            "GaussianKullbackLeiblerDivergence between current and prior posterior state estimate.")
    max_iterations: int = Property(
        default=5,
        doc="Number of iterations before while loop is exited and a non-convergence warning is "
            "returned")

    def update(self, hypothesis, **kwargs):
        r"""The IPLF update method.

        Parameters
        ----------
        hypothesis : :class:`~.SingleHypothesis`
            the prediction-measurement association hypothesis. This hypothesis
            may carry a predicted measurement, or a predicted state. In the
            latter case a predicted measurement will be calculated.
        **kwargs : various
            These are passed to the measurement model function

        Returns
        -------
        : :class:`~.GaussianStateUpdate`
            The posterior state Gaussian with mean :math:`\mathbf{x}_{k|k}` and
            covariance :math:`P_{k|k}`

        """

        # Record the starting point (not a posterior here, rather a variable that stores an entry for KLD computation)
        prev_post_state = hypothesis.prediction  # Prior is only on the first step, later updated

        # Get the measurement model out of the measurement if it's there.
        # If not, use the one native to the updater (which might still be none).
        measurement_model = self._check_measurement_model(hypothesis.measurement.measurement_model)

        # The first iteration is just the application of the UKF update.
        hypothesis.measurement_prediction = super().predict_measurement(
            predicted_state=hypothesis.prediction,
            measurement_model=measurement_model
        )  # UKF measurement prediction that relies on Unscented Transform and is required in the update
        post_state = super().update(hypothesis, **kwargs)  # <- just this line alone isn't enough as it implements KF
        # Now update the measurement prediction mean and loop
        iterations = 1
        while self.measure(prev_post_state, post_state) > self.tolerance:

            # SLR is wrt to tne approximated posterior in post_state, not the original prior in hypothesis.prediction
            measurement_prediction = UnscentedKalmanUpdater().predict_measurement(
                predicted_state=post_state,
                measurement_model=measurement_model
            )
            h_matrix, b_vector, omega_cov_matrix = slr_definition(post_state, measurement_prediction)

            r_cov_matrix = measurement_model.noise_covar
            measurement_model_linearized = GeneralLinearGaussian(
                ndim_state=measurement_model.ndim_state,
                mapping=measurement_model.mapping,
                meas_matrix=h_matrix,
                bias_value=b_vector,
                noise_covar=r_cov_matrix + omega_cov_matrix)

            hypothesis.measurement_prediction = super().predict_measurement(
                predicted_state=hypothesis.prediction,
                measurement_model=measurement_model_linearized)
            hypothesis.measurement.measurement_model = measurement_model_linearized

            prev_post_state = post_state
            # update is computed using the original prior in hypothesis.prediction
            post_state = super().update(hypothesis, **kwargs)  # classic Kalman update
            post_state.hypothesis.measurement.measurement_model = measurement_model

            post_state.covar = (post_state.covar + post_state.covar.T) / 2
            try:
                np.linalg.cholesky(post_state.covar)
            except:
                print("Matrix is not positive definite.")
                breakpoint()

            # increment counter
            iterations += 1

        print("IPLF update took {} iterations and the KLD value of {}.".format(
            iterations, *self.measure(prev_post_state, post_state)
        ))

        return post_state
