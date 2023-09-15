import warnings
import numpy as np
from functools import lru_cache

from ...base import Property
from ...functions import gauss2sigma, unscented_transform
from ...types.prediction import Prediction, MeasurementPrediction
from ...updater.kalman import UnscentedKalmanUpdater as UnscentedKalmanUpdaterOriginal
from ...measures import Measure

# ROBUSSTOD CLASSES
from .models.measurement import GeneralLinearGaussian
from .measures import GaussianKullbackLeiblerDivergence


class UnscentedKalmanUpdater(UnscentedKalmanUpdaterOriginal):

    @lru_cache()
    def predict_measurement(self, predicted_state, measurement_model=None, **kwargs):
        """Unscented Kalman Filter measurement prediction step. Uses the
        unscented transform to estimate a Gauss-distributed predicted
        measurement.

        Parameters
        ----------
        predicted_state : :class:`~.GaussianStatePrediction`
            A predicted state
        measurement_model : :class:`~.MeasurementModel`, optional
            The measurement model used to generate the measurement prediction.
            This should be used in cases where the measurement model is
            dependent on the received measurement (the default is `None`, in
            which case the updater will use the measurement model specified on
            initialisation)

        Returns
        -------
        : :class:`~.GaussianMeasurementPrediction`
            The measurement prediction

        """

        measurement_model = self._check_measurement_model(measurement_model)

        sigma_points, mean_weights, covar_weights = \
            gauss2sigma(predicted_state,
                        self.alpha, self.beta, self.kappa)

        meas_pred_mean, meas_pred_covar, cross_covar, _, _, _ = \
            unscented_transform(sigma_points, mean_weights, covar_weights,
                                measurement_model.function,
                                covar_noise=measurement_model.covar())

        return MeasurementPrediction.from_state(
            predicted_state, meas_pred_mean, meas_pred_covar, cross_covar=cross_covar)


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

    def _slr_calculations(self, prediction, measurement_model, **kwargs):
        """
        Notation follows https://github.com/Agarciafernandez/IPLF/blob/main/IPLF_maneuvering.m
        """

        mean_pos = prediction.state_vector
        cov_pos = prediction.covar

        measurement_prediction = self.predict_measurement(
            predicted_state=prediction,
            measurement_model=measurement_model, **kwargs)  # using sigma points in UKF
        z_pred = measurement_prediction.state_vector
        var_pred = measurement_prediction.covar  # = Phi
        var_xz = measurement_prediction.cross_covar  # = Psi

        # Statistical linear regression parameters
        A_l = var_xz.T @ np.linalg.inv(cov_pos)
        b_l = z_pred - A_l @ mean_pos
        Omega_l = var_pred - A_l @ cov_pos @ A_l.T

        return {'A_l': A_l, 'b_l': b_l, 'Omega_l': Omega_l}

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

        # Record the starting point
        prev_state = hypothesis.prediction

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
        while self.measure(prev_state, post_state) > self.tolerance:

            if iterations >= self.max_iterations:
                warnings.warn("IPLF update reached maximum number of iterations.")
                break

            # hypothesis.prediction = Prediction.from_state(
            #     state=post_state,
            #     state_vector=post_state.state_vector,
            #     covar=post_state.covar,
            #     timestamp=post_state.timestamp
            # )

            "SLR is computed wrt to posterior, not the original prior"
            # slr = self._slr_calculations(hypothesis.prediction, measurement_model)
            slr = self._slr_calculations(post_state, measurement_model)

            measurement_model_linearized = GeneralLinearGaussian(
                ndim_state=measurement_model.ndim_state,
                mapping=measurement_model.mapping,
                meas_matrix=slr['A_l'],
                bias_value=slr['b_l'],
                noise_covar=measurement_model.noise_covar + slr['Omega_l'])

            hypothesis.measurement_prediction = super().predict_measurement(
                predicted_state=hypothesis.prediction,
                measurement_model=measurement_model_linearized)

            hypothesis.measurement.measurement_model = measurement_model_linearized

            "Update is computed wrt the original prior"
            post_state = super().update(hypothesis, **kwargs)
            post_state.hypothesis.measurement.measurement_model = measurement_model

            post_state.covar = (post_state.covar + post_state.covar.T) / 2
            try:
                np.linalg.cholesky(post_state.covar)
            except:
                print("Matrix is not positive definite.")
                breakpoint()

            # increment counter
            iterations += 1

        # post_state.hypothesis.prediction = prev_state
        print("IPLF update took {} iterations and the KLD value of {}.".format(iterations, *self.measure(prev_state, post_state)))

        return post_state
