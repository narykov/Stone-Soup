import warnings
import numpy as np
from functools import lru_cache

from ...base import Property
from ...functions import gauss2sigma, unscented_transform
from ...types.prediction import Prediction, MeasurementPrediction
from ...updater.kalman import KalmanUpdater #UnscentedKalmanUpdater
from ...measures import Measure
from ...models.measurement import MeasurementModel
from ...types.array import CovarianceMatrix, StateVector
import matplotlib.pyplot as plt

# ROBUSSTOD CLASSES
from .models.measurement import GeneralLinearGaussian
from .measures import GaussianKullbackLeiblerDivergence
from .functions import slr_definition, unscented_transform
from stonesoup.types.angle import Bearing, Elevation

# Debugging
from stonesoup.types.state import State
import matplotlib.pyplot as plt
import datetime as dt


class UnscentedKalmanUpdater(KalmanUpdater):
    """The Unscented Kalman Filter version of the Kalman Updater. Inherits most
    of the functionality from :class:`~.KalmanUpdater`.

    In this case the :meth:`predict_measurement` function uses the
    :func:`unscented_transform` function to estimate a (Gaussian) predicted
    measurement. This is then updated via the standard Kalman update equations.

    """
    # Can be non-linear and non-differentiable
    measurement_model: MeasurementModel = Property(
        default=None,
        doc="The measurement model to be used. This need not be defined if a "
            "measurement model is provided in the measurement. If no model "
            "specified on construction, or in the measurement, then error "
            "will be thrown.")
    alpha: float = Property(
        default=0.5,
        doc="Primary sigma point spread scaling parameter. Default is 0.5.")
    beta: float = Property(
        default=2,
        doc="Used to incorporate prior knowledge of the distribution. If the "
            "true distribution is Gaussian, the value of 2 is optimal. "
            "Default is 2")
    kappa: float = Property(
        default=0,
        doc="Secondary spread scaling parameter. Default is calculated as "
            "3-Ns")

    def _posterior_mean(self, predicted_state, kalman_gain, measurement, measurement_prediction):
        r"""Compute the posterior mean, :math:`\mathbf{x}_{k|k} = \mathbf{x}_{k|k-1} + K_k
        \mathbf{y}_k`, where the innovation :math:`\mathbf{y}_k = \mathbf{z}_k -
        h(\mathbf{x}_{k|k-1}).

        Parameters
        ----------
        predicted_state : :class:`State`, :class:`Prediction`
            The predicted state
        kalman_gain : numpy.ndarray
            Kalman gain
        measurement : :class:`Detection`
            The measurement
        measurement_prediction : :class:`MeasurementPrediction`
            Predicted measurement

        Returns
        -------
        : :class:`StateVector`
            The posterior mean estimate
        """
        post_mean = predicted_state.state_vector + \
            kalman_gain @ (measurement.state_vector - measurement_prediction.state_vector)
        if len(measurement.state_vector) == 4:
            from ...robusstod.crouse import h_wrap_sphere_single
            from stonesoup.types.angle import Bearing, Elevation
            meas_diff = h_wrap_sphere_single(measurement.state_vector - measurement_prediction.state_vector)
            post_mean = predicted_state.state_vector + kalman_gain @ meas_diff


        return post_mean.view(StateVector)

    @lru_cache()
    def predict_measurement(self, predicted_state, measurement_model=None):
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

        meas_pred_mean, meas_pred_covar, cross_covar, sigma_points_t, _, _ = \
            unscented_transform(sigma_points, mean_weights, covar_weights,
                                measurement_model.function,
                                covar_noise=measurement_model.covar())



        import matplotlib.pyplot as plt

        # sv_sens = measurement_model.translation_offset
        # plt.plot(*sv_sens, 'b.')
        #
        # fig, ax = plt.subplots()
        # for sp_t in sigma_points_t:
        #     # print(sp_t)
        #     ax.plot(sp_t[0], sp_t[1], '.')
        # ax.plot(meas_pred_mean[0], meas_pred_mean[1], 'x')
        # ax.set_xlabel('Elevation')
        # ax.set_ylabel('Bearing')
        # plt.show()

        import matplotlib.pyplot as plt
        # for sigma_point in sigma_points:
        #     mapping = measurement_model.mapping
        #     sv = sigma_point.state_vector
        #     plt.plot(*sv[mapping, :], marker='.', color='k', markersize=2)
        #
        # from stonesoup.types.detection import Detection
        # for sigma_point_t in sigma_points_t:
        #     sv = measurement_model.inverse_function(Detection(state_vector=sigma_point_t))
        #     plt.plot(*sv[mapping, :], marker='+', color='r', markersize=2)
        #
        # sv_mean = measurement_model.inverse_function(Detection(state_vector=meas_pred_mean))
        # plt.plot(*sv_mean[mapping, :], marker='x', color='g', markersize=20)
        #
        # before_crouse = StateVector([[0.4041282052661798],
        #                              [-1.0254644980825756],
        #                              [1085542.8695008296],
        #                              [2359.200089030421]])
        # sv_before_crouse = measurement_model.inverse_function(Detection(state_vector=before_crouse))
        # plt.plot(*sv_before_crouse[mapping, :], marker='x', color='r', markersize=20)

        return MeasurementPrediction.from_state(
            predicted_state, meas_pred_mean, meas_pred_covar, cross_covar=cross_covar)


class IPLFKalmanUpdater(UnscentedKalmanUpdater):
    """
    The update step of the IPLF algorithm.
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

        if 'time_of_interest' in kwargs:
            if hypothesis.prediction.timestamp >= kwargs['time_of_interest']:
                breakpoint()

        # The first iteration is just the application of the UKF update.
        hypothesis.measurement_prediction = super().predict_measurement(
            predicted_state=hypothesis.prediction,
            measurement_model=measurement_model
        )  # UKF measurement prediction that relies on Unscented Transform and is required in the update

        post_state = super().update(hypothesis, **kwargs)  # <- just this line alone isn't enough as it implements KF
        import matplotlib.pyplot as plt
        mapping = measurement_model.mapping
        # plt.gca().plot(hypothesis.measurement.state_vector[0], hypothesis.measurement.state_vector[1], 'b+')
        corresp_meas = measurement_model.function(post_state)

        if 'time_of_interest' in kwargs:
            if hypothesis.prediction.timestamp >= kwargs['time_of_interest']:
                # measurement_model = self._check_measurement_model(prev_hypothesis.measurement.measurement_model)
                plt.gca().plot(*post_state.state_vector[mapping, :], marker='o', color='b', markersize=2)
                prediction_from_predicted_meas = measurement_model.inverse_function(hypothesis.measurement_prediction)
                plt.gca().plot(*prediction_from_predicted_meas[mapping, :], marker='x', color='b', markersize=2)

        # Now update the measurement prediction mean and loop
        iterations = 1
        while self.measure(prev_post_state, post_state) > self.tolerance:

            if iterations >= self.max_iterations:
                # warnings.warn("IPLF update reached maximum number of iterations.")
                break

            if 'time_of_interest' in kwargs:
                if hypothesis.prediction.timestamp >= kwargs['time_of_interest']:
                    breakpoint()
            md = hypothesis.measurement.metadata
            true_state = State(state_vector=StateVector([float(md['TRUE_X']) * 1000, float(md['TRUE_VX']) * 1000,
                                                         float(md['TRUE_Y']) * 1000, float(md['TRUE_VY']) * 1000,
                                                         float(md['TRUE_Z']) * 1000, float(md['TRUE_VZ']) * 1000]),
                               timestamp=hypothesis.measurement.timestamp)
            # measurement_gen = measurement_model.function(true_state)

            # SLR is wrt to tne approximated posterior in post_state, not the original prior in hypothesis.prediction
            measurement_prediction = UnscentedKalmanUpdater().predict_measurement(
                predicted_state=post_state,
                measurement_model=measurement_model
            )

            # recovered_state = measurement_model.inverse_function(
            #     hypothesis.measurement_prediction
            # )
            # mapping = measurement_model.mapping
            # plt.plot(*recovered_state[measurement_model.mapping, :], marker='o', color='k', markersize=2)
            # plt.plot(*measurement_model.translation_offset, marker='o', color='g', markersize=2)
            h_matrix, b_vector, omega_cov_matrix = slr_definition(post_state, measurement_prediction)

            r_cov_matrix = measurement_model.noise_covar
            measurement_model_linearized = GeneralLinearGaussian(
                ndim_state=measurement_model.ndim_state,
                mapping=measurement_model.mapping,
                meas_matrix=h_matrix,
                bias_value=b_vector,
                noise_covar=r_cov_matrix + omega_cov_matrix)

            hypothesis.measurement_prediction = super(UnscentedKalmanUpdater, self).predict_measurement(
                predicted_state=hypothesis.prediction,
                measurement_model=measurement_model_linearized)
            hypothesis.measurement.measurement_model = measurement_model_linearized

            # if 'time_of_interest' in kwargs:
            #     if hypothesis.prediction.timestamp >= kwargs['time_of_interest']:
            #         print(f'UKF predicted measurement: '
            #               f'\n {measurement_prediction.state_vector}')
            #         print(f'Linearised predicted measurement: '
            #               f'\n {hypothesis.measurement_prediction.state_vector}')
            #         print()

            prev_post_state = post_state
            # update is computed using the original prior in hypothesis.prediction
            post_state = super(UnscentedKalmanUpdater, self).update(hypothesis, **kwargs)  # classic Kalman update
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
