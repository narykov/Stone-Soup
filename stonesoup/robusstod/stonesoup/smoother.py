import numpy as np

from ...base import Property
from ...models.transition.base import TransitionModel
from ...smoother.kalman import KalmanSmoother, ExtendedKalmanSmoother, UnscentedKalmanSmoother
from ...types.hypothesis import SingleHypothesis
from ...types.track import Track
from ...updater.kalman import KalmanUpdater, UnscentedKalmanUpdater
from ...predictor.kalman import UnscentedKalmanPredictor, KalmanPredictor

from .predictor import ExtendedKalmanPredictor
from .updater import IPLFKalmanUpdater
from .models.measurement import GeneralLinearGaussian
from .models.transition import LinearGaussianTransitionModel
from .predictor import AugmentedUnscentedKalmanPredictor

from .functions import slr_definition

from ...models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity


class IPLSKalmanSmoother(UnscentedKalmanSmoother):
    r"""The unscented version of the Kalman filter. As with the parent version of the Kalman
    smoother, the mean and covariance of the prediction are retrieved from the track. The
    unscented transform is used to calculate the smoothing gain.

    """
    transition_model: TransitionModel = Property(doc="The transition model to be used.")

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
    n_iterations: int = Property(
        default=5,
        doc="Number of iterations.")

    def smooth(self, track):
        """
        Execute the IPLS algorithm.

        Parameters
        ----------
        track : :class:`~.Track`
            The input track.

        Returns
        -------
         : :class:`~.Track`
            Smoothed track

        """

        # A filtered track is the input to this smoother.
        # First, we perform initialisation by performing sigma-point smoothing

        # track_smoothed = super().smooth(track)  # <- UKF smoother
        track_smoothed = UnscentedKalmanSmoother.smooth(self, track)  # <- UKF smoother
        measurement_model = track[-1].hypothesis.measurement.measurement_model

        predictor = ExtendedKalmanPredictor(self.transition_model)
        # predictor = UnscentedKalmanPredictor(self.transition_model)
        updater_iplf = IPLFKalmanUpdater()  # this is only to use its _slr_calculations() method
        updater_kf = KalmanUpdater()  # this is for the RTS smoother

        for n in range(self.n_iterations):

            track_forward = Track()

            for smoothed_update in track_smoothed:

                # Get prior/predicted pdf
                timestamp = smoothed_update.timestamp  # check if first iteration
                if timestamp == track_smoothed[0].timestamp:
                    prediction = smoothed_update.hypothesis.prediction  # get original prior from hypothesis
                else:
                    prediction = predictor.predict(prev_state,
                                                   timestamp=timestamp)  # calculate prior from previous update


                # Get linearization wrt a smoothed posterior
                # measurement_model = smoothed_update.hypothesis.measurement.measurement_model
                slr = updater_iplf._slr_calculations(smoothed_update, measurement_model)
                measurement_model_linearized = GeneralLinearGaussian(
                    ndim_state=measurement_model.ndim_state,
                    mapping=measurement_model.mapping,
                    meas_matrix=slr['A_l'],
                    bias_value=slr['b_l'],
                    noise_covar=measurement_model.noise_covar + slr['Omega_l'])

                # Get the measurement plus its prediction for the above model using the predicted pdf
                measurement = smoothed_update.hypothesis.measurement
                measurement_prediction = updater_kf.predict_measurement(
                    predicted_state=prediction,
                    measurement_model=measurement_model_linearized
                )
                measurement.measurement_model = measurement_model_linearized

                hypothesis = SingleHypothesis(prediction=prediction,
                                              measurement=measurement,
                                              measurement_prediction=measurement_prediction)

                update = updater_kf.update(hypothesis)
                track_forward.append(update)
                prev_state = update

            track_smoothed = ExtendedKalmanSmoother(self.transition_model).smooth(track_forward)
            # we do not use KF implementation in order to accomodate the way transition matrix is returned by Van Loan

        return track_smoothed


class IPLSKalmanSmootherFull(IPLSKalmanSmoother):
    r"""The unscented version of the Kalman filter. As with the parent version of the Kalman
    smoother, the mean and covariance of the prediction are retrieved from the track. The
    unscented transform is used to calculate the smoothing gain.

    """
    transition_model: TransitionModel = Property(doc="The transition model to be used.")

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
    n_iterations: int = Property(
        default=5,
        doc="Number of smoothing iterations.")

    def smooth(self, track):
        """
        Execute the IPLS algorithm.

        Parameters
        ----------
        track : :class:`~.Track`
            The input track.

        Returns
        -------
         : :class:`~.Track`
            Smoothed track

        """

        # A filtered track is the input to this smoother.

        transition_model = self.transition_model
        measurement_model = track[-1].hypothesis.measurement.measurement_model

        # First, we perform initialisation by performing sigma-point smoothing
        # below is the UKF smoother; using super().smooth(track) doesn't reach the UKF smoother
        track_smoothed = UnscentedKalmanSmoother.smooth(self, track)

        for n in range(self.n_iterations):
            print(f'Iteration {n+1} out of {self.n_iterations}')

            track_forward = Track(track[0])  # based entirely on the linear approximations

            previous_update = track_smoothed[0]  #

            for current_update in track_smoothed[1:]:

                # Necessary timings for state prediction
                timestamp_current = current_update.timestamp
                timestamp_previous = previous_update.timestamp
                time_interval = timestamp_current - timestamp_previous

                "Linearising the state prediction and perform linear prediction"
                # Do the linearisation and state prediction to the current step
                predicted_state = AugmentedUnscentedKalmanPredictor(transition_model=transition_model).predict(
                    previous_update, timestamp=timestamp_current
                )
                slr_transition = slr_definition(previous_update, predicted_state)
                q_cov_matrix = transition_model.covar(time_interval=time_interval, prior=previous_update)
                transition_model_linearised = LinearGaussianTransitionModel(
                    transition_matrix=slr_transition['matrix'],
                    bias_value=slr_transition['vector'],
                    noise_covar=q_cov_matrix + slr_transition['cov_matrix']
                )
                prediction_linear = KalmanPredictor(transition_model_linearised).predict(
                    track_forward[-1], timestamp=timestamp_current
                )
                # prediction_linear.transition_model = self.transition_model

                "Linearising the measurement prediction and perform linear measurement update"
                predicted_measurement = UnscentedKalmanUpdater().predict_measurement(
                    predicted_state=current_update, measurement_model=measurement_model
                )
                slr_measurement = slr_definition(current_update, predicted_measurement)
                r_cov_matrix = measurement_model.noise_covar
                measurement_model_linearized = GeneralLinearGaussian(
                    ndim_state=measurement_model.ndim_state,
                    mapping=measurement_model.mapping,
                    meas_matrix=slr_measurement['matrix'],
                    bias_value=slr_measurement['vector'],
                    noise_covar=r_cov_matrix + slr_measurement['cov_matrix'])

                # Get the actual measurement plus its prediction for the above model using the predicted pdf
                measurement = current_update.hypothesis.measurement
                measurement_prediction = KalmanUpdater().predict_measurement(
                    predicted_state=prediction_linear,
                    measurement_model=measurement_model_linearized
                )
                measurement.measurement_model = measurement_model_linearized

                hypothesis = SingleHypothesis(prediction=prediction_linear,
                                              measurement=measurement,
                                              measurement_prediction=measurement_prediction)

                update_linear = KalmanUpdater().update(hypothesis)
                update_linear.hypothesis.measurement.measurement_model = measurement_model  # restores the model
                track_forward.append(update_linear)
                previous_update = current_update


            from ..utils import plot_tracks
            plot_tracks([track], color='r')
            plot_tracks([track_smoothed], color='g')
            plot_tracks([track_forward], color='m')

            transition_model_dummy = CombinedLinearGaussianTransitionModel(
                [ConstantVelocity(0), ConstantVelocity(0), ConstantVelocity(0)])
            smoother_here = KalmanSmoother(transition_model=transition_model_dummy)
            track_smoothed = smoother_here.smooth(track_forward)
            plot_tracks([track_smoothed], color='y')
            print()
            # track_smoothed = KalmanSmoother.smooth(self, track_forward)
            # for upd in track_smoothed:
            #     try:
            #         np.linalg.cholesky(upd.covar)
            #     except ValueError:
            #         print('Bad covariance')
            #         print()

            # print()
            # plot_tracks([track_forward], color='m')
            # plot_tracks([track_smoothed], color='y')
            # print()


            # track_smoothed = ExtendedKalmanSmoother(self.transition_model).smooth(track_forward)
            # we do not use KF implementation in order to accomodate the way transition matrix is returned by Van Loan

        return track_smoothed
