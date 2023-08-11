from ...base import Property
from ...models.transition.base import TransitionModel
from ...smoother.kalman import ExtendedKalmanSmoother, UnscentedKalmanSmoother
from ...types.hypothesis import SingleHypothesis
from ...types.track import Track
from ...updater.kalman import KalmanUpdater

from .predictor import ExtendedKalmanPredictor
from .updater import IPLFKalmanUpdater
from .models.measurement import GeneralLinearGaussian


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

        track_smoothed = super().smooth(track)  # <- UKF smoother

        predictor = ExtendedKalmanPredictor(self.transition_model)
        updater_iplf = IPLFKalmanUpdater()  # this is only to use its _slr_calculations() method
        updater_kalman = KalmanUpdater()  # this is for the RTS smoother

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
                measurement_model = smoothed_update.hypothesis.measurement.measurement_model
                slr = updater_iplf._slr_calculations(smoothed_update, measurement_model)
                measurement_model_linearized = GeneralLinearGaussian(
                    ndim_state=measurement_model.ndim_state,
                    mapping=measurement_model.mapping,
                    meas_matrix=slr['A_l'],
                    bias_value=slr['b_l'],
                    noise_covar=measurement_model.noise_covar + slr['Omega_l'])

                # Get the measurement plus its prediction for the above model using the predicted pdf
                measurement = smoothed_update.hypothesis.measurement
                measurement_prediction = updater_kalman.predict_measurement(
                    predicted_state=prediction,
                    measurement_model=measurement_model_linearized
                )

                hypothesis = SingleHypothesis(prediction=prediction,
                                              measurement=measurement,
                                              measurement_prediction=measurement_prediction)

                update = updater_kalman.update(hypothesis)
                track_forward.append(update)
                prev_state = update

            track_smoothed = ExtendedKalmanSmoother(self.transition_model).smooth(track_forward)
            # we do not use KF implementation in order to accomodate the way transition matrix is returned by Van Loan

        return track_smoothed
