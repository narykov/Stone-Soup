import warnings

from ...base import Property
from ...smoother.kalman import KalmanSmoother, UnscentedKalmanSmoother
from ...types.hypothesis import SingleHypothesis
from ...types.track import Track
from ...updater.kalman import KalmanUpdater, UnscentedKalmanUpdater
from ...models.transition.base import TransitionModel
from ...models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity


from .models.measurement import GeneralLinearGaussian
from .models.transition import LinearTransitionModel
from .predictor import AugmentedUnscentedKalmanPredictor
from .predictor import AugmentedKalmanPredictor
from .functions import slr_definition


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

        measurement_model = track[-1].hypothesis.measurement.measurement_model
        smoothed_tracks = []

        while True:
            # we have no test of convergence, but limited the number of iterations
            if len(smoothed_tracks) >= self.n_iterations:
                warnings.warn("IPLS reached pre-specified number of iterations.")
                break
            print(f'IPLS iteration {len(smoothed_tracks) + 1} out of {self.n_iterations}')

            if not smoothed_tracks:
                # initialising by performing sigma-point smoothing via the UKF smoother
                smoothed_track = UnscentedKalmanSmoother.smooth(self, track)
                smoothed_track_uks = smoothed_track
                smoothed_tracks.append(smoothed_track)
                continue

            track_forward = Track(track[0])  # starting the new forward track to be

            for current_state in smoothed_track:

                if current_state.timestamp == smoothed_track[0].timestamp:
                    previous_state = current_state
                    continue

                """ Compute SLR parameters. """
                #TODO: check if any models are linear and skip linearisation
                current_prediction = AugmentedUnscentedKalmanPredictor(transition_model=self.transition_model).predict(
                    prior=previous_state,
                    timestamp=current_state.timestamp
                )
                measurement_prediction = UnscentedKalmanUpdater().predict_measurement(
                    predicted_state=current_state,
                    measurement_model=measurement_model
                )
                f_matrix, a_vector, lambda_cov_matrix = slr_definition(previous_state, current_prediction)
                h_matrix, b_vector, omega_cov_matrix = slr_definition(current_state, measurement_prediction)

                "Perform linear time update"
                q_cov_matrix = self.transition_model.covar(
                    time_interval=current_state.timestamp - previous_state.timestamp, prior=track_forward[-1]
                )
                transition_model_linearised = LinearTransitionModel(
                    transition_matrix=f_matrix,
                    bias_value=a_vector,
                    noise_covar=q_cov_matrix + lambda_cov_matrix
                )
                prediction_linear = AugmentedKalmanPredictor(transition_model_linearised).predict(
                    track_forward[-1], timestamp=current_state.timestamp
                )

                "Perform linear data update"
                r_cov_matrix = measurement_model.noise_covar
                measurement_model_linearized = GeneralLinearGaussian(
                    ndim_state=measurement_model.ndim_state,
                    mapping=measurement_model.mapping,
                    meas_matrix=h_matrix,
                    bias_value=b_vector,
                    noise_covar=r_cov_matrix + omega_cov_matrix
                )

                # Get the actual measurement plus its prediction for the above model using the predicted pdf
                measurement = current_state.hypothesis.measurement
                measurement.measurement_model = measurement_model_linearized
                hypothesis = SingleHypothesis(prediction=prediction_linear, measurement=measurement)
                update_linear = KalmanUpdater().update(hypothesis)

                # restores the model (ensures visualisation is OK)
                update_linear.hypothesis.measurement.measurement_model = measurement_model
                # append the track with an update (that contains hypothesis and info needed for the backwards go)
                track_forward.append(update_linear)

                previous_state = current_state

            transition_model_dummy = CombinedLinearGaussianTransitionModel(
                len(measurement_model.mapping) * [ConstantVelocity(0)]
            )
            linear_smoother = KalmanSmoother(transition_model=transition_model_dummy)
            smoothed_track = linear_smoother.smooth(track_forward)
            smoothed_tracks.append(smoothed_track)

            # from ..utils import plot_tracks, plot_ground_truth, plot_detections, plot_linear_detections
            # from matplotlib import pyplot as plt
            # points = [update.hypothesis.prediction.state_vector for update in track_forward[1:]]
            # points_orig = [update.hypothesis.prediction.state_vector for update in track[1:]]
            # plot_ground_truth([track[1].hypothesis.measurement.groundtruth_path])
            # plot_detections([state.hypothesis.measurement for state in track[1:]])
            # plot_tracks([track], color='r', label='IPLF')
            # plot_tracks([smoothed_track_uks], color='g', label='UKS')
            # plot_tracks([track_forward], color='m', label='IPLS (fwd only)')
            # for point in points:
            #     plt.plot(point[0], point[2], marker='.', color='m')
            # for point in points_orig:
            #     plt.plot(point[0], point[2], marker='.', color='r')
            # # measurement predictions
            # mp = [state.hypothesis.measurement for state in track_forward[1:]]
            # plot_linear_detections(mp, color='y')
            # plot_tracks([smoothed_track], color='y', label='IPLS')
            # plt.legend()
            # print()

        return smoothed_tracks[-1]
