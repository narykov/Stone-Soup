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
from .models.transition import LinearTransitionModel
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

        for i in range(self.n_iterations):
            print(f'Iteration {i+1} out of {self.n_iterations}')
            """ Compute SLR parameters. """
            if i == 0:
                # First, we perform initialisation by performing sigma-point smoothing
                # below is the UKF smoother; using super().smooth(track) doesn't reach the UKF smoother
                smoothed_track = UnscentedKalmanSmoother.smooth(self, track)
                smoothed_tracks = [smoothed_track]
                continue

            ndim_meas = measurement_model.ndim_meas
            ndim_state = measurement_model.ndim_state

            depth = len(track)-1  # beacuse no measurement on the first time step
            f_slr = np.empty((depth, ndim_state, ndim_state))
            a_slr = np.empty((depth, ndim_state, 1))
            lambda_slr = np.empty((depth, ndim_state, ndim_state))
            h_slr = np.empty((depth, ndim_meas, ndim_state))
            b_slr = np.empty((depth, ndim_meas, 1))
            omega_slr = np.empty((depth, ndim_meas, ndim_meas))

            k = 0
            for current_update in smoothed_tracks[-1]:
                if current_update.timestamp == smoothed_tracks[-1][0].timestamp:
                    previous_update = current_update
                    continue

                # Do the linearisation and state prediction to the current step
                predicted_state = AugmentedUnscentedKalmanPredictor(transition_model=transition_model).predict(
                    previous_update, timestamp=current_update.timestamp)
                f_slr[k], a_slr[k], lambda_slr[k] = slr_definition(previous_update, predicted_state)
                predicted_measurement = UnscentedKalmanUpdater().predict_measurement(
                    predicted_state=current_update, measurement_model=measurement_model)
                h_slr[k], b_slr[k], omega_slr[k] = slr_definition(current_update, predicted_measurement)

                # slr_parameters = {**slr_transition_dict, **slr_measurement_dict}
                previous_update = current_update
                k += 1

            theta = {'F': f_slr, 'a': a_slr, 'Lambda': lambda_slr, 'H': h_slr, 'b': b_slr, 'Omega': omega_slr}

            """ Perform forward-backward smoothing using the SLR parameters. """
            track_forward = Track(track[0])  # based entirely on the linear approximations
            k = 0
            for current_update in track:
                if current_update.timestamp == smoothed_tracks[-1][0].timestamp:
                    previous_update = current_update
                    continue
                """
                TIME UPDATE
                """
                "Linearising the state prediction and perform linear prediction"

                # Necessary timings for state prediction
                predict_over_interval = current_update.timestamp - previous_update.timestamp
                q_cov_matrix = transition_model.covar(time_interval=predict_over_interval, prior=track_forward[-1])
                transition_model_linearised = LinearTransitionModel(
                    transition_matrix=theta['F'][k],
                    bias_value=theta['a'][k],
                    noise_covar=q_cov_matrix + theta['Lambda'][k]
                )
                prediction_linear = KalmanPredictor(transition_model_linearised).predict(
                    track_forward[-1], timestamp=current_update.timestamp
                )
                # prediction_linear.transition_model = self.transition_model
                """
                DATA UPDATE
                """
                "Linearising the measurement prediction and perform linear measurement update"
                r_cov_matrix = measurement_model.noise_covar
                measurement_model_linearized = GeneralLinearGaussian(
                    ndim_state=measurement_model.ndim_state,
                    mapping=measurement_model.mapping,
                    meas_matrix=theta['H'][k],
                    bias_value=theta['b'][k],
                    noise_covar=r_cov_matrix + theta['Omega'][k])

                # Get the actual measurement plus its prediction for the above model using the predicted pdf
                measurement = current_update.hypothesis.measurement
                measurement.measurement_model = measurement_model_linearized
                hypothesis = SingleHypothesis(prediction=prediction_linear,
                                              measurement=measurement)
                # using measurement_prediction=measurement_prediction is not necessary

                update_linear = KalmanUpdater().update(hypothesis)
                update_linear.hypothesis.measurement.measurement_model = measurement_model  # restores the model
                track_forward.append(update_linear)
                k += 1



            from ..utils import plot_tracks, plot_ground_truth, plot_detections, plot_linear_detections
            from matplotlib import pyplot as plt
            points = [update.hypothesis.prediction.state_vector for update in track_forward[1:]]
            points_orig = [update.hypothesis.prediction.state_vector for update in track[1:]]
            plot_ground_truth([track[1].hypothesis.measurement.groundtruth_path])
            plot_detections([state.hypothesis.measurement for state in track[1:]])
            plot_tracks([track], color='r', label='IPLF')
            plot_tracks([smoothed_track], color='g', label='UKS')
            plot_tracks([track_forward], color='m', label='IPLS (fwd only)')
            for point in points:
                plt.plot(point[0], point[2], marker='.', color='m')
            for point in points_orig:
                plt.plot(point[0], point[2], marker='.', color='r')
            # measurement predictions
            mp = [state.hypothesis.measurement for state in track_forward[1:]]
            plot_linear_detections(mp, color='y')
            transition_model_dummy = CombinedLinearGaussianTransitionModel(
                [ConstantVelocity(0), ConstantVelocity(0), ConstantVelocity(0)])
            smoother_here = KalmanSmoother(transition_model=transition_model_dummy)
            track_smoothed = smoother_here.smooth(track_forward)
            smoothed_tracks.append(track_smoothed)
            plot_tracks([track_smoothed], color='y', label='IPLS')
            plt.legend()
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


# class IPLSKalmanSmootherFull(IPLSKalmanSmoother):
#     r"""The unscented version of the Kalman filter. As with the parent version of the Kalman
#     smoother, the mean and covariance of the prediction are retrieved from the track. The
#     unscented transform is used to calculate the smoothing gain.
#
#     """
#     transition_model: TransitionModel = Property(doc="The transition model to be used.")
#
#     alpha: float = Property(
#         default=0.5,
#         doc="Primary sigma point spread scaling parameter. Default is 0.5.")
#     beta: float = Property(
#         default=2,
#         doc="Used to incorporate prior knowledge of the distribution. If the "
#             "true distribution is Gaussian, the value of 2 is optimal. "
#             "Default is 2")
#     kappa: float = Property(
#         default=0,
#         doc="Secondary spread scaling parameter. Default is calculated as "
#             "3-Ns")
#     n_iterations: int = Property(
#         default=5,
#         doc="Number of smoothing iterations.")
#
#     def smooth(self, track):
#         """
#         Execute the IPLS algorithm.
#
#         Parameters
#         ----------
#         track : :class:`~.Track`
#             The input track.
#
#         Returns
#         -------
#          : :class:`~.Track`
#             Smoothed track
#
#         """
#
#         # A filtered track is the input to this smoother.
#
#         transition_model = self.transition_model
#         measurement_model = track[-1].hypothesis.measurement.measurement_model
#         thetas = []
#
#         for i in range(self.n_iterations):
#             print(f'Iteration {i + 1} out of {self.n_iterations}')
#             theta = []
#             """ Compute SLR parameters. """
#             if i == 0:
#                 # First, we perform initialisation by performing sigma-point smoothing
#                 # below is the UKF smoother; using super().smooth(track) doesn't reach the UKF smoother
#                 smoothed_track = UnscentedKalmanSmoother.smooth(self, track)
#                 smoothed_tracks = [smoothed_track]
#                 continue
#
#             ndim_meas = measurement_model.ndim_meas
#             ndim_state = measurement_model.ndim_state
#
#             depth = len(track)  # beacuse no measurement on the first time step
#             f_slr = np.empty((depth, ndim_state, ndim_state))
#             a_slr = np.empty((depth, ndim_state, 1))
#             lambda_slr = np.empty((depth, ndim_state, ndim_state))
#             h_slr = np.empty((depth, ndim_meas, ndim_state))
#             b_slr = np.empty((depth, ndim_meas, 1))
#             omega_slr = np.empty((depth, ndim_meas, ndim_meas))
#
#             k = 0
#             for current_update in smoothed_tracks[-1]:
#                 # if current_update.timestamp == smoothed_tracks[-1][0].timestamp:
#                 #     previous_update = current_update
#                 #     continue
#
#                 predicted_measurement = UnscentedKalmanUpdater().predict_measurement(
#                     predicted_state=current_update, measurement_model=measurement_model
#                 )
#                 slr_measurement = slr_definition(current_update, predicted_measurement)
#                 h_slr[k], b_slr[k], omega_slr[k] = list(slr_measurement.values())
#                 # slr_measurement_dict = dict(zip(['H', 'b', 'Omega'], [*slr_measurement.values()]))
#
#                 # Do the linearisation and state prediction to the current step
#                 predicted_state = AugmentedUnscentedKalmanPredictor(transition_model=transition_model).predict(
#                     current_update, timestamp=current_update.timestamp + timedelta(seconds=50)
#                 )
#                 slr_transition = slr_definition(current_update, predicted_state)
#                 f_slr[k], a_slr[k], lambda_slr[k] = list(slr_transition.values())
#                 # slr_transition_dict = dict(zip(['F', 'a', 'Lambda'], [*slr_transition.values()]))
#
#                 # slr_parameters = {**slr_transition_dict, **slr_measurement_dict}
#                 k += 1
#                 # previous_update = current_update
#
#             # theta.append(slr_parameters)
#             theta = {'F': f_slr, 'a': a_slr, 'Lambda': lambda_slr, 'H': h_slr, 'b': b_slr, 'Omega': omega_slr}
#             thetas.append(theta)
#             """ Perform forward-backward smoothing using the SLR parameters. """
#
#             track_forward = Track()  # based entirely on the linear approximations
#             k = 0
#             prediction_linear = track[0].hypothesis.prediction
#             for current_update in track:
#                 # if current_update.timestamp == smoothed_tracks[-1][0].timestamp:
#                 #     prediction_linear = track_forward[-1]
#                 #     continue
#                 """
#
#                 DATA UPDATE
#                 """
#                 "Linearising the measurement prediction and perform linear measurement update"
#                 r_cov_matrix = measurement_model.noise_covar
#                 measurement_model_linearized = GeneralLinearGaussian(
#                     ndim_state=measurement_model.ndim_state,
#                     mapping=measurement_model.mapping,
#                     meas_matrix=theta['H'][k],
#                     bias_value=theta['b'][k],
#                     noise_covar=r_cov_matrix + theta['Omega'][k])
#
#                 # Get the actual measurement plus its prediction for the above model using the predicted pdf
#                 measurement = current_update.hypothesis.measurement
#                 measurement_prediction = KalmanUpdater().predict_measurement(
#                     predicted_state=prediction_linear,
#                     measurement_model=measurement_model_linearized
#                 )
#                 measurement.measurement_model = measurement_model_linearized
#
#                 hypothesis = SingleHypothesis(prediction=prediction_linear,
#                                               measurement=measurement,
#                                               measurement_prediction=measurement_prediction)
#
#                 update_linear = KalmanUpdater().update(hypothesis)
#                 update_linear.hypothesis.measurement.measurement_model = measurement_model  # restores the model
#                 track_forward.append(update_linear)
#
#                 """
#                 TIME UPDATE
#                 """
#                 "Linearising the state prediction and perform linear prediction"
#
#                 # Necessary timings for state prediction
#                 timestamp_current = current_update.timestamp
#                 # timestamp_previous = previous_update.timestamp
#                 # predict_over_interval = timestamp_current - timestamp_previous
#                 q_cov_matrix = transition_model.covar(time_interval=timedelta(seconds=50), prior=update_linear)
#                 transition_model_linearised = LinearTransitionModel(
#                     transition_matrix=theta['F'][k],
#                     bias_value=theta['a'][k],
#                     noise_covar=q_cov_matrix + theta['Lambda'][k]
#                 )
#                 predictor = KalmanPredictor1(transition_model_linearised)
#                 prediction_linear = predictor.predict(
#                     update_linear, timestamp=timestamp_current
#                 )
#                 # prediction_linear.transition_model = self.transition_model
#
#             from ..utils import plot_tracks, plot_ground_truth, plot_detections
#             from matplotlib import pyplot as plt
#             plot_ground_truth([track[1].hypothesis.measurement.groundtruth_path])
#             plot_detections([state.hypothesis.measurement for state in track[1:]])
#             plot_tracks([track], color='r', label='IPLF')
#             plot_tracks([smoothed_track], color='g', label='smoothed UKF')
#             plot_tracks([track_forward], color='m', label='IPLS (fwd only)')
#
#             transition_model_dummy = CombinedLinearGaussianTransitionModel(
#                 [ConstantVelocity(0), ConstantVelocity(0), ConstantVelocity(0)])
#             smoother_here = KalmanSmoother(transition_model=transition_model_dummy)
#             track_smoothed = smoother_here.smooth(track_forward)
#             plot_tracks([track_smoothed], color='y', label='IPLS')
#             plt.legend()
#             print()
#             # track_smoothed = KalmanSmoother.smooth(self, track_forward)
#             # for upd in track_smoothed:
#             #     try:
#             #         np.linalg.cholesky(upd.covar)
#             #     except ValueError:
#             #         print('Bad covariance')
#             #         print()
#
#             # print()
#             # plot_tracks([track_forward], color='m')
#             # plot_tracks([track_smoothed], color='y')
#             # print()
#
#             # track_smoothed = ExtendedKalmanSmoother(self.transition_model).smooth(track_forward)
#             # we do not use KF implementation in order to accomodate the way transition matrix is returned by Van Loan
#
#         return track_smoothed