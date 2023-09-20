import numpy as np

from ...base import Property
from ...models.transition.base import TransitionModel
from ...smoother.kalman import KalmanSmoother, ExtendedKalmanSmoother, UnscentedKalmanSmoother
from ...types.hypothesis import SingleHypothesis
from ...types.track import Track
from ...updater.kalman import KalmanUpdater
from ...predictor.kalman import UnscentedKalmanPredictor, KalmanPredictor

from .predictor import ExtendedKalmanPredictor
from .updater import IPLFKalmanUpdater
from .models.measurement import GeneralLinearGaussian
from .models.transition import LinearGaussianTransitionModel
from .predictor import UnscentedKalmanPredictorCrossCovariance


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
        doc="Number of iterations.")


    def _slr_tm_calculations(self, prediction, transition_model, timestamp, **kwargs):
        """
        Notation follows https://github.com/Agarciafernandez/IPLF/blob/main/IPLF_maneuvering.m
        """

        mean_pos = prediction.state_vector
        cov_pos = prediction.covar

        predictor = UnscentedKalmanPredictorCrossCovariance(transition_model=transition_model)
        state_prediction = predictor.predict(prediction, timestamp=timestamp, **kwargs)
        z_pred = state_prediction.state_vector
        var_pred = state_prediction.covar
        # var_xz = state_prediction.cross_covar
        # we defined this method separately since state prediction does not return cross_covariance
        var_xz = predictor.predict_cross_covar(prediction, timestamp=timestamp, **kwargs)

        # measurement_prediction = self.predict_measurement(
        #     predicted_state=prediction,
        #     measurement_model=measurement_model, **kwargs)  # using sigma points in UKF
        # z_pred = measurement_prediction.state_vector
        # var_pred = measurement_prediction.covar  # = Phi
        # var_xz = measurement_prediction.cross_covar  # = Psi

        # Statistical linear regression parameters
        A_l = var_xz.T @ np.linalg.inv(cov_pos)
        b_l = z_pred - A_l @ mean_pos
        Omega_l = var_pred - A_l @ cov_pos @ A_l.T

        return {'A_l': A_l, 'b_l': b_l, 'Omega_l': Omega_l}

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

        # track_smoothed = super().smooth(track)  # <- this doesn't make it to the UKF smoother
        track_smoothed = UnscentedKalmanSmoother.smooth(self, track)  # <- UKF smoother
        measurement_model = track[-1].hypothesis.measurement.measurement_model

        # predictor = ExtendedKalmanPredictor(self.transition_model)
        updater_iplf = IPLFKalmanUpdater()  # this is only to use its _slr_calculations() method
        updater_kf = KalmanUpdater()  # this is for the RTS smoother

        for n in range(self.n_iterations):
            # print(n)

            track_forward = Track()


            for smoothed_update in track_smoothed:

                # Get prior/predicted pdf
                timestamp = smoothed_update.timestamp  # check if first time step
                if timestamp == track_smoothed[0].timestamp:
                    # prev_state has not been defined yet (i.e., this is the first time s tep)
                    prediction = smoothed_update.hypothesis.prediction  # get original prior from hypothesis
                    prediction.transition_model = None
                    print()
                    # if
                else:
                    # prev_state has been defined in the first time step
                    transition_model = self.transition_model
                    slr_tm = self._slr_tm_calculations(smoothed_update, transition_model, timestamp)
                    # print(slr_tm)
                    time_interval = smoothed_update.timestamp - prev_state.timestamp
                    covar = transition_model.covar(time_interval=time_interval, prior=prev_state)
                    transition_model_linearised = LinearGaussianTransitionModel(
                        transition_matrix=slr_tm['A_l'],
                        bias_value=slr_tm['b_l'],
                        noise_covar=covar + slr_tm['Omega_l']
                    )
                    if n > 0:
                        print()
                    predictor_kf = KalmanPredictor(transition_model_linearised)
                    prediction = predictor_kf.predict(prev_state, timestamp=timestamp)  # calculate prior from previous update

                    # print()

                # Get linearization wrt a smoothed posterior
                # measurement_model = smoothed_update.hypothesis.measurement.measurement_model
                print(timestamp)
                if n > 0:
                    try:
                        np.linalg.cholesky(smoothed_update.covar)
                    except ValueError:
                        print('Bad covariance')
                    print()
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

            from ..utils import plot_tracks
            from matplotlib import pyplot as plt
            # for step, (time, current_tracks) in enumerate(tracker, 0):
            #     pixels = reader.sensor_data.pixels
            #     im = plt.imshow(pixels, interpolation='none', origin='lower', cmap='jet',
            #                     extent=[-rng_cutoff, rng_cutoff, -rng_cutoff, rng_cutoff], vmin=0, vmax=255)
            #     cbar = plt.colorbar(im, orientation='vertical')
            #     # cbar.set_label('Receiver units')
            #     plot_detections(detector.detections)
            #     tracks.update(current_tracks)
            plot_tracks([track], color='r')
            plot_tracks([track_smoothed], color='g')
                # print("Step: {} Time: {}".format(step, time))
                # plt.title(time.strftime('%Y-%m-%d %H:%M:%S'))
                # plt.gca().set_xlabel('Eastings, [m]')
                # plt.gca().set_ylabel('Northings, [m]')
                # plt.gca().set_xlim([-rng_cutoff, rng_cutoff])
                # plt.gca().set_ylim([-rng_cutoff, rng_cutoff])
                # name = 'image' + str(step).zfill(6)
                # fig.savefig('img/{}.png'.format(name), dpi=192)
                # plt.pause(0.05)
                # plt.clf()

            track_smoothed = KalmanSmoother.smooth(self, track_forward)
            plot_tracks([track_forward], color='m')
            plot_tracks([track_smoothed], color='y')
            print()


            # track_smoothed = ExtendedKalmanSmoother(self.transition_model).smooth(track_forward)
            # we do not use KF implementation in order to accomodate the way transition matrix is returned by Van Loan

        return track_smoothed
