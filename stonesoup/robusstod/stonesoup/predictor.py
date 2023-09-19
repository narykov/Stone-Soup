from functools import partial
from ...predictor.kalman import ExtendedKalmanPredictor as ExtendedKalmanPredictorOriginal
from ...predictor.kalman import UnscentedKalmanPredictor
from ...predictor._utils import predict_lru_cache
from ...base import Property
from ...models.transition import TransitionModel
from ...models.control import ControlModel
from ...functions import gauss2sigma, unscented_transform
from ...types.prediction import Prediction


class ExtendedKalmanPredictor(ExtendedKalmanPredictorOriginal):
    """Updates the arguments that are passed to covariance prediction in KalmanPredictor"""

    def _predicted_covariance(self, prior, predict_over_interval, **kwargs):
        r"""Simply includes prior into kwargs of self.transition_model.covar() so as to make it possible
        to evaluate covariance using prior pdf (unlike in the original implementation)"""

        prior_cov = prior.covar
        trans_m = self._transition_matrix(prior=prior, time_interval=predict_over_interval,
                                          **kwargs)
        # trans_cov = self.transition_model.covar(time_interval=predict_over_interval, **kwargs)  # <- previous version
        trans_cov = self.transition_model.covar(prior=prior, time_interval=predict_over_interval, **kwargs)

        # As this is Kalman-like, the control model must be capable of
        # returning a control matrix (B)
        ctrl_mat = self._control_matrix
        ctrl_noi = self.control_model.control_noise

        return trans_m @ prior_cov @ trans_m.T + trans_cov + ctrl_mat @ ctrl_noi @ ctrl_mat.T


class UnscentedKalmanPredictorCrossCovariance(UnscentedKalmanPredictor):
    transition_model: TransitionModel = Property(doc="The transition model to be used.")
    control_model: ControlModel = Property(
        default=None,
        doc="The control model to be used. Default `None` where the predictor "
            "will create a zero-effect linear :class:`~.ControlModel`.")
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

    def _predicted_covariance(self, prior, predict_over_interval, **kwargs):
        r"""Simply includes prior into kwargs of self.transition_model.covar() so as to make it possible
        to evaluate covariance using prior pdf (unlike in the original implementation)"""

        prior_cov = prior.covar
        trans_m = self._transition_matrix(prior=prior, time_interval=predict_over_interval,
                                          **kwargs)
        # trans_cov = self.transition_model.covar(time_interval=predict_over_interval, **kwargs)  # <- previous version
        trans_cov = self.transition_model.covar(prior=prior, time_interval=predict_over_interval, **kwargs)

        # As this is Kalman-like, the control model must be capable of
        # returning a control matrix (B)
        ctrl_mat = self._control_matrix
        ctrl_noi = self.control_model.control_noise

        return trans_m @ prior_cov @ trans_m.T + trans_cov + ctrl_mat @ ctrl_noi @ ctrl_mat.T

    @predict_lru_cache()
    def predict_cross_covar(self, prior, timestamp=None, **kwargs):
        r"""The unscented version of the predict step

        Parameters
        ----------
        prior : :class:`~.State`
            Prior state, :math:`\mathbf{x}_{k-1}`
        timestamp : :class:`datetime.datetime`
            Time to transit to (:math:`k`)
        **kwargs : various, optional
            These are passed to :meth:`~.TransitionModel.covar`

        Returns
        -------
        : :class:`~.GaussianStatePrediction`
            The predicted state :math:`\mathbf{x}_{k|k-1}` and the predicted
            state covariance :math:`P_{k|k-1}`
        """

        # Get the prediction interval
        predict_over_interval = self._predict_over_interval(prior, timestamp)

        # The covariance on the transition model + the control model
        # TODO: Note that I'm not sure you can actually do this with the
        # TODO: covariances, i.e. sum them before calculating
        # TODO: the sigma points and then just sticking them into the
        # TODO: unscented transform, and I haven't checked the statistics.
        total_noise_covar = \
            self.transition_model.covar(
                prior=prior,
                time_interval=predict_over_interval, **kwargs) \
            + self.control_model.control_noise

        # Get the sigma points from the prior mean and covariance.
        sigma_point_states, mean_weights, covar_weights = gauss2sigma(
            prior, self.alpha, self.beta, self.kappa)

        # This ensures that function passed to unscented transform has the
        # correct time interval
        transition_and_control_function = partial(
            self._transition_and_control_function,
            time_interval=predict_over_interval)

        # Put these through the unscented transform, together with the total
        # covariance to get the parameters of the Gaussian
        _, _, cross_covar, _, _, _ = unscented_transform(
            sigma_point_states, mean_weights, covar_weights,
            transition_and_control_function, covar_noise=total_noise_covar
        )

        # and return a Gaussian state based on these parameters
        return cross_covar

    @predict_lru_cache()
    def predict(self, prior, timestamp=None, **kwargs):
        r"""The unscented version of the predict step

        Parameters
        ----------
        prior : :class:`~.State`
            Prior state, :math:`\mathbf{x}_{k-1}`
        timestamp : :class:`datetime.datetime`
            Time to transit to (:math:`k`)
        **kwargs : various, optional
            These are passed to :meth:`~.TransitionModel.covar`

        Returns
        -------
        : :class:`~.GaussianStatePrediction`
            The predicted state :math:`\mathbf{x}_{k|k-1}` and the predicted
            state covariance :math:`P_{k|k-1}`
        """

        # Get the prediction interval
        predict_over_interval = self._predict_over_interval(prior, timestamp)

        # The covariance on the transition model + the control model
        # TODO: Note that I'm not sure you can actually do this with the
        # TODO: covariances, i.e. sum them before calculating
        # TODO: the sigma points and then just sticking them into the
        # TODO: unscented transform, and I haven't checked the statistics.
        total_noise_covar = \
            self.transition_model.covar(
                prior=prior,
                time_interval=predict_over_interval, **kwargs) \
            + self.control_model.control_noise

        # Get the sigma points from the prior mean and covariance.
        sigma_point_states, mean_weights, covar_weights = gauss2sigma(
            prior, self.alpha, self.beta, self.kappa)

        # This ensures that function passed to unscented transform has the
        # correct time interval
        transition_and_control_function = partial(
            self._transition_and_control_function,
            time_interval=predict_over_interval)

        # Put these through the unscented transform, together with the total
        # covariance to get the parameters of the Gaussian
        x_pred, p_pred, _, _, _, _ = unscented_transform(
            sigma_point_states, mean_weights, covar_weights,
            transition_and_control_function, covar_noise=total_noise_covar
        )

        # and return a Gaussian state based on these parameters
        return Prediction.from_state(prior, x_pred, p_pred, timestamp=timestamp,
                                     transition_model=self.transition_model)