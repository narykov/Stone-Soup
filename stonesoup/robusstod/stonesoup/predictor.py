from ....stonesoup.predictor.kalman import ExtendedKalmanPredictor as ExtendedKalmanPredictorOriginal


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