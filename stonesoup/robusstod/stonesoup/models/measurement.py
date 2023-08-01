from ....base import Property
from ....types.array import CovarianceMatrix, StateVector, Matrix
from ....models.base import LinearModel, GaussianModel
from ....models.measurement.base import MeasurementModel


class GeneralLinearGaussian(MeasurementModel, LinearModel, GaussianModel):
    meas_matrix: Matrix = Property(doc="Measurement matrix")
    bias_value: StateVector = Property(doc="Bias value")
    noise_covar: CovarianceMatrix = Property(doc="Noise covariance")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.meas_matrix, Matrix):
            self.meas_matrix = Matrix(self.meas_matrix)
        if not isinstance(self.bias_value, StateVector):
            self.bias_value = StateVector(self.bias_value)
        if not isinstance(self.noise_covar, CovarianceMatrix):
            self.noise_covar = CovarianceMatrix(self.noise_covar)

    @property
    def ndim_meas(self):
        """ndim_meas getter method

        Returns
        -------
        :class:`int`
            The number of measurement dimensions
        """

        return len(self.mapping)

    def matrix(self, **kwargs): return self.meas_matrix

    def bias(self, **kwargs): return self.bias_value

    def covar(self, **kwargs):
        """Returns the measurement model noise covariance matrix.

        Returns
        -------
        :class:`~.CovarianceMatrix` of shape\
        (:py:attr:`~ndim_meas`, :py:attr:`~ndim_meas`)
            The measurement noise covariance.
        """

        return self.noise_covar

    def function(self, state, noise=False, **kwargs):
        """Model function :math:`h(t,x(t),w(t))`

        Parameters
        ----------
        state: :class:`~.State`
            An input state
        noise: :class:`numpy.ndarray` or bool
            An externally generated random process noise sample (the default is
            `False`, in which case no noise will be added
            if 'True', the output of :meth:`~.Model.rvs` is added)

        Returns
        -------
        :class:`numpy.ndarray` of shape (:py:attr:`~ndim_meas`, 1)
            The model function evaluated given the provided time interval.
        """

        if isinstance(noise, bool) or noise is None:
            if noise:
                noise = self.rvs(num_samples=state.state_vector.shape[1], **kwargs)
            else:
                noise = 0

        return self.matrix(**kwargs)@state.state_vector + self.bias_value + noise
