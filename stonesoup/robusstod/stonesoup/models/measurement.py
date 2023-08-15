from ....base import Property
from ....types.array import CovarianceMatrix, StateVector, Matrix
from ....models.base import LinearModel, GaussianModel
from ....models.measurement.base import MeasurementModel

import godot
from ....models.base import ReversibleModel
from ....models.measurement.nonlinear import NonLinearGaussianMeasurement
from ....types.array import StateVectors
import numpy as np
from scipy.linalg import inv, pinv, block_diag
from ....functions import cart2pol, pol2cart,  cart2sphere, sphere2cart, cart2angles, build_rotation_matrix
from collections.abc import Callable
from typing import Sequence, Tuple, Union
from ....types.angle import Bearing, Elevation



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


class CartesianToElevationBearingRangeGODOT(NonLinearGaussianMeasurement, ReversibleModel):
    uni: Callable = Property(doc="Universe")
    station: Callable = Property(doc="Sensor location")

    @property
    def ndim_meas(self) -> int:
        """ndim_meas getter method

        Returns
        -------
        :class:`int`
            The number of measurement dimensions
        """

        return 3

    def function(self, state, timestamp, noise=False, **kwargs) -> StateVector:

        if isinstance(noise, bool) or noise is None:
            if noise:
                noise = self.rvs()
            else:
                noise = 0

        # # Account for origin offset
        # xyz = state.state_vector[self.mapping, :] - self.translation_offset

        # # Rotate coordinates
        # xyz_rot = self._rotation_matrix @ xyz

        # # Convert to Spherical
        # rho, phi, theta = cart2sphere(*xyz_rot)
        # elevations = [Elevation(i) for i in np.atleast_1d(theta)]
        # bearings = [Bearing(i) for i in np.atleast_1d(phi)]
        # rhos = np.atleast_1d(rho)
        # 0. Turn timestamp into epoch
        timeiso = timestamp.isoformat(timespec='microseconds')
        timescale = 'TDB'
        t = ' '.join([timeiso, timescale])
        epoch = godot.core.tempo.XEpoch(t)
        # 1. Convert state to Orbital coordinates
        # 2. Convert orbital to GODOT object
        # spacecraft = godot.
        spacecraft = [state]

        rhos = self.uni.frames.distance(self.station, spacecraft, epoch)
        elevations = []
        bearings = []


        return StateVectors([elevations, bearings, rhos]) + noise

    def inverse_function(self, detection, **kwargs) -> StateVector:

        theta, phi, rho = detection.state_vector
        xyz = StateVector(sphere2cart(rho, phi, theta))

        inv_rotation_matrix = inv(self._rotation_matrix)
        xyz = inv_rotation_matrix @ xyz

        res = np.zeros((self.ndim_state, 1)).view(StateVector)
        res[self.mapping, :] = xyz + self.translation_offset

        return res

    def rvs(self, num_samples=1, **kwargs) -> Union[StateVector, StateVectors]:
        out = super().rvs(num_samples, **kwargs)
        out = np.array([[Elevation(0.)], [Bearing(0.)], [0.]]) + out
        return out

    # uni.frames.distance(station, spacecraft, epoch)