from ....base import Property
from ....types.array import CovarianceMatrix, StateVector, Matrix
from ....models.base import LinearModel, GaussianModel
from ....models.measurement.base import MeasurementModel

# import godot
from ....models.base import ReversibleModel
from ....models.measurement.nonlinear import NonLinearGaussianMeasurement
from ....types.array import StateVectors
import numpy as np
from scipy.linalg import inv, pinv, block_diag
from ....functions import cart2pol, pol2cart,  cart2sphere, sphere2cart, cart2angles, build_rotation_matrix
from collections.abc import Callable
from typing import Sequence, Tuple, Union
from ....types.angle import Bearing, Elevation
# from ...physics.godot import rng


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


# class CartesianToElevationBearingRangeGODOT(NonLinearGaussianMeasurement, ReversibleModel):
#     #TODO: rename the function to ROBUSSTOD, and maybe inherit CartesianToElevationBearingRange for simplicity
#
#     uni: Callable = Property(doc="Universe")
#     station: Callable = Property(doc="Sensor location")
#     translation_offset: StateVector = Property(
#         default=None,
#         doc="A 3x1 array specifying the Cartesian origin offset in terms of :math:`x,y,z` "
#             "coordinates.")
#
#     def __init__(self, *args, **kwargs):
#         """
#         Ensure that the translation offset is initiated
#         """
#         super().__init__(*args, **kwargs)
#         # Set values to defaults if not provided
#         if self.translation_offset is None:
#             self.translation_offset = StateVector([0] * 3)
#
#     @property
#     def ndim_meas(self) -> int:
#         """ndim_meas getter method
#
#         Returns
#         -------
#         :class:`int`
#             The number of measurement dimensions
#         """
#
#         return 3
#
#     def function(self, state, noise=False, **kwargs) -> StateVector:
#
#         if isinstance(noise, bool) or noise is None:
#             if noise:
#                 noise = self.rvs()
#             else:
#                 noise = 0
#
#         # Account for origin offset
#         xyz = state.state_vector[self.mapping, :] - self.translation_offset
#
#         # Rotate coordinates
#         xyz_rot = self.rotation_matrix @ xyz
#
#         # Convert to Spherical
#         rho, phi, theta = cart2sphere(xyz_rot[0, :], xyz_rot[1, :], xyz_rot[2, :])
#         elevations = [Elevation(i) for i in theta]
#         bearings = [Bearing(i) for i in phi]
#
#         # rho_godot = distance(state, self.station, self.uni, statevectors=isinstance(state.state_vector, StateVectors))
#         # print(rho_basic-rho) <- to verify that they return the same value
#
#         return StateVectors([elevations, bearings, rho]) + noise
#
#     def inverse_function(self, detection, **kwargs) -> StateVector:
#
#         theta, phi, rho = detection.state_vector
#         xyz = StateVector(sphere2cart(rho, phi, theta))
#
#         inv_rotation_matrix = inv(self.rotation_matrix)
#         xyz = inv_rotation_matrix @ xyz
#
#         res = np.zeros((self.ndim_state, 1)).view(StateVector)
#         res[self.mapping, :] = xyz + self.translation_offset
#
#         return res
#
#     def rvs(self, num_samples=1, **kwargs) -> Union[StateVector, StateVectors]:
#         out = super().rvs(num_samples, **kwargs)
#         out = np.array([[Elevation(0.)], [Bearing(0.)], [0.]]) + out
#         return out
