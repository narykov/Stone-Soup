import numpy as np
from scipy.linalg import expm
import torch
from collections.abc import Callable
from functools import lru_cache

from ....types.array import StateVector
from ....models.transition.nonlinear import GaussianTransitionModel
from ....models.transition.linear import LinearGaussianTransitionModel
from ....models.base import TimeVariantModel
from ....base import Property
from ....types.array import CovarianceMatrix
from ....types.array import CovarianceMatrix, StateVector, Matrix

from ....robusstod.physics.godot import jacobian_godot


class LinearisedDiscretisation(GaussianTransitionModel, TimeVariantModel):
    """ We follow the approach of linearised discretization to handle nonlinear continuous-time dynamic models.
    """

    linear_noise_coeffs: np.ndarray = Property(
        doc=r"The acceleration noise diffusion coefficients :math:`[q_x, \: q_y, \: q_z]^T`")
    diff_equation: Callable = Property(doc=r"Differential equation describing the force model")
    jacobian_godot: Callable = Property(default=None, doc=r"Whether GODOT is used")

    @property
    def ndim_state(self):
        """ndim_state getter method

        Returns
        -------
        : :class:`int`
            The number of combined model state dimensions.
        """
        return 6

    @lru_cache()
    def _get_jacobian(self, f, x, **kwargs):
        """This is how we linearise the nonlinear dynamic model"""
        timestamp = x.timestamp
        state = x.state_vector
        if self.jacobian_godot is None:
            """ Torch Jacobian"""
            da = lambda a: f(a, timestamp=timestamp)
            nx = len(state)
            A = np.zeros([nx, nx])
            state_input = [i for i in state]

            jacrows = torch.autograd.functional.jacobian(da, torch.tensor(state_input))
            for i, r in enumerate(jacrows):
                A[i] = r
        else:
            """ GODOT Jacobian"""
            A = self.jacobian_godot(state, timestamp=timestamp)
        return (A)

    def _do_linearise(self, da, x, dt):
        timestamp = x.timestamp
        dA = self._get_jacobian(da, x, timestamp=timestamp)  # state here is GroundTruthState
        nx = len(x.state_vector)
        # Get \int e^{dA*s}\,ds
        int_eA = expm(dt * np.block([[dA, np.identity(nx)], [np.zeros([nx, 2 * nx])]]))[:nx, nx:]

        # Get new value of x
        x = [i for i in x.state_vector]
        newx = x + int_eA @ da(torch.tensor(x))

        return newx

    def jacobian(self, state, **kwargs):
        """Here it represents the transition matrix F in discrete time, as if nonlinear
        dynamics model gets linearised. I didn't use 'transition_matrix' method of Linear model, as it doesn't take
        initial state as an input."""
        da = self.diff_equation
        timestamp = state.timestamp
        dA = self._get_jacobian(da, state, timestamp=timestamp)  # it is A in the Paul's note
        dt = kwargs['time_interval'].total_seconds()
        A_d = expm(dA * dt)

        return A_d

    def function(self, state, noise=False, **kwargs) -> StateVector:
        da = self.diff_equation
        dt = kwargs['time_interval'].total_seconds()
        sv2 = self._do_linearise(da, state, dt)
        new_state = StateVector(sv2)

        if isinstance(noise, bool) or noise is None:
            if noise:
                noise = self.rvs(prior=state, **kwargs)
            else:
                noise = 0

        return new_state + noise


    def covar(self, time_interval, **kwargs):
        da = self.diff_equation
        x = kwargs['prior']
        timestamp = x.timestamp
        dA = self._get_jacobian(da, x, timestamp=timestamp)  # state here is GroundTruthState

        # Get Q
        q_xdot, q_ydot, q_zdot = self.linear_noise_coeffs
        dQ = np.diag([0., q_xdot, 0., q_ydot, 0., q_zdot])

        nx = len(x.state_vector)
        dt = time_interval.total_seconds()

        G = expm(dt * np.block([[-dA, dQ], [np.zeros([nx, nx]), np.transpose(dA)]]))
        Q = np.transpose(G[nx:, nx:]) @ (G[:nx, nx:])
        Q = (Q + np.transpose(Q)) / 2.

        return CovarianceMatrix(Q)


class LinearTransitionModel(LinearGaussianTransitionModel, TimeVariantModel):
    transition_matrix: Matrix = Property(doc="Measurement matrix")
    bias_value: StateVector = Property(doc="Bias value")
    noise_covar: CovarianceMatrix = Property(doc="Noise covariance")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.transition_matrix, Matrix):
            self.transition_matrix = Matrix(self.transition_matrix)
        if not isinstance(self.bias_value, StateVector):
            self.bias_value = StateVector(self.bias_value)
        if not isinstance(self.noise_covar, CovarianceMatrix):
            self.noise_covar = CovarianceMatrix(self.noise_covar)

    # @property
    # def ndim_meas(self):
    #     """ndim_meas getter method
    #
    #     Returns
    #     -------
    #     :class:`int`
    #         The number of measurement dimensions
    #     """
    #
    #     return len(self.mapping)

    @property
    def ndim_state(self):
        """ndim_state getter method

        Returns
        -------
        : :class:`int`
            The number of model state dimensions.
        """

        return self.matrix().shape[0]

    def matrix(self, **kwargs): return self.transition_matrix

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