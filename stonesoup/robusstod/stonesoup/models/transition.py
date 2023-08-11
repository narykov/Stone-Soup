import numpy as np
from scipy.linalg import expm
import torch
from collections.abc import Callable

from ....types.array import StateVector
from ....models.transition.nonlinear import GaussianTransitionModel
from ....models.base import TimeVariantModel
from ....base import Property
from ....types.array import CovarianceMatrix


class LinearisedDiscretisation(GaussianTransitionModel, TimeVariantModel):
    """ We follow the approach of linearised discretization to handle nonlinear continuous-time dynamic models.
    """

    linear_noise_coeffs: np.ndarray = Property(
        doc=r"The acceleration noise diffusion coefficients :math:`[q_x, \: q_y, \: q_z]^T`")
    diff_equation: Callable = Property(doc=r"Differential equation describing the force model")

    @property
    def ndim_state(self):
        """ndim_state getter method

        Returns
        -------
        : :class:`int`
            The number of combined model state dimensions.
        """
        return 6

    def _get_jacobian(self, f, x):
        state = x.state_vector
        nx = len(state)
        A = np.zeros([nx, nx])
        state_input = [i for i in state]

        jacrows = torch.autograd.functional.jacobian(f, torch.tensor(state_input))
        for i, r in enumerate(jacrows):
            A[i] = r

        return (A)

    def _do_linearise(self, da, x, dt):
        dA = self._get_jacobian(da, x)  # state here is GroundTruthState
        nx = len(x.state_vector)
        # Get \int e^{dA*s}\,ds
        int_eA = expm(dt * np.block([[dA, np.identity(nx)], [np.zeros([nx, 2 * nx])]]))[:nx, nx:]

        # Get new value of x
        x = [i for i in x.state_vector]
        newx = x + int_eA @ da(torch.tensor(x))

        return newx

    def jacobian(self, state, **kwargs):
        da = self.diff_equation
        dA = self._get_jacobian(da, state)  # state here is GroundTruthState
        dt = kwargs['time_interval'].total_seconds()
        A = expm(dA * dt)

        return A

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
        dA = self._get_jacobian(da, x)  # state here is GroundTruthState

        # Get Q
        q_xdot, q_ydot, q_zdot = self.linear_noise_coeffs
        dQ = np.diag([0., q_xdot, 0., q_ydot, 0., q_zdot])

        nx = len(x.state_vector)
        dt = time_interval.total_seconds()

        G = expm(dt * np.block([[-dA, dQ], [np.zeros([nx, nx]), np.transpose(dA)]]))
        Q = np.transpose(G[nx:, nx:]) @ (G[:nx, nx:])
        Q = (Q + np.transpose(Q)) / 2.

        return CovarianceMatrix(Q)
