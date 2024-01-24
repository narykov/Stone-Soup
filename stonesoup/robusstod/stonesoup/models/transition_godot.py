import numpy as np
from pathlib import Path
from typing import Sequence

from datetime import timedelta
from stonesoup.base import Property
from stonesoup.models.transition.nonlinear import GaussianTransitionModel
from stonesoup.models.base import TimeVariantModel
from stonesoup.types.array import CovarianceMatrix, StateVector

from godot import cosmos
from godot.core import tempo
from stonesoup.robusstod.utils_gmv import Config

class GaussianTransitionGODOT(GaussianTransitionModel, TimeVariantModel):
    universe_path: Path = Property(doc="Path to file to be opened. Str will be converted to path.")
    trajectory_path: Path = Property(doc="Path to file to be opened. Str will be converted to path.")
    noise_diff_coeff: float = Property(
        doc="The Nth derivative noise diffusion coefficient (Variance) :math:`q`")
    mapping: Sequence[int] = Property(
        default=(0, 2, 4),
        doc="Mapping between measurement and state dims"
    )
    mapping_velocity: Sequence[int] = Property(
        default=(1, 3, 5),
        doc="Mapping between measurement and state dims"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.universe_path, Path):
            self.universe_path = Path(self.universe_path)  # Ensure Path
        if not isinstance(self.trajectory_path, Path):
            self.trajectory_path = Path(self.trajectory_path)  # Ensure Path
        # we assemble a configuration that will be repeatedly re-configured when performing individual propagations
        self.config = Config(self.universe_path, self.trajectory_path)
        self.overshoot_time = timedelta(seconds=1)  # this is to ensure that the propagated point is within the scope of propagation
        self.strf_tai = "%Y-%m-%dT%H:%M:%S TAI"
        self.strf_tdb = "%Y-%m-%dT%H:%M:%S TDB"
        self.mapping_godot = (0, 1, 2)
        self.mapping_velocity_godot = (3, 4, 5)


    @property
    def ndim_state(self):
        """ndim_state getter method

        Returns
        -------
        : :class:`int`
            The number of combined model state dimensions.
        """
        return 6

    def godot_to_stonesoup(self, state):
        out = np.empty_like(state)
        out[[self.mapping]] = state[[self.mapping_godot]]  # position
        out[[self.mapping_velocity]] = state[[self.mapping_velocity_godot]]  # velocity
        return out * 1000

    def stonesoup_to_godot(self, state):
        out = np.empty_like(state)
        out[[self.mapping_godot]] = state[[self.mapping]]  # position
        out[[self.mapping_velocity_godot]] = state[[self.mapping_velocity]]  # velocity
        return out / 1000

    def datetime_to_epoch(self, timestamp, epoch=True, timescale='TDB'):
        formatting = self.strf_tdb if timescale is 'TDB' else self.strf_tai
        t = timestamp.strftime(formatting)
        return tempo.Epoch(t) if epoch else t

    def function(self, state, noise=False, **kwargs) -> StateVector:
        # time_interval_sec = kwargs['time_interval'].total_seconds()
        sv1 = state.state_vector  # state in cartesian coordiantes
        sv1_godot = self.stonesoup_to_godot(sv1)

        timestamp_current = state.timestamp
        time_interval = kwargs['time_interval']
        godot_times = {
            'current': self.datetime_to_epoch(timestamp_current),
            'prediction': self.datetime_to_epoch(timestamp_current + time_interval),
            'overshoot': self.datetime_to_epoch(timestamp_current + time_interval + self.overshoot_time)
        }
        # initialise the universe with a configuration in universe.yml
        universe = cosmos.Universe(self.config.universe)
        # re-configure the trajectory.yml entries to fit the current propagation needs
        self.config.set_epoch(godot_times['current'])  # use the current time stamp to specify the epoch in trajectory.yml
        self.config.set_epoch_future(godot_times['overshoot'])  # specify the future epoch in trajectory.yml
        self.config.set_state_cart(sv1_godot)  # use the current state to specify the control point in trajectory.yml
        trajectory = cosmos.Trajectory(universe, self.config.trajectory)
        trajectory.compute(partials=True)

        sv2_godot = universe.frames.vector6('Earth', 'SC_center', 'ICRF', godot_times['overshoot'])
        sv2 = StateVector(self.godot_to_stonesoup(sv2_godot))

        if isinstance(noise, bool) or noise is None:
            if noise:
                noise = self.rvs(num_samples=state.state_vector.shape[1], **kwargs)
            else:
                noise = 0
        return sv2 + noise

    def covar(self, time_interval, **kwargs):
        time_interval_sec = time_interval.total_seconds()
        dt = time_interval_sec
        covar = np.array([[dt**3 / 3, dt**2 / 2],
                          [dt**2 / 2, dt]])
        covar = np.kron(np.eye(3), covar)
        covar *= self.noise_diff_coeff
        return CovarianceMatrix(covar)




#
# def plot_truth(truth, ax=None, color='r', label='Tracks', mapping=None):
#     if mapping is None:
#         mapping = [0, 2]
#
#     if not ax:
#         ax = plt.gca()
#     ax.plot([], [], '-', color=color, label=label)
#     x = np.array([state.state_vector[0] for state in truth])
#     y = np.array([state.state_vector[2] for state in truth])
#     z = np.array([state.state_vector[4] for state in truth])
#     data = np.array([state.state_vector for state in truth])
#     ax.plot(x, y, '-', marker='.', color=color)
#
#
# def main():
#     noise_diff_coeff = 0.05
#     universe_path = 'universe.yml'
#     trajectory_path = 'trajectory.yml'
#     transition_model = GaussianTransitionGODOT(
#         universe_path=universe_path,
#         trajectory_path=trajectory_path,
#         noise_diff_coeff=noise_diff_coeff
#     )
#     # we take initial state_vector [X,Y,Z,VX,VY,VZ] from an old GMV script
#     cart_godot = np.array([-4685.75946803037,
#                            3965.81226070460,
#                            3721.45235707666,
#                            -2.07276827105304,
#                            3.45342193298209,
#                            -6.27128141230935])
#     cart = godot_to_stonesoup(cart_godot)
#
#     timesteps = [tempo.Epoch('2013-01-01T00:00:00 TAI')]
#     truth = GroundTruthPath([GroundTruthState(cart, timestamp=timesteps[0])])
#
#     num_steps = 10
#     time_interval = 10
#     for k in range(1, num_steps + 1):
#         timesteps.append(timesteps[k-1]+time_interval)  # add next timestep to list of timesteps
#         propagated_state = transition_model.function(truth[k-1], noise=False, time_interval=timedelta(seconds=time_interval))
#         truth.append(GroundTruthState(
#             propagated_state,
#             timestamp=timesteps[k]))
#
#     plot_truth(truth)
#     plt.plot(truth[0].state_vector[0], truth[0].state_vector[2], 'x')
#
#     print()
#
#
# if __name__ == "__main__":
#     main()
