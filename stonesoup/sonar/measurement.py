import numpy as np
from scipy.optimize import fsolve

# StoneSoup imports
from ..base import Property
from ..models.base import ReversibleModel
from ..models.measurement.nonlinear import NonLinearGaussianMeasurement
from ..types.angle import Bearing
from ..types.array import StateVectors
from ..types.state import StateVector

# Custom imports
from .type import Direction

class ConicalAngle(NonLinearGaussianMeasurement):
    """
    Towed array sonar measurement model in Cartesian coordinates.
    It inherits from CartesianToElevationBearing to reuse their `rotation_offset` and `translation_offset`.
    """

    translation_offset: StateVector = Property(default=None, doc="Centre of the sensor")
    orientation: StateVector = Property(default=None, doc="Directional vector of the sensor, e.g., array's orientation")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.translation_offset is None:
            self.translation_offset = StateVector([0] * len(self.mapping))

        if self.orientation is None:
            self.orientation = StateVector([0] * len(self.mapping))
            self.orientation[0] = 1  # Default to pointing forward in x

        self.orientation = self.orientation/np.linalg.norm(self.orientation)

    @staticmethod
    def sv2orientation(state_vector, velocity_mapping):
        """A method to extract orientation from the velocity values in a sensor state vector"""
        centre_velocity = state_vector[[velocity_mapping]]
        centre_speed = np.linalg.norm(centre_velocity)
        return centre_velocity / centre_speed

    @property
    def ndim_meas(self) -> int:
        """ndim_meas getter method

        Returns
        -------
        :class:`int`
            The number of measurement dimensions
        """

        return 1

    def _get_direction(self, state_vector, centre, orientation):

        diff = (state_vector - centre).flatten()
        d_vector = diff / np.linalg.norm(diff)  # unit vector from centre to target
        v_vector = orientation.flatten()  # unit vector of the sensor orientation
        cosine = np.clip(np.dot(d_vector, v_vector), -1.0, 1.0)  # angle between two vectors
        direction = np.arccos(cosine)

        return Direction(direction)

    def _get_measurement(self, state_vector):
        return self._get_direction(state_vector, self.translation_offset, self.orientation)

    def function(self, state, noise=False, **kwargs) -> StateVector:
        r"""Model function :math:`h(\vec{x}_t,\vec{v}_t)`

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
        :class:`numpy.ndarray` of shape (:py:attr:`~ndim_state`, 1)
            The model function evaluated given the provided time interval.
        """

        if isinstance(noise, bool) or noise is None:
            if noise:
                noise = self.rvs(num_samples=state.state_vector.shape[1], **kwargs)
            else:
                noise = 0

        measurements = [[] for _ in range(self.ndim_meas)]

        state_vectors = [state.state_vector] if state.state_vector.shape[1] == 1 else state.state_vector

        for state_vector in state_vectors:
            measurement = self._get_measurement(state_vector)

            if not isinstance(measurement, (list, tuple)):
                measurement = [measurement]

            for i, value in enumerate(measurement):
                measurements[i].append(value)

        return StateVectors(measurements) + noise


class MonostaticConicalAngleDelay(ConicalAngle):

    wave_propagation_speed: float = Property(default=1500, doc="Speed of waves in medium")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def ndim_meas(self) -> int:
        """ndim_meas getter method

        Returns
        -------
        :class:`int`
            The number of measurement dimensions
        """

        return 2

    def _get_time_delay(self, state_vector, centre):
        return np.linalg.norm(state_vector - centre) / self.wave_propagation_speed

    def _get_measurement(self, state_vector, one_way_delay=True):
        pos_vector = state_vector[[self.mapping]]
        theta = self._get_direction(pos_vector, self.translation_offset, self.orientation)
        n = 1 if one_way_delay else 2
        tof = n * self._get_time_delay(pos_vector, self.translation_offset)  # One-way value
        return theta, tof

class BistaticConicalAnglesDelay(MonostaticConicalAngleDelay):
    """Returns Rx/Tx cosines and time delay 3D"""
    translation_offset_origin: StateVector = Property(default=None, doc="Centre of the sensor")
    orientation_origin: StateVector = Property(default=None, doc="Directional vector of the sensor, e.g., array's orientation")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.translation_offset_origin is None:
            self.translation_offset_origin = StateVector([0] * len(self.mapping))

        assert len(self.translation_offset) == len(self.translation_offset_origin)

        if self.orientation_origin is None:
            self.orientation_origin = StateVector([0] * len(self.mapping))
            self.orientation_origin[-1] = -1  # Default to pointing downwards
        self.orientation_origin = self.orientation_origin / np.linalg.norm(self.orientation_origin)

    @property
    def ndim_meas(self) -> int:
        """ndim_meas getter method

        Returns
        -------
        :class:`int`
            The number of measurement dimensions
        """

        return 3

    def _get_measurement(self, state_vector):
        pos_vector = StateVector(state_vector[[self.mapping]].flatten())
        theta_origin = self._get_direction(pos_vector, self.translation_offset_origin, self.orientation_origin)
        theta = self._get_direction(pos_vector, self.translation_offset, self.orientation)
        tof = (
            self._get_time_delay(self.translation_offset_origin, pos_vector) +
            self._get_time_delay(pos_vector, self.translation_offset)
        )
        return theta_origin, theta, tof


class BistaticConicalAnglesDelayAzimuth(BistaticConicalAnglesDelay, ReversibleModel):
    """Assumes triplets"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def ndim_meas(self) -> int:
        """ndim_meas getter method

        Returns
        -------
        :class:`int`
            The number of measurement dimensions
        """

        return 4


    def _get_azimuth(self, state_vector):

        # Compute vector from array centre to target
        centre_to_target = state_vector - self.translation_offset

        # Project/normalise both vectors (array orientation and vector to target) onto the horizontal (x-y) plane
        orientation_xy = np.array([self.orientation[0], self.orientation[1], 0])
        orientation_xy = orientation_xy / np.linalg.norm(orientation_xy)

        vector_to_target_xy = np.array([centre_to_target[0], centre_to_target[1], 0])
        vector_to_target_xy = vector_to_target_xy / np.linalg.norm(vector_to_target_xy)

        # Compute the azimuth using the dot product
        dot_product = np.clip(np.dot(orientation_xy, vector_to_target_xy), -1.0, 1.0)

        # Compute the azimuth angle in radians
        azimuth = np.arccos(dot_product)

        # Determine the sign of the azimuth (left or right side) using the cross product (to get direction)
        cross_product = np.cross(orientation_xy, vector_to_target_xy)
        if cross_product[2] < 0:
            azimuth = -azimuth  # Negative azimuth means target is to the right of the array orientation

        return Bearing(azimuth)

    def _get_measurement(self, state_vector):
        pos_vector = StateVector(state_vector[[self.mapping]].flatten())
        theta_origin = self._get_direction(pos_vector, self.translation_offset_origin, self.orientation_origin)
        theta = self._get_direction(pos_vector, self.translation_offset, self.orientation)
        tof = (
            self._get_time_delay(self.translation_offset_origin, pos_vector) +
            self._get_time_delay(pos_vector, self.translation_offset)
        )
        az = self._get_azimuth(pos_vector)
        return theta_origin, theta, tof, az

    def inverse_function(self, detection, **kwargs) -> StateVector:

        def find_point_from_azimuth_range(azimuth, range):
            orientation = self.orientation.flatten()

            # Project orientation onto the x-y plane (ignore z-component)
            orientation_xy = np.array([orientation[0], orientation[1], 0])

            # If orientation is exactly vertical, we would handle that, but assuming horizontal orientation here
            orientation_xy = orientation_xy / np.linalg.norm(orientation_xy)  # Normalize projected orientation

            # Now, define the local x-y plane for azimuth calculations.
            # Azimuth is applied in the x-y plane; target is at same depth (z remains unchanged)

            # Calculate local x-y displacement based on azimuth and range
            dx = np.cos(azimuth) * range  # x displacement (in local frame)
            dy = np.sin(azimuth) * range  # y displacement (in local frame)

            # Compute the displacement in the global x-y plane
            # Rotate the displacement based on the orientation of the array
            displacement = np.array([dx * orientation_xy[0] - dy * orientation_xy[1],
                                     dx * orientation_xy[1] + dy * orientation_xy[0],
                                     0])  # z remains 0 because we are in the horizontal plane

            displacement = StateVector(displacement)

            # Add the displacement to the centre of the array
            target_location = self.translation_offset + displacement

            return target_location

        def equations(state_vector, detection):
            sv = StateVector(state_vector)

            cone_angle_transmitter = detection.state_vector[0]
            cone_angle_receiver = detection.state_vector[1]
            total_delay = detection.state_vector[2]

            # Vector from transmitter to target
            v_T_target = (sv - self.translation_offset_origin).flatten()
            d_T_target = np.linalg.norm(v_T_target)

            # Vector from receiver to target
            v_R_target = (sv - self.translation_offset).flatten()
            d_R_target = np.linalg.norm(v_R_target)

            # Constraint 1: Cone angle from transmitter
            orientation_origin = self.orientation_origin.flatten()
            cone_T_constraint = np.cos(cone_angle_transmitter) - np.dot(v_T_target, orientation_origin) / d_T_target


            # Constraint 2: Cone angle from receiver
            orientation = self.orientation.flatten()
            cone_R_constraint = np.cos(cone_angle_receiver) - np.dot(v_R_target.flatten(), orientation) / d_R_target

            # Constraint 3: Total travel distance
            distance_constraint = (d_T_target + d_R_target) - total_delay * self.wave_propagation_speed

            return [cone_T_constraint, cone_R_constraint, distance_constraint]

        azimuth = detection.state_vector[3]
        total_delay = detection.state_vector[2]
        travel_distance = total_delay * self.wave_propagation_speed
        range_guess = travel_distance / 2
        initial_guess = StateVector(find_point_from_azimuth_range(azimuth, range_guess))

        result = fsolve(equations, initial_guess, args=(detection,))

        output = np.zeros(self.ndim_state)
        output[::2] = result

        return StateVector(output)
