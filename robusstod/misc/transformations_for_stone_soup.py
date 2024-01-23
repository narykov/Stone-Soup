import copy
import csv
import numpy as np

from stonesoup.base import Property
from stonesoup.buffered_generator import BufferedGenerator
from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRangeRate
from stonesoup.models.measurement.base import MeasurementModel
from stonesoup.reader.generic import _CSVReader, DetectionReader
from stonesoup.types.angle import Bearing, Elevation
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.detection import Detection
from stonesoup.types.state import State

import matplotlib.pyplot as plt


class CustomDetectionReader(DetectionReader, _CSVReader):
    """A detection reader for csv files of ground-based radar detections.

    Parameters
    ----------
    """
    filters: dict = Property(default=None, doc='Entry')
    max_datapoints: int = Property(default=None, doc='N datapoints to consider')
    measurement_model: MeasurementModel = Property(default=None, doc='Entry')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_fields = ['TRUE_X', 'TRUE_VX', 'TRUE_Y', 'TRUE_VY', 'TRUE_Z', 'TRUE_VZ']
        self.station_fields = ['XSTAT_X', 'XSTAT_VX', 'XSTAT_Y', 'XSTAT_VY', 'XSTAT_Z', 'XSTAT_VZ']
        self.km_meas_fields = ['RANGE', 'DOPPLER_INSTANTANEOUS']
        self.km_fields = self.target_fields + self.station_fields + self.km_meas_fields
        self.m_in_km = 1000
        self.bearing_name = 'ANGLE_1'
        self.elevation_name = 'ANGLE_2'

    @staticmethod
    def _to_standard_angle(angle):
        angle = np.pi / 2 - angle

        # Ensure the angle is within the range [0, 360)
        if angle >= 2 * np.pi:
            angle -= 2 * np.pi
        elif angle < 0:
            angle += 2 * np.pi

        return angle

    @staticmethod
    def _get_translation_offset(station_ecef, mapping):
        return station_ecef[mapping, :]

    @staticmethod
    def _get_velocity(station_ecef, velocity_mapping):
        return station_ecef[velocity_mapping, :]

    @staticmethod
    def _get_rotation_offset(station_ecef, mapping):
        # https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates
        station_location = station_ecef[mapping, :]
        x, y, z = station_location
        r = np.linalg.norm(station_location)
        lon = np.arcsin(z / r)  # also lambda
        lat = np.arctan2(y, x)
        roll, pitch, yaw = (0.5 * np.pi - lon, 0, 0.5 * np.pi + lat)
        return StateVector([roll, pitch, yaw])

    def _get_measurement_adjusted(self, row):
        state_vector = []
        for col_name in self.state_vector_fields:
            value = float(row[col_name])  # csv containst strings
            value = self.m_in_km * value if col_name in self.km_fields else value  # turn into m if in km
            value = Bearing(self._to_standard_angle(value)) if col_name == self.bearing_name else value
            value = Elevation(value) if col_name == self.elevation_name else value
            state_vector.append([value])
        return StateVector(state_vector)

    def _get_measurement_model(self, row):
        measurement_model = copy.deepcopy(self.measurement_model)
        station_ecef = StateVector([self.m_in_km * float(row[key]) for key in self.station_fields])
        measurement_model.translation_offset = self._get_translation_offset(station_ecef, measurement_model.mapping)
        measurement_model.rotation_offset = self._get_rotation_offset(station_ecef, measurement_model.mapping)
        measurement_model.velocity = self._get_velocity(station_ecef, measurement_model.velocity_mapping)
        return measurement_model

    def _get_ss_measurement(self, row, measurement_model):
        target_state = []
        for col_name in self.target_fields:
            coef = self.m_in_km if col_name in self.km_fields else 1
            value = coef * float(row[col_name])
            if col_name == self.bearing_name:
                value = Bearing(self._to_standard_angle(value))
            if col_name == self.elevation_name:
                value = Elevation(value)
            target_state.append([value])

        return measurement_model.function(State(state_vector=StateVector(target_state)))


    @BufferedGenerator.generator_method
    def detections_gen(self, from_ground_truth=False):
        n_datapoints = 0
        with self.path.open(encoding=self.encoding, newline='') as csv_file:

            for row in csv.DictReader(csv_file, **self.csv_options):

                n_datapoints += 1

                if self.max_datapoints is not None:
                    if n_datapoints > self.max_datapoints:
                        break

                measurement_model = self._get_measurement_model(row)
                if from_ground_truth:
                    measurement = self._get_ss_measurement(row, measurement_model)
                else:
                    measurement = self._get_measurement_adjusted(row)

                detection = Detection(state_vector=measurement,
                                      timestamp=self._get_time(row),
                                      measurement_model=self._get_measurement_model(row),
                                      metadata=self._get_metadata(row))

                yield detection

def main():

    ndim_state = 6
    mapping_location = (0, 2, 4)  # encodes location indices in the state vector
    mapping_velocity = (1, 3, 5)  # encodes velocity indices in the state vector


    # Specify sensor parameters and generate a history of measurements for the time steps (parameters for RR01)
    sigma_r = 20  # Range: 20.0 m
    sigma_a = np.deg2rad(400*0.001)  # Azimuth - elevation: 400.0 mdeg
    sigma_rr = 650.0 * 0.001  # Range-rate: 650.0 mm/s
    sigma_el, sigma_b, sigma_range, sigma_range_rate = sigma_a, sigma_a, sigma_r, sigma_rr
    noise_covar = CovarianceMatrix(np.diag([sigma_el ** 2, sigma_b ** 2, sigma_range ** 2, sigma_range_rate ** 2]))
    sensor_parameters = {
        'ndim_state': ndim_state,
        'mapping': mapping_location,
        'noise_covar': noise_covar
    }

    measurement_model = CartesianToElevationBearingRangeRate(
        ndim_state=sensor_parameters['ndim_state'],
        mapping=sensor_parameters['mapping'],
        velocity_mapping=mapping_velocity,
        noise_covar=sensor_parameters['noise_covar']
    )
    path = "measurements.csv"
    measurement_fields = ("ANGLE_2", "ANGLE_1", "RANGE", "DOPPLER_INSTANTANEOUS")

    detector = CustomDetectionReader(
        path=path,
        state_vector_fields=measurement_fields,
        time_field="TIME",
        measurement_model=measurement_model
    )

    measurements = []
    for measurement in detector.detections_gen():
        measurements.append(measurement)
    # TODO: not sure how this works, i.e., that the generator does not need to be restarted
    measurements_ss = []
    for measurement in detector.detections_gen(from_ground_truth=True):
        measurements_ss.append(measurement)

    fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1)
    ax0.set_ylabel('El, [rad]')
    ax0.get_xaxis().set_ticks([])
    ax1.set_ylabel('Az, [rad]')
    ax1.get_xaxis().set_ticks([])
    ax2.set_ylabel('R, [m]')
    ax2.get_xaxis().set_ticks([])
    ax3.set_ylabel('RR, [m/s]')
    ax3.set_xlabel('Time')

    for label in ['dataset values', 'Stone Soup model']:
        if label == 'dataset values':
            data = measurements
            plot_param = {'color': 'b', 'marker': 'x'}
        else:
            data = measurements_ss
            plot_param = {'color': 'r', 'marker': '.'}

        timestamps = [measurement.timestamp for measurement in data]
        ax0.plot(timestamps, [measurement.state_vector[0] for measurement in data], label=label, **plot_param)
        ax1.plot(timestamps, [measurement.state_vector[1] for measurement in data], label=label, **plot_param)
        ax2.plot(timestamps, [measurement.state_vector[2] for measurement in data], label=label, **plot_param)
        ax3.plot(timestamps, [measurement.state_vector[3] for measurement in data], label=label, **plot_param)

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
