#!/usr/bin/env python

"""
Tracking a single orbiting object with no detection failures (no false alarms and missed detections)
========================================
This is a demonstration using the implemented IEKF/IPLF/IPLS algorithms in the context of space situation awareness.
It can use either built-in model of acceleration or GODOT's capability to evaluate acceleration.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRange, CartesianToElevationBearingRangeRate
from stonesoup.models.measurement.base import MeasurementModel
from stonesoup.plotter import Plotterly
from stonesoup.types.array import CovarianceMatrix
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.detection import TrueDetection
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.prediction import GaussianState, GaussianStatePrediction
from stonesoup.types.track import Track
from stonesoup.updater.kalman import IteratedKalmanUpdater

# ROBUSSTOD MODULES
from stonesoup.robusstod.stonesoup.models.transition import LinearisedDiscretisation
from stonesoup.robusstod.stonesoup.predictor import ExtendedKalmanPredictor, UnscentedKalmanPredictor
from stonesoup.robusstod.stonesoup.smoother import IPLSKalmanSmoother
from stonesoup.robusstod.stonesoup.updater import IPLFKalmanUpdater
from stonesoup.robusstod.physics.constants import G, M_earth
from stonesoup.robusstod.physics.other import get_noise_coefficients

# for csv reading
import csv
from stonesoup.types.detection import Detection
from stonesoup.types.array import StateVector
from stonesoup.buffered_generator import BufferedGenerator
from stonesoup.reader.generic import _CSVReader, DetectionReader
from stonesoup.base import Property
from stonesoup.types.state import State

use_godot = False
if use_godot:
    try:
        import godot
    except ModuleNotFoundError as e:
        print(e.msg)
        sys.exit(1)  # the exit code of 1 is a convention that means something went wrong
    from stonesoup.robusstod.physics.godot import KeplerianToCartesian, diff_equation
    fig_title = ' with GODOT physics'
else:
    from stonesoup.robusstod.physics.basic import KeplerianToCartesian, diff_equation
    fig_title = ' with basic physics'


def do_single_target_tracking(prior=None, predictor=None, updater=None, measurements=None):
    if measurements is None:
        measurements = []

    track = Track([prior])
    for measurement in measurements:
        prediction = predictor.predict(prior, timestamp=measurement.timestamp)
        hypothesis = SingleHypothesis(prediction, measurement)  # Group a prediction and measurement
        post = updater.update(hypothesis)
        track.append(post)
        prior = track[-1]

    return track

def do_stt(prior=None, predictor=None, updater=None, detector=None):

    track = Track([prior])
    for measurement in detector.detections_gen():
        prediction = predictor.predict(prior, timestamp=measurement.timestamp)
        hypothesis = SingleHypothesis(prediction, measurement)  # Group a prediction and measurement
        post = updater.update(hypothesis)
        track.append(post)
        prior = track[-1]

    return track


# class CSVDetectionReader(DetectionReader, _CSVReader):
#     """A detection reader for csv files of GMV detections.
#
#     CSV file must have headers, as these are used to determine which fields to use to generate
#     the detection. Detections at the same time are yielded together, and such assume file is in
#     time order.
#
#     Parameters
#     ----------
#     """
#
#     def _unit_conversion(self, row):
#         state_vector = []
#         for col_name in self.state_vector_fields:
#             value = float(row[col_name])
#             if col_name.startswith('ANGLE'):
#                 value = np.deg2rad(value)
#             if col_name in ['RANGE', 'DOPPLER_INSTANTANEOUS']:
#                 value = value * 1000
#             state_vector.append([value])
#
#         return state_vector
#
#     @BufferedGenerator.generator_method
#     def detections_gen(self):
#         with self.path.open(encoding=self.encoding, newline='') as csv_file:
#             detections = set()
#             previous_time = None
#             for row in csv.DictReader(csv_file, **self.csv_options):
#
#                 time = self._get_time(row)
#                 if previous_time is not None and previous_time != time:
#                     yield previous_time, detections
#                     detections = set()
#                 previous_time = time
#
#                 detections.add(Detection(
#                     np.array([[row[col_name]] for col_name in self.state_vector_fields],
#                              dtype=np.float_),
#                     timestamp=time,
#                     metadata=self._get_metadata(row)))
#
#             # Yield remaining
#             yield previous_time, detections


class CSVDetectionReader(DetectionReader, _CSVReader):
    """A simple detection reader for csv files of detections.

    CSV file must have headers, as these are used to determine which fields to use to generate
    the detection. Detections at the same time are yielded together, and such assume file is in
    time order.

    Parameters
    ----------
    """
    filters: dict = Property(default=None, doc='Entry')
    measurement_model: MeasurementModel = Property(default=None, doc='Entry')

    def _unit_conversion(self, row):
        state_vector = []
        for col_name in self.state_vector_fields:
            value = float(row[col_name])
            if col_name.startswith('ANGLE'):
                value = np.deg2rad(value)
            if col_name in ['RANGE', 'DOPPLER_INSTANTANEOUS']:
                value = value * 1000
            state_vector.append([value])

        return state_vector

    @BufferedGenerator.generator_method
    def detections_gen(self):
        with self.path.open(encoding=self.encoding, newline='') as csv_file:

            for row in csv.DictReader(csv_file, **self.csv_options):

                skip = False
                for key in self.filters:
                    if row[key] != self.filters[key]:
                        skip = True
                        continue

                if skip:
                    continue

                time = self._get_time(row)
                sv = self._unit_conversion(row)
                detection = Detection(state_vector=np.array(sv,
                                                            dtype=np.float_),
                                      timestamp=time,
                                      measurement_model=self.measurement_model,
                                      metadata=self._get_metadata(row))
                yield detection

    def get_initial_state(self):
        with self.path.open(encoding=self.encoding, newline='') as csv_file:
            for row in csv.DictReader(csv_file, **self.csv_options):

                skip = False
                for key in self.filters:
                    if row[key] != self.filters[key]:
                        skip = True
                        continue

                if skip:
                    continue

                time = self._get_time(row)
                sv = self._unit_conversion(row)
                detection = Detection(state_vector=np.array(sv, dtype=np.float_),
                                      timestamp=time,
                                      measurement_model=self.measurement_model,
                                      metadata=self._get_metadata(row))

                state_vector = self.measurement_model.inverse_function(detection)
                return State(state_vector=state_vector, timestamp=time)


def main():
    np.random.seed(1991)
    start_time = datetime(2000, 1, 1)
    time_parameters = {
        'n_time_steps': 50,
        'time_interval': timedelta(seconds=120)
    }
    # TODO: consider arbitrary time intervals to demonstrate the flexibility of the approach
    timesteps = [start_time + k * time_parameters['time_interval'] for k in range(time_parameters['n_time_steps'])]

    # We begin by specifying the true initial target state by picking a credible set of Keplerian elements, and
    # then converting them into Cartesian domain.

    a, e, i, w, omega, nu = (9164000, 0.03, 70, 0, 0, 0)
    # Orbital elements, see https://en.wikipedia.org/wiki/Orbital_elements
    # Two elements define the shape and size of the ellipse:
    # a = semimajor axis
    # e = eccentricity
    # Two elements define the orientation of the orbital plane in which the ellipse is embedded:
    # i = inclination
    # omega = longitude of the ascending node
    # The remaining two elements are as follows:
    # w = argument of periapsis
    # nu = true anomaly
    # Visualisation (png): https://en.wikipedia.org/wiki/Orbital_elements#/media/File:Orbit1.svg
    # A simple interactive visualisation can be accessed at https://orbitalmechanics.info,
    # where a is given in the multiples of the Earth's radius, and i, omega, w and nu are defined in degrees.
    #
    # a, e, i, w, omega, nu (m, _, deg, deg, deg, deg)
    # NB: a, e, I, RAAN, argP, ta (km, _, rad, rad, rad, rad) as in https://godot.io.esa.int/tutorials/T04_Astro/T04scv/
    K = np.array([a, e, np.radians(i), np.radians(w), np.radians(omega), np.radians(nu)])  # now in SI units (m & rad)
    ndim_state = 6
    mapping_location = (0, 2, 4)  # encodes location indices in the state vector
    mapping_velocity = (1, 3, 5)  # encodes velocity indices in the state vector
    GM = G * M_earth  # https://en.wikipedia.org/wiki/Standard_gravitational_parameter (m^3 s^−2)


    # Specify sensor parameters and generate a history of measurements for the time steps
    sigma_el, sigma_b, sigma_range, sigma_range_rate = np.deg2rad(0.01), np.deg2rad(0.01), 10000, 100
    # parameters for RR01
    sigma_r = 20  # Range: 20.0 m
    sigma_a = np.deg2rad(400*0.001)  # Azimuth - elevation: 400.0 mdeg
    sigma_rr = 650.0 * 0.001  # Range-rate: 650.0 mm/s
    sigma_el, sigma_b, sigma_range, sigma_range_rate = sigma_a, sigma_a, sigma_r, sigma_rr
    sensor_x, sensor_y, sensor_z = 0, 0, 0
    sensor_parameters = {
        'ndim_state': ndim_state,
        'mapping': mapping_location,
        'noise_covar': CovarianceMatrix(np.diag([sigma_el ** 2, sigma_b ** 2, sigma_range ** 2, sigma_range_rate ** 2])),
        'translation_offset': np.array([[sensor_x], [sensor_y], [sensor_z]])
    }

    # measurement_model = CartesianToElevationBearingRange(
    #     ndim_state=sensor_parameters['ndim_state'],
    #     mapping=sensor_parameters['mapping'],
    #     noise_covar=sensor_parameters['noise_covar'],
    #     translation_offset=sensor_parameters['translation_offset']
    # )
    measurement_model = CartesianToElevationBearingRangeRate(
        ndim_state=sensor_parameters['ndim_state'],
        mapping=sensor_parameters['mapping'],
        noise_covar=sensor_parameters['noise_covar'],
        translation_offset=sensor_parameters['translation_offset']
    )
    path = "src/csv/RODDAS_OD_00_015.csv"
    filter_value = 'RR02'
    # from stonesoup.reader.generic import CSVDetectionReader
    filters = {'STATION': 'RR02', 'TARGET_ID': '00039451'}
    detector = CSVDetectionReader(
        path=path,
        state_vector_fields=("ANGLE_1", "ANGLE_2", "RANGE", "DOPPLER_INSTANTANEOUS"),
        time_field="TIME",
        filters=filters,
        measurement_model=measurement_model
    )

    initial_state = detector.get_initial_state()

    # initial_state_vector = KeplerianToCartesian(K, GM, ndim_state, mapping_location, mapping_velocity)# into Cartesian
    initial_state = GroundTruthState(state_vector=initial_state.state_vector, timestamp=initial_state.timestamp)
    initial_covariance = CovarianceMatrix(np.diag([50000 ** 2, 100 ** 2, 50000 ** 2, 100 ** 2, 50000 ** 2, 100 ** 2]))

    deviation = np.linalg.cholesky(initial_covariance).T @ np.random.normal(size=initial_state.state_vector.shape)
    # deviation = np.array([-122068.01433784, 69.37315652, 507.76211348, -86.74038986, -58321.63970861, 89.04789997]).reshape((6, 1))
    # prior = GaussianState(state_vector=initial_state.state_vector + deviation,
    #                       covar=initial_covariance,
    #                       timestamp=start_time)

    prior = GaussianStatePrediction(state_vector=initial_state.state_vector + deviation,
                                    covar=initial_covariance,
                                    timestamp=start_time)

    transition_model = LinearisedDiscretisation(
        diff_equation=diff_equation,
        linear_noise_coeffs=get_noise_coefficients(GM)
    )
    # from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
    # q = 1
    # transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(q),
    #                                                           ConstantVelocity(q), ConstantVelocity(q)])

    # Generate ground truth trajectory, using the initial target state, target dynamics and the grid of timesteps
    truth = GroundTruthPath(initial_state)
    successive_time_steps = timesteps[1:]  # dropping the very first start_time
    for timestamp in successive_time_steps:
        truth.append(GroundTruthState(
            transition_model.function(state=truth[-1], noise=True, time_interval=time_parameters['time_interval']),
            timestamp=timestamp)
        )

    measurements = []
    for measurement in detector.detections_gen():
        measurements.append(measurement)


    # measurements = []
    # for state in truth[1:]:
    #     measurement = measurement_model.function(state=state, noise=True)
    #     timestamp = state.timestamp
    #     measurements.append(TrueDetection(
    #         state_vector=measurement,
    #         timestamp=timestamp,
    #         groundtruth_path=truth,
    #         measurement_model=measurement_model)
    #     )

    # Here we finally specify how the filtering recursion is implemented
    predictor_ekf = ExtendedKalmanPredictor(transition_model)
    predictor_ukf = UnscentedKalmanPredictor(transition_model)
    updater_iplf = IPLFKalmanUpdater(tolerance=1e-1, max_iterations=5)  # Using default values
    updater_iekf = IteratedKalmanUpdater(max_iterations=5)

    # Perform tracking/filtering/smooting
    # track_iplf = do_single_target_tracking(prior=prior, predictor=predictor_ukf, updater=updater_iplf, measurements=measurements)
    track_iekf = do_single_target_tracking(prior=prior, predictor=predictor_ekf, updater=updater_iekf, measurements=measurements)
    # # track_ipls = IPLSKalmanSmoother(transition_model=transition_model).smooth(track_iplf)
    # track_iplf = do_stt(prior=prior, predictor=predictor_ukf, updater=updater_iplf, detector=detector)
    # track_iekf = do_stt(prior=prior, predictor=predictor_ekf, updater=updater_iekf, detector=detector)


    # Plotting the results using Plotterly

    fig, ax = plt.subplots()
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.ylim((-2520788.3357520015, 2474705.9185321466))
    # plt.xlim((-2073231.8435078696, 2068652.27618956))

    # for measurement in measurements:
    #     state = measurement_model.inverse_function(measurement)
    #     plt.plot(state[0], state[2], color='k', marker='.')
    #     plt.pause(0.5)
    #     # plt.show()
    #
    # print()



    plotter = Plotterly()
    plotter.fig.update_layout(title=dict(text='Single target processing' + fig_title), title_x=0.5)
    # plotter.plot_ground_truths(truth, [0, 2], truths_label='Ground truth', line=dict(dash="dash", color='black'))
    # plotter.plot_tracks(Track(prior), [0, 2], uncertainty=True, track_label='Target prior')
    plotter.plot_measurements(measurements, [2, 4], measurements_label='Measurements')
    # plotter.plot_tracks(track_iplf, [0, 2], uncertainty=True, track_label='IPLF track')
    plotter.plot_tracks(track_iekf, [0, 2], uncertainty=True, track_label='IEKF track')
    # plotter.plot_tracks(track_ipls, [0, 2], uncertainty=True, track_label='IPLS track')
    plotter.fig.show()

    # # Plotting results using Plotter
    # from stonesoup.plotter import Plotter
    # plotter = Plotter()
    # plotter.plot_ground_truths(truth, [0, 2], truths_label='Ground truth')
    # plotter.plot_measurements(measurements, [0, 2], measurements_label='Measurements')
    # plotter.plot_tracks(track_iplf, [0, 2], uncertainty=True, track_label='IPLF track')
    # plotter.plot_tracks(track_iekf, [0, 2], uncertainty=True, track_label='IEKF track')
    # plotter.plot_tracks(track_ipls, [0, 2], uncertainty=True, track_label='IPLS track')
    # plotter.fig.show()


if __name__ == "__main__":
    main()
