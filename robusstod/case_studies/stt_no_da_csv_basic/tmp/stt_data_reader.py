#!/usr/bin/env python

"""
Tracking a single orbiting object with no detection failures (no false alarms and missed detections)
========================================
This is a demonstration using the implemented IEKF/IPLF/IPLS algorithms in the context of space situation awareness.
It can use either built-in model of acceleration or GODOT's capability to evaluate acceleration.
"""
from stonesoup.types.angle import Bearing, Elevation
import sys
from matplotlib.patches import Ellipse
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


def plot_cov_ellipse(cov, pos, nstd=3, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.
    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta,
                    alpha=0.4, **kwargs)

    # ax.add_artist(ellip)
    ax.add_patch(ellip)
    return ellip


def plot_tracks(tracks, show_error=True, ax=None, color='r', label='Tracks', mapping=None):
    if mapping is None:
        mapping = [0, 2]

    if not ax:
        ax = plt.gca()
    ax.plot([], [], '-', color=color, label=label)
    for track in tracks:
        data = np.array([state.state_vector for state in track.states])
        ax.plot(data[:, mapping[0]], data[:, mapping[1]], '-', marker='.', color=color)
        if show_error:
            plot_cov_ellipse(track.state.covar[mapping, :][:, mapping],
                             track.state.mean[mapping, :], edgecolor=color,
                             facecolor='r', ax=ax)

def plot_detection(detection, ax='Detections', color='b', marker=None, label=None, mapping=None):
    if not ax:
        ax = plt.gca()
    if mapping is None:
        mapping = [0, 2]
    if marker is None:
        marker = 'x'
    datum = detection.measurement_model.inverse_function(detection)
    plt.plot(datum[mapping[0]], datum[mapping[1]], color=color, marker=marker, label=label)

def plot_prediction(pred, ax=None, color='red', show_error=True, label='Prediction', marker=None, mapping=None, step=None):
    if not ax:
        ax = plt.gca()
    if mapping is None:
        mapping = [0, 2]
    if marker is None:
        marker = '.'
    plt.plot(pred.state_vector[mapping[0]], pred.state_vector[mapping[1]], marker=marker, color=color, label=label)
    if step is not None:
        plt.text(pred.state_vector[mapping[0]], pred.state_vector[mapping[1]], str(step))
    if show_error:
        plot_cov_ellipse(pred.covar[mapping, :][:, mapping],
                         pred.state_vector[mapping, :], edgecolor=color,
                         facecolor='none', ax=ax)


def do_single_target_tracking(prior=None, predictor=None, updater=None, measurements=None):
    if measurements is None:
        measurements = []
    mapping = [2, 4]
    track = Track([prior])
    plot_tracks([Track([prior])], mapping=mapping)
    plt.plot(0, 0, marker='d', color='k')
    measurement_predictions = []
    for i, measurement in enumerate(measurements):
        plot_detection(measurement, mapping=mapping)
        prediction = predictor.predict(prior, timestamp=measurement.timestamp)
        # plot_prediction(prediction, mapping=mapping, step=i)
        plt.plot(0, 0, 'd')
        hypothesis = SingleHypothesis(prediction, measurement)  # Group a prediction and measurement
        # print(measurement.state_vector[0])
        post = updater.update(hypothesis)
        # measurement_prediction = Detection(state_vector=post.hypothesis.measurement_prediction.state_vector,
        #                                    measurement_model=measurement.measurement_model)
        # measurement_predictions.append(measurement_prediction)
        # plot_detection(measurement_prediction, mapping=mapping, marker='o')
        track.append(post)
        prior = track[-1]
        plt.gca().autoscale

    plot_tracks([track], mapping=mapping)

    fig, ax = plt.subplots()
    for i, data in enumerate(zip(measurements, measurement_predictions)):
        true = data[0]
        pred = data[1]
        plt.scatter(i, true.state_vector[1], marker='x')
        plt.scatter(i, pred.state_vector[1], marker='o')


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
    max_datapoints: int = Property(default=None, doc='N datapoints to consider')
    measurement_model: MeasurementModel = Property(default=None, doc='Entry')

    def _unit_conversion(self, row):
        state_vector = []
        for col_name in self.state_vector_fields:
            value = float(row[col_name])
            if col_name.startswith('ANGLE'):
                if col_name == 'ANGLE_1':
                    # value = value - 90
                    if value > 90:
                        pass
                        # value = value - 360
                    # print(value)
                if col_name == 'ANGLE_1':
                    print(f'In degrees {value}')
                    value = np.deg2rad(value)
                    print(f'In radians {value}')
                    value = Bearing(value)
                    print(f'In elevations {value}')
                    print()
                if col_name == 'ANGLE_2':
                    # print(value)
                    # print(f'In degrees {value}')
                    value = np.deg2rad(value)
                    # print(f'In radians {value}')
                    value = Bearing(value)
                    # print(f'In bearings {value}')
            if col_name in ['RANGE', 'DOPPLER_INSTANTANEOUS']:
                value = value * 1000
            state_vector.append([value])

        return state_vector

    @BufferedGenerator.generator_method
    def detections_gen(self):
        n_datapoints = 0
        with self.path.open(encoding=self.encoding, newline='') as csv_file:

            for row in csv.DictReader(csv_file, **self.csv_options):

                skip = False
                for key in self.filters:
                    if row[key] != self.filters[key]:
                        skip = True
                        continue

                if skip:
                    continue

                n_datapoints += 1
                if self.max_datapoints is not None:
                    if n_datapoints > self.max_datapoints:
                        break

                time = self._get_time(row)
                sv = self._unit_conversion(row)
                detection = Detection(state_vector=StateVector(sv),
                                      timestamp=time,
                                      measurement_model=self.measurement_model,
                                      metadata=self._get_metadata(row))
                yield detection

    def get_initial_state(self, prior_state):
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

                # state_vector = self.measurement_model.inverse_function(detection)
                # state_vector[[1, 3, 5]] = 0
                # return State(state_vector=state_vector, timestamp=time)

                from stonesoup.initiator.simple import SimpleMeasurementInitiator
                initiator = SimpleMeasurementInitiator(prior_state=prior_state)
                detections = [detection]
                tracks = initiator.initiate(detections, time)

                return tracks.pop()


def main():
    np.random.seed(1991)
    # start_time = datetime(2000, 1, 1)
    # time_parameters = {
    #     'n_time_steps': 50,
    #     'time_interval': timedelta(seconds=1200)
    # }
    # TODO: consider arbitrary time intervals to demonstrate the flexibility of the approach
    # timesteps = [start_time + k * time_parameters['time_interval'] for k in range(time_parameters['n_time_steps'])]

    # We begin by specifying the true initial target state by picking a credible set of Keplerian elements, and
    # then converting them into Cartesian domain.

    # a, e, i, w, omega, nu = (9164000, 0.03, 70, 0, 0, 0)
    # # Orbital elements, see https://en.wikipedia.org/wiki/Orbital_elements
    # # Two elements define the shape and size of the ellipse:
    # # a = semimajor axis
    # # e = eccentricity
    # # Two elements define the orientation of the orbital plane in which the ellipse is embedded:
    # # i = inclination
    # # omega = longitude of the ascending node
    # # The remaining two elements are as follows:
    # # w = argument of periapsis
    # # nu = true anomaly
    # # Visualisation (png): https://en.wikipedia.org/wiki/Orbital_elements#/media/File:Orbit1.svg
    # # A simple interactive visualisation can be accessed at https://orbitalmechanics.info,
    # # where a is given in the multiples of the Earth's radius, and i, omega, w and nu are defined in degrees.
    # #
    # # a, e, i, w, omega, nu (m, _, deg, deg, deg, deg)
    # # NB: a, e, I, RAAN, argP, ta (km, _, rad, rad, rad, rad) as in https://godot.io.esa.int/tutorials/T04_Astro/T04scv/
    # K = np.array([a, e, np.radians(i), np.radians(w), np.radians(omega), np.radians(nu)])  # now in SI units (m & rad)
    ndim_state = 6
    mapping_location = (0, 2, 4)  # encodes location indices in the state vector
    mapping_velocity = (1, 3, 5)  # encodes velocity indices in the state vector
    GM = G * M_earth  # https://en.wikipedia.org/wiki/Standard_gravitational_parameter (m^3 s^−2)


    # Specify sensor parameters and generate a history of measurements for the time steps
    # sigma_el, sigma_b, sigma_range, sigma_range_rate = np.deg2rad(0.01), np.deg2rad(0.01), 10000, 100
    # parameters for RR01
    sigma_r = 20  # Range: 20.0 m
    sigma_a = np.deg2rad(400*0.001)  # Azimuth - elevation: 400.0 mdeg
    sigma_rr = 650.0 * 0.001  # Range-rate: 650.0 mm/s
    sigma_el, sigma_b, sigma_range, sigma_range_rate = sigma_a, sigma_a, sigma_r, sigma_rr
    sensor_x, sensor_y, sensor_z = 0, 0, 0
    noise_covar = CovarianceMatrix(np.diag([sigma_el ** 2, sigma_b ** 2, sigma_range ** 2, sigma_range_rate ** 2]))
    noise_covar = CovarianceMatrix(np.diag([sigma_el ** 2, sigma_b ** 2, sigma_range ** 2]))
    sensor_parameters = {
        'ndim_state': ndim_state,
        'mapping': mapping_location,
        'noise_covar': noise_covar,
        'translation_offset': np.array([[sensor_x], [sensor_y], [sensor_z]])
    }

    # measurement_model = CartesianToElevationBearingRange(
    #     ndim_state=sensor_parameters['ndim_state'],
    #     mapping=sensor_parameters['mapping'],
    #     noise_covar=sensor_parameters['noise_covar'],
    #     translation_offset=sensor_parameters['translation_offset']
    # )
    measurement_model = CartesianToElevationBearingRange(
        ndim_state=sensor_parameters['ndim_state'],
        mapping=sensor_parameters['mapping'],
        noise_covar=sensor_parameters['noise_covar'],
        translation_offset=sensor_parameters['translation_offset']
    )
    path = "src/csv/RODDAS_OD_00_015.csv"
    filter_value = 'RR02'
    # from stonesoup.reader.generic import CSVDetectionReader
    filters = {'STATION': 'RR02',
               'TARGET_ID': '00039451'}  # targets: 00055165, 00039451
    state_vector_fields = ("ANGLE_1", "ANGLE_2", "RANGE", "DOPPLER_INSTANTANEOUS")
    state_vector_fields = ("ANGLE_1", "ANGLE_2", "RANGE")
    detector = CSVDetectionReader(
        path=path,
        state_vector_fields=state_vector_fields,
        time_field="TIME",
        filters=filters,
        max_datapoints=10,
        measurement_model=measurement_model
    )

    initial_covariance = CovarianceMatrix(np.diag([10000 ** 2, 1000 ** 2, 10000 ** 2, 1000 ** 2, 10000 ** 2, 1000 ** 2]))
    from stonesoup.types.state import GaussianState
    prior_state = GaussianState(state_vector=[0, 0, 0, 0, 0, 0], covar=initial_covariance)
    initial_state = detector.get_initial_state(prior_state=prior_state)

    # initial_state_vector = KeplerianToCartesian(K, GM, ndim_state, mapping_location, mapping_velocity)# into Cartesian
    initial_state = GroundTruthState(state_vector=initial_state.state_vector, timestamp=initial_state.timestamp)

    deviation = np.linalg.cholesky(initial_covariance).T @ np.random.normal(size=initial_state.state_vector.shape)
    # deviation = np.array([-122068.01433784, 69.37315652, 507.76211348, -86.74038986, -58321.63970861, 89.04789997]).reshape((6, 1))
    # prior = GaussianState(state_vector=initial_state.state_vector + deviation,
    #                       covar=initial_covariance,
    #                       timestamp=start_time)
    deviation = 0

    prior = GaussianStatePrediction(state_vector=initial_state.state_vector + deviation,
                                    covar=initial_covariance,
                                    timestamp=initial_state.timestamp)

    transition_model = LinearisedDiscretisation(
        diff_equation=diff_equation,
        linear_noise_coeffs=get_noise_coefficients(GM)
    )
    # from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
    # q = 1
    # transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(q),
    #                                                           ConstantVelocity(q), ConstantVelocity(q)])

    # Generate ground truth trajectory, using the initial target state, target dynamics and the grid of timesteps
    # truth = GroundTruthPath(initial_state)
    # successive_time_steps = timesteps[1:]  # dropping the very first start_time
    # for timestamp in successive_time_steps:
    #     truth.append(GroundTruthState(
    #         transition_model.function(state=truth[-1], noise=True, time_interval=time_parameters['time_interval']),
    #         timestamp=timestamp)
    #     )

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
    track_iplf = do_single_target_tracking(prior=prior, predictor=predictor_ukf, updater=updater_iplf, measurements=measurements)
    # track_iekf = do_single_target_tracking(prior=prior, predictor=predictor_ekf, updater=updater_iekf, measurements=measurements)
    # # track_ipls = IPLSKalmanSmoother(transition_model=transition_model).smooth(track_iplf)
    # track_iplf = do_stt(prior=prior, predictor=predictor_ukf, updater=updater_iplf, detector=detector)
    # track_iekf = do_stt(prior=prior, predictor=predictor_ekf, updater=updater_iekf, detector=detector)


    # Plotting the results using Plotterly

    # fig, ax = plt.subplots()
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.ylim((-2520788.3357520015, 2474705.9185321466))
    # plt.xlim((-2073231.8435078696, 2068652.27618956))


    # Plot ranges in time and just ordered
    # ranges = [measurement.state_vector[0] for measurement in measurements]
    measurements = measurements[:-1]
    angles_1 = [measurement.state_vector[0] for measurement in measurements]
    angles_2 = [measurement.state_vector[1] for measurement in measurements]
    times = [measurement.timestamp for measurement in measurements]
    # plt.plot(ranges, linestyle='None', marker='.', color='b')
    color = __import__('matplotlib.pyplot').cm.rainbow(np.linspace(0, 1, len(measurements)))
    for i, value in enumerate(zip(angles_1, angles_2, times)):
        plt.scatter(x=value[2], y=value[0], marker='.', color=color[i])
        plt.scatter(x=value[2], y=value[1], marker='x', color=color[i])

    print()
    # plt.show()
    # #
    # # # plt.scatter(x=times, y=ranges)

    # previous_timestep = None
    # timedeltas = []
    # for measurement in measurements:
    #     if previous_timestep is None:
    #         previous_timestep = measurement.timestamp
    #         continue
    #     delta = timedelta.total_seconds(measurement.timestamp-previous_timestep)
    #     previous_timestep = measurement.timestamp
    #     timedeltas.append(delta)
    # unique_deltas = list(set(timedeltas))
    #
    # plt.plot(timedeltas, linestyle='None', marker='.')
    # # for unique_delta in unique_deltas:
    # #     plt.plot(unique_delta, marker='x', label=str(unique_delta)+' sec')
    # # plt.legend()
    # plt.gca().set_yscale('log')
    # plt.xlabel('Timestep')
    # plt.ylabel('Seconds')
    #
    # mode_1 = [i for i in unique_deltas if i < 1e1]
    # mode_2 = [i for i in unique_deltas if 1e1 < i < 1e5]
    # mode_3 = [i for i in unique_deltas if i > 1e5]
    #
    # mode_1_mean = sum(mode_1) / len(mode_1) if len(mode_1) != 0 else None
    # mode_2_mean = sum(mode_2) / len(mode_2) if len(mode_2) != 0 else None
    # mode_3_mean = sum(mode_3) / len(mode_3) if len(mode_3) != 0 else None
    mapping = [2, 4]
    color = __import__('matplotlib.pyplot').cm.rainbow(np.linspace(0, 1, len(measurements)))

    # color = list(np.random.choice(range(256), size=3))
    # for i, measurement in enumerate(measurements):
    #     state = measurement_model.inverse_function(measurement)
    #     plt.plot(state[mapping[0]], state[mapping[1]], color=color[i], marker='x')
    #     plt.pause(0.01)
    #     # plt.show()

    delta = 8
    previous_time = None
    color = 'r'
    # def create_figure():
    #     plt.figure()
    #     plt.gca().set_aspect('equal', adjustable='box')
    #     plt.xlim((-1097748.5938401269, 1096867.4888711423))
    #     plt.ylim((-1304659.393434252, 1288530.0494793188))
    #
    # create_figure()
    # for i, measurement in enumerate(measurements):
    #     if previous_time is not None:
    #         if delta > 10:
    #             color = list(np.random.choice(range(256), size=3)/256)
    #             plt.show(block=False)
    #             create_figure()
    #     state = measurement.measurement_model.inverse_function(measurement)
    #     plt.plot(state[mapping[0]], state[mapping[1]], color=color, marker='x')
    #     if previous_time is None:
    #         previous_time = measurement.timestamp
    #         continue
    #     delta = (measurement.timestamp - previous_time).total_seconds()
    #     previous_time = measurement.timestamp
    #     plt.pause(0.01)




    print()

    # fig, ax = plt.subplots()
    # track = track_iekf
    track = track_iplf
    mapping = mapping
    plot_tracks([track], mapping=mapping)
    # plt.gca().autoscale_view()
    # plt.show()

    data = np.array([state.state_vector.squeeze() for state in track]).T
    linestyle = '-' if len(track) > 1 else '.'  # Use a line for tracks with more than one state
    plt.plot(data[2, :], data[4, :], f'{linestyle}', color='xkcd:black', marker='.', markersize=5)
    plt.show()
    print()
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
