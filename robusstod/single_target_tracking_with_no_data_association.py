#!/usr/bin/env python

"""
Tracking a single orbiting object with no detection failures (no false alarms and missed detections)
========================================
This is a demonstration using the implemented IPLF filter in the context of space situation awareness.
"""

import numpy as np
from datetime import datetime, timedelta

from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRange
from stonesoup.plotter import Plotterly
from stonesoup.types.array import CovarianceMatrix
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.detection import TrueDetection
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.prediction import GaussianStatePrediction
from stonesoup.types.track import Track

from stonesoup.robusstod.stonesoup.models.transition import LinearisedDiscretisation
from stonesoup.robusstod.stonesoup.predictor import ExtendedKalmanPredictor
from stonesoup.robusstod.stonesoup.updater import IPLFKalmanUpdater
from stonesoup.robusstod.physics.constants import G, M_earth
from stonesoup.robusstod.physics.other import get_noise_coefficients
use_godot = True
if use_godot:
    from stonesoup.robusstod.physics.godot import KeplerianToCartesian, twoBody3d_da
else:
    from stonesoup.robusstod.physics.basic import KeplerianToCartesian, twoBody3d_da


def do_single_target_tracking(prior=None, predictor=None, updater=None, measurements=None):
    if measurements is None:
        measurements = []

    track = Track(prior)
    for measurement in measurements:
        prediction = predictor.predict(prior, timestamp=measurement.timestamp)
        hypothesis = SingleHypothesis(prediction, measurement)  # Group a prediction and measurement
        post = updater.update(hypothesis)
        track.append(post)
        prior = track[-1]

    return track


def main():
    np.random.seed(1991)
    start_time = datetime(2000, 1, 1)
    time_parameters = {
        'n_time_steps': 10,
        'time_interval': timedelta(seconds=50)
    }
    timesteps = [start_time + k * time_parameters['time_interval'] for k in range(time_parameters['n_time_steps'])]

    # We begin by specifying the mean state of a target state by picking a credible set of Keplerian elements, and
    # then converting them into Cartesian domain.

    a, e, i, w, omega, nu = (9164000, 0.03, 70, 0, 0, 0)
    # the values above are from https://github.com/alecksphillips/SatelliteModel/blob/main/Stan-InitialStateTarget.py
    # a, e, i, w, omega, nu (m, _, deg, deg, deg, deg)
    # NB: a, e, I, RAAN, argP, ta (km, _, rad, rad, rad, rad) as in https://godot.io.esa.int/tutorials/T04_Astro/T04scv/
    K = np.array([a, e, np.radians(i), np.radians(w), np.radians(omega), np.radians(nu)])  # now in SI units (m & rad)
    ndim_state = 6
    mapping_location = (0, 2, 4)  # encodes location indices in the state vector
    mapping_velocity = (1, 3, 5)  # encodes velocity indices in the state vector
    GM = G * M_earth  # https://en.wikipedia.org/wiki/Standard_gravitational_parameter (m^3 s^−2)
    initial_state_vector = KeplerianToCartesian(K, GM, ndim_state, mapping_location, mapping_velocity)  # into Cartesian
    initial_state = GroundTruthState(state_vector=initial_state_vector, timestamp=start_time)
    initial_covariance = CovarianceMatrix(np.diag([50000 ** 2, 100 ** 2, 50000 ** 2, 100 ** 2, 50000 ** 2, 100 ** 2]))

    deviation = np.linalg.cholesky(initial_covariance).T @ np.random.normal(size=initial_state.state_vector.shape)
    # deviation = np.matrix([-22068.01433784, 69.37315652, 507.76211348, -86.74038986, -58321.63970861, 89.04789997])
    prior = GaussianStatePrediction(state_vector=initial_state.state_vector + deviation,
                                    covar=initial_covariance,
                                    timestamp=start_time)

    transition_model = LinearisedDiscretisation(
        diff_equation=twoBody3d_da,
        linear_noise_coeffs=get_noise_coefficients(GM)
    )

    # Generate ground truth trajectory, using the initial target state, target dynamics and the grid of timesteps
    successive_time_steps = timesteps[1:]  # dropping the very first start_time
    truth = GroundTruthPath(initial_state)
    for timestamp in successive_time_steps:
        truth.append(GroundTruthState(
            transition_model.function(state=truth[-1], noise=True, time_interval=time_parameters['time_interval']),
            timestamp=timestamp)
        )

    # Specify sensor parameters and generate a history of measurements for the timesteps
    sigma_el, sigma_b, sigma_range = np.deg2rad(0.01), np.deg2rad(0.01), 10000
    sensor_x, sensor_y, sensor_z = 0, 0, 0
    sensor_parameters = {
        'ndim_state': ndim_state,
        'mapping': mapping_location,
        'noise_covar': CovarianceMatrix(np.diag([sigma_el ** 2, sigma_b ** 2, sigma_range ** 2])),
        'translation_offset': np.array([[sensor_x], [sensor_y], [sensor_z]])
    }

    measurement_model = CartesianToElevationBearingRange(
        ndim_state=sensor_parameters['ndim_state'],
        mapping=sensor_parameters['mapping'],
        noise_covar=sensor_parameters['noise_covar'],
        translation_offset=sensor_parameters['translation_offset']
    )

    measurements = []
    for state in truth:
        measurement = measurement_model.function(state=state, noise=True)
        timestamp = state.timestamp
        measurements.append(TrueDetection(
            state_vector=measurement,
            timestamp=timestamp,
            groundtruth_path=truth,
            measurement_model=measurement_model)
        )

    # Here we finally specify how the filtering recursion is implemented
    predictor = ExtendedKalmanPredictor(transition_model)
    updater = IPLFKalmanUpdater()

    # Perform tracking/filtering
    track_stt = do_single_target_tracking(prior=prior, predictor=predictor, updater=updater, measurements=measurements)

    # Plotting the results
    plotter = Plotterly()
    plotter.plot_ground_truths(truth, [0, 2], truths_label='Ground truth', line=dict(dash="dash", color='black'))
    plotter.plot_measurements(measurements, [0, 2], measurements_label='Measurements')
    plotter.plot_tracks(track_stt, [0, 2], uncertainty=True, track_label='IPLF track')
    plotter.fig.show()


if __name__ == "__main__":
    main()
