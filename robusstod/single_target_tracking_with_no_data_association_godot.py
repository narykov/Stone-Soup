#!/usr/bin/env python

"""
Tracking a single orbiting object with no detection failures (no false alarms and missed detections)
========================================
This is a demonstration using the implemented IPLF/IPLS algorithms in the context of space situation awareness.
It uses GODOT's capability to propagate trajectories.
"""

import sys
import numpy as np
from datetime import datetime, timedelta

from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRange
from stonesoup.plotter import Plotterly
from stonesoup.types.array import CovarianceMatrix, StateVector
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.detection import TrueDetection
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.prediction import GaussianStatePrediction
from stonesoup.types.track import Track
from stonesoup.updater.kalman import IteratedKalmanUpdater

# ROBUSSTOD MODULES
from stonesoup.robusstod.stonesoup.models.transition import LinearisedDiscretisation
from stonesoup.robusstod.stonesoup.predictor import ExtendedKalmanPredictor, UnscentedKalmanPredictor
from stonesoup.robusstod.stonesoup.smoother import IPLSKalmanSmoother
from stonesoup.robusstod.stonesoup.updater import IPLFKalmanUpdater
from stonesoup.robusstod.physics.constants import G, M_earth
from stonesoup.robusstod.physics.other import get_noise_coefficients
import sys

use_godot = False

if use_godot:
    try:
        import godot
    except ModuleNotFoundError as e:
        print(e.msg)
        sys.exit(1)  # the exit code of 1 is a convention that means something went wrong
    from stonesoup.robusstod.physics.godot import KeplerianToCartesian, diff_equation
    from stonesoup.robusstod.physics.godot import jacobian_godot
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


def main():
    np.random.seed(1991)
    time_parameters = {
        'n_time_steps': 10,
        'time_interval': timedelta(seconds=10),
        'start_time': datetime(2013, 1, 1)
    }

    from stonesoup.robusstod.stonesoup.models.transition_godot import GaussianTransitionGODOT
    transition_model = GaussianTransitionGODOT(
        universe_path='universe_test.yml',
        trajectory_path='trajectory_test.yml',
        noise_diff_coeff=0.05
    )
    cart_godot = np.array([-4685.75946803037,
                           3965.81226070460,
                           3721.45235707666,
                           -2.07276827105304,
                           3.45342193298209,
                           -6.27128141230935])
    initial_state = StateVector(transition_model.godot_to_stonesoup(cart_godot))
    initial_covariance = CovarianceMatrix(np.diag([50000 ** 2, 100 ** 2, 50000 ** 2, 100 ** 2, 50000 ** 2, 100 ** 2]))
    deviation = np.linalg.cholesky(initial_covariance).T @ np.random.normal(size=initial_state.shape)
    prior = GaussianStatePrediction(state_vector=StateVector(initial_state) + deviation,
                                    covar=initial_covariance,
                                    timestamp=time_parameters['start_time'])
    timesteps = [time_parameters['start_time']]

    truth = GroundTruthPath([GroundTruthState(initial_state, timestamp=timesteps[0])])

    num_steps = time_parameters['n_time_steps']
    time_interval = time_parameters['time_interval']
    for k in range(1, num_steps + 1):
        timesteps.append(timesteps[k-1] + time_interval)  # add next timestep to list of timesteps
        propagated_state = transition_model.function(truth[k-1], noise=True, time_interval=time_interval)
        truth.append(GroundTruthState(
            propagated_state,
            timestamp=timesteps[k]))

    # Specify sensor parameters and generate a history of measurements for the time steps
    sigma_el, sigma_b, sigma_range = np.deg2rad(0.01), np.deg2rad(0.01), 10000
    sensor_x, sensor_y, sensor_z = 0, 0, 0
    sensor_parameters = {
        'ndim_state': 6,
        'mapping': (0, 2, 4),
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
    predictor_ukf = UnscentedKalmanPredictor(transition_model)
    updater_iplf = IPLFKalmanUpdater(tolerance=1e-1, max_iterations=5)  # Using default values

    # Perform tracking/filtering/smooting
    track_iplf = do_single_target_tracking(prior=prior, predictor=predictor_ukf, updater=updater_iplf, measurements=measurements)
    track_ipls = IPLSKalmanSmoother(transition_model=transition_model).smooth(track_iplf)


    # Plotting the results using Plotterly
    plotter = Plotterly()
    plotter.fig.update_layout(title=dict(text='Single target processing' + fig_title), title_x=0.5)
    plotter.plot_ground_truths(truth, [0, 2], truths_label='Ground truth', line=dict(dash="dash", color='black'))
    plotter.plot_tracks(Track(prior), [0, 2], uncertainty=True, track_label='Target prior')
    plotter.plot_measurements(measurements, [0, 2], measurements_label='Measurements')
    plotter.plot_tracks(track_iplf, [0, 2], uncertainty=True, track_label='IPLF track')
    plotter.plot_tracks(track_ipls, [0, 2], uncertainty=True, track_label='IPLS track')
    plotter.fig.show()
    print()
    # # Plotting results using Plotter
    # from stonesoup.plotter import Plotter
    # plotter = Plotter()
    # plotter.plot_ground_truths(truth, [0, 2], truths_label='Ground truth')
    # plotter.plot_measurements(measurements, [0, 2], measurements_label='Measurements')
    # plotter.plot_tracks(track_iplf, [0, 2], uncertainty=True, track_label='IPLF track')
    # plotter.plot_tracks(track_ipls, [0, 2], uncertainty=True, track_label='IPLS track')
    # plotter.fig.show()


if __name__ == "__main__":
    main()
