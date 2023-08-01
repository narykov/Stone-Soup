#!/usr/bin/env python
import warnings

warnings.simplefilter('always', UserWarning)

import numpy as np
# import copy
from datetime import datetime, timedelta
from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRange
from stonesoup.plotter import Plotterly
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track

from stonesoup.robusstod.stonesoup.models.transition import LinearisedDiscretisation
from stonesoup.robusstod.stonesoup.predictor import ExtendedKalmanPredictor
from stonesoup.robusstod.stonesoup.updater import IPLFKalmanUpdater

from stonesoup.robusstod.utils import get_initial_state
from stonesoup.robusstod.utils import get_groundtruth_path
from stonesoup.robusstod.utils import get_prior
from stonesoup.robusstod.utils import get_measurement_histories

from stonesoup.robusstod.physics.constants import G, M_earth
from stonesoup.robusstod.physics.other import get_noise_coefficients

use_godot = False
if not use_godot:
    from stonesoup.robusstod.physics.godot import KeplerianToCartesian
    from stonesoup.robusstod.physics.godot import twoBody3d_da
else:
    from stonesoup.robusstod.physics.basic import KeplerianToCartesian
    from stonesoup.robusstod.physics.basic import twoBody3d_da


def do_STT(prior=None, predictor=None, updater=None, measurement_history=None):
    track = Track(prior)
    for measurement in measurement_history:
        prediction = predictor.predict(prior, timestamp=measurement.timestamp)
        hypothesis = SingleHypothesis(prediction, measurement)  # Group a prediction and measurement
        post = updater.update(hypothesis)
        track.append(post)
        prior = track[-1]
    return track


def main():
    start_time = datetime(2000, 1, 1)
    np.random.seed(1991)

    # We begin by specifying the mean state of a target state by picking a credible set of Keplerian elements, and
    # then converting them into Cartesian domain.

    a, e, i, w, omega, nu = (9164000, 0.03, 70, 0, 0, 0)
    # the values above from Gemma https://github.com/alecksphillips/SatelliteModel/blob/main/Stan-InitialStateTarget.py
    # a, e, i, w, omega, nu (m, _, deg, deg, deg, deg)
    # NB: a, e, I, RAAN, argP, ta (km, _, rad, rad, rad, rad) as in https://godot.io.esa.int/tutorials/T04_Astro/T04scv/
    K = np.array([a, e, np.radians(i), np.radians(w), np.radians(omega), np.radians(nu)])  # now in SI units (m & rad)
    ndim_state = 6
    mapping_location = (0, 2, 4)  # encodes location indices in the state vector
    mapping_velocity = (1, 3, 5)  # encodes velocity indices in the state vector
    GM = G.value * M_earth.value  # https://en.wikipedia.org/wiki/Standard_gravitational_parameter (m^3 s^−2)
    population_mean = KeplerianToCartesian(K, GM, ndim_state, mapping_location, mapping_velocity)  # into Cartesian

    target_initial_covariance = np.diag([50000 ** 2, 100 ** 2, 50000 ** 2, 100 ** 2, 50000 ** 2, 100 ** 2])

    scenario_parameters = {
        'n_mc_runs': 1,
        'time_interval': timedelta(seconds=50),
        'n_time_steps': 10,
        'target_initial_covariance': target_initial_covariance
    }

    initial_state = get_initial_state(state_vector=population_mean, start_time=start_time)
    prior = get_prior(initial_state, scenario_parameters['target_initial_covariance'], start_time, bias_fixed=True)

    timesteps = [start_time + k * scenario_parameters['time_interval'] for k in
                 range(scenario_parameters['n_time_steps'])]
    transition_model = LinearisedDiscretisation(
        diff_equation=twoBody3d_da,
        linear_noise_coeffs=get_noise_coefficients(GM)
    )
    truth = get_groundtruth_path(
        initial_target_state=initial_state,
        transition_model=transition_model,
        timesteps=timesteps,
        noise=True
    )

    # Generate measurements
    sigma_el, sigma_b, sigma_range = np.deg2rad(0.01), np.deg2rad(0.01), 100
    sensor_x, sensor_y, sensor_z = 0, 0, 0
    sensor_parameters = {
        'ndim_state': ndim_state,
        'mapping': mapping_location,
        'noise_covar': np.diag([sigma_el ** 2, sigma_b ** 2, sigma_range ** 2]),
        'translation_offset': np.array([[sensor_x], [sensor_y], [sensor_z]])
    }
    measurement_model = CartesianToElevationBearingRange(
        ndim_state=sensor_parameters['ndim_state'],
        mapping=sensor_parameters['mapping'],
        noise_covar=sensor_parameters['noise_covar'],
        translation_offset=sensor_parameters['translation_offset']
    )
    measurement_histories = get_measurement_histories(
        truth=truth, measurement_model=measurement_model, n_mc_runs=scenario_parameters['n_mc_runs']
    )

    # Put together a filter
    predictor = ExtendedKalmanPredictor(transition_model)
    updater = IPLFKalmanUpdater()

    track_STT_list = []
    for measurement_history in measurement_histories:
        track_STT = do_STT(prior=prior, predictor=predictor, updater=updater, measurement_history=measurement_history)
        track_STT_list.append(track_STT)

    plotter = Plotterly()
    mc_run_to_plot = 0
    plotter.plot_ground_truths(truth, [0, 2], line=dict(dash="dash", color='black'))
    plotter.plot_measurements(measurement_histories[mc_run_to_plot], [0, 2])
    plotter.plot_tracks(track_STT_list[mc_run_to_plot], [0, 2], uncertainty=True)
    plotter.fig.show()


if __name__ == "__main__":
    main()
