#!/usr/bin/env python

"""
Tracking multiple orbiting object with detection failures (false alarms and missed detections)
========================================
This is a demonstration using the implemented IPLF filter in the context of space situation awareness.
It can use either built-in model of acceleration or GODOT's capability to evaluate acceleration.
"""

import sys
import numpy as np
from datetime import datetime, timedelta

from scipy.stats import uniform
from ordered_set import OrderedSet

from stonesoup.functions import gm_reduce_single
from stonesoup.plotter import Plotterly
from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRange
from stonesoup.types.array import StateVectors, CovarianceMatrix
from stonesoup.types.detection import TrueDetection, Clutter
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.prediction import GaussianStatePrediction
from stonesoup.types.state import State
from stonesoup.types.track import Track
from stonesoup.types.update import GaussianStateUpdate
from stonesoup.dataassociator.probability import JPDA

# ROBUSSTOD MODULES
from stonesoup.robusstod.stonesoup.hypothesiser import PDAHypothesiser
from stonesoup.robusstod.stonesoup.models.transition import LinearisedDiscretisation
from stonesoup.robusstod.stonesoup.predictor import ExtendedKalmanPredictor
from stonesoup.robusstod.stonesoup.updater import IPLFKalmanUpdater
from stonesoup.robusstod.physics.constants import G, M_earth
from stonesoup.robusstod.physics.other import get_noise_coefficients

use_godot = True
if use_godot:
    try:
        import godot
    except ModuleNotFoundError as e:
        print(e.msg)
        sys.exit(1)  # the exit code of 1 is a convention that means something went wrong
    from stonesoup.robusstod.physics.godot import KeplerianToCartesian, diff_equation
else:
    from stonesoup.robusstod.physics.basic import KeplerianToCartesian, diff_equation


def get_observation_history(truths, timesteps, measurement_model, sensor_parameters):
    """ This follows the clutter generation as in
    https://stonesoup.readthedocs.io/en/v1.0/auto_tutorials/06_DataAssociation-MultiTargetTutorial.html#generate-detections-with-clutter
    """

    prob_detect = sensor_parameters['prob_detect']
    clutter_rate = sensor_parameters['clutter_rate']

    observation_history = []

    for timestamp in timesteps:
        observation = set()

        for truth in truths:
            # True detections. Generate actual detection from the state with a 1-prob_detect chance of no detection.
            if np.random.rand() <= prob_detect:
                measurement = measurement_model.function(truth[timestamp], noise=True)
                observation.add(TrueDetection(state_vector=measurement,
                                              groundtruth_path=truth,
                                              timestamp=timestamp,
                                              measurement_model=measurement_model))

            # False alarm. Generate clutter measurements at this timestep.
            truth_x = truth[timestamp].state_vector[0]
            truth_y = truth[timestamp].state_vector[2]
            truth_z = truth[timestamp].state_vector[4]
            if clutter_rate == 0:
                print('Interpreting \lambda=0 as no clutter.')
                n_false_alarms = 0
            else:
                n_false_alarms = np.random.randint(clutter_rate)

            for _ in range(n_false_alarms):
                width = 100000
                x = uniform.rvs(truth_x - 0.5 * width, width)
                y = uniform.rvs(truth_y - 0.5 * width, width)
                z = uniform.rvs(truth_z - 0.5 * width, width)
                clutter_state = State(state_vector=np.array([x, np.nan, y, np.nan, z, np.nan]),
                                      timestamp=timestamp)
                clutter_measurement = measurement_model.function(clutter_state, noise=False)
                clutter_detection = Clutter(state_vector=clutter_measurement,
                                            timestamp=timestamp,
                                            measurement_model=measurement_model)
                observation.add(clutter_detection)

        observation_history.append(observation)

    return observation_history


def do_JPDA(priors, timesteps, observation_history, data_associator):
    """Implementation follows https://stonesoup.readthedocs.io/en/latest/auto_tutorials/08_JPDATutorial.html"""

    tracks = set()
    for prior_pdf in priors:
        tracks.add(Track(prior_pdf))

    for timestep, observation in zip(timesteps, observation_history):
        # NB: Observation is understood here as a set of true measurements and false alarms.
        hypotheses = data_associator.associate(tracks, observation, timestep, allow_singular=True)

        # Loop through each track, performing the association step with weights adjusted according to JPDA.
        for track in tracks:
            track_hypotheses = hypotheses[track]

            posterior_states = []
            posterior_state_weights = []
            for hypothesis in track_hypotheses:
                if not hypothesis:
                    posterior_states.append(hypothesis.prediction)
                else:
                    posterior_state = data_associator.hypothesiser.updater.update(hypothesis)
                    posterior_states.append(posterior_state)
                posterior_state_weights.append(hypothesis.probability)

            means = StateVectors([state.state_vector for state in posterior_states])
            covars = np.stack([state.covar for state in posterior_states], axis=2)
            weights = np.asarray(posterior_state_weights)

            # Reduce mixture of states to one posterior estimate Gaussian.
            post_mean, post_covar = gm_reduce_single(means, covars, weights)

            # Add a Gaussian state approximation to the track.
            track.append(GaussianStateUpdate(post_mean, post_covar, track_hypotheses, timestep))

    return tracks


def main():
    start_time = datetime(2000, 1, 1)
    np.random.seed(1991)

    # We begin by specifying the mean state of a target population by picking a credible set of Keplerian elements, and
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
    population_mean = KeplerianToCartesian(K, GM, ndim_state, mapping_location, mapping_velocity)  # into Cartesian

    population_covariance = np.diag([150000 ** 2, 100 ** 2, 150000 ** 2, 100 ** 2, 150000 ** 2, 100 ** 2])
    target_initial_covariance = np.diag([50000 ** 2, 100 ** 2, 50000 ** 2, 100 ** 2, 50000 ** 2, 100 ** 2])

    scenario_parameters = {
        'n_time_steps': 10,
        'time_interval': timedelta(seconds=10),
        'n_targets': 5,
        'population_mean': population_mean,
        'population_covariance': population_covariance,
        'target_initial_covariance': target_initial_covariance
    }

    initial_states = []
    for _ in range(scenario_parameters['n_targets']):
        deviation = np.linalg.cholesky(scenario_parameters['population_covariance']).T @ np.random.normal(size=scenario_parameters['population_mean'].shape)
        state_vector = population_mean + deviation
        initial_state = GroundTruthState(state_vector=state_vector, timestamp=start_time)
        initial_states.append(initial_state)

    priors = []
    for initial_state in initial_states:
        deviation = np.linalg.cholesky(scenario_parameters['target_initial_covariance']).T @ np.random.normal(size=initial_states[0].state_vector.shape)
        prior = GaussianStatePrediction(state_vector=initial_state.state_vector + deviation,
                                        covar=target_initial_covariance,
                                        timestamp=start_time)
        priors.append(prior)

    timesteps = [start_time + k * scenario_parameters['time_interval'] for k in range(scenario_parameters['n_time_steps'])]
    transition_model = LinearisedDiscretisation(
        diff_equation=diff_equation,
        linear_noise_coeffs=get_noise_coefficients(GM)
    )
    truths = OrderedSet()
    for initial_state in initial_states:
        truth = GroundTruthPath(initial_state)
        successive_time_steps = timesteps[1:]  # dropping the very first start_time
        for timestamp in successive_time_steps:
            truth.append(GroundTruthState(
                transition_model.function(state=truth[-1], noise=True, time_interval=scenario_parameters['time_interval']),
                timestamp=timestamp)
            )
        truths.add(truth)

    # Generate measurements
    sigma_el, sigma_b, sigma_range = np.deg2rad(0.01), np.deg2rad(0.01), 10000
    sensor_x, sensor_y, sensor_z = 0, 0, 0
    sensor_parameters = {
        'prob_detect': 0.9,
        'clutter_rate': 3,
        'clutter_spatial_density': 0.125 * 0.00001,
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
    observations = get_observation_history(truths, timesteps, measurement_model, sensor_parameters)

    # Next, we specify filtering solution. First, we specify an implementation of the filtering recursion
    predictor = ExtendedKalmanPredictor(transition_model)
    updater = IPLFKalmanUpdater(tolerance=1e-1, max_iterations=5)  # Using default values
    # Second, we specify the data association algorithm
    hypothesiser = PDAHypothesiser(
        predictor=predictor,
        updater=updater,
        clutter_spatial_density=sensor_parameters['clutter_spatial_density'],
        prob_detect=sensor_parameters['prob_detect']
    )
    data_associator = JPDA(hypothesiser=hypothesiser)

    tracks = do_JPDA(priors, timesteps, observations, data_associator)

    plotter = Plotterly()
    plotter.plot_ground_truths(truths, [0, 2], line=dict(dash="dash", color='black'))
    plotter.plot_measurements(observations, [0, 2])
    plotter.plot_tracks(tracks, [0, 2], uncertainty=True)
    plotter.fig.show()


if __name__ == "__main__":
    main()
