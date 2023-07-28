#!/usr/bin/env python
import warnings

warnings.simplefilter('always', UserWarning)

import numpy as np
# import copy

from datetime import datetime, timedelta
from stonesoup.types.detection import TrueDetection, Clutter
from stonesoup.predictor.kalman import ExtendedKalmanPredictorROBUSSTOD
from stonesoup.types.array import StateVectors

from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.track import Track
from stonesoup.plotter import Plotterly
from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRange

from stonesoup.types.update import GaussianStateUpdate
from stonesoup.types.prediction import GaussianStatePrediction
from stonesoup.types.state import State
from scipy.stats import uniform

from stonesoup.functions import gm_reduce_single
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.dataassociator.probability import JPDA
from ordered_set import OrderedSet

from astropy.constants import G, M_earth, R_earth

# NEW CLASSES:
from stonesoup.updater.kalman import IPLFKalmanUpdater
from stonesoup.models.transition.nonlinear import LinearisedDiscretisation


using_godot = False

if using_godot:
    from funcs_godot import KeplerianToCartesian
    from funcs_godot import twoBody3d_da
else:
    from funcs_basic import KeplerianToCartesian
    from funcs_basic import twoBody3d_da


def get_noise_coefficients(GM):
    """ A function that returns adequate noise coefficients for Van Loan's method, ideally from physical considerations.
    We need these values that drive the process noise in the prediction step.
    """
    # ratio = 1e-2  # 2 orders are suggested by Simon following a Lee's manuscript
    ratio = 1e-14  # otherwise it `lands in Bolivia'
    st_dev = ratio * GM
    q = st_dev ** 2
    q_xdot, q_ydot, q_zdot = q, q, q
    return np.array([q_xdot, q_ydot, q_zdot])


def get_initial_states(n_targets=None, population_mean=None, population_covariance=None, start_time=None, **kwargs):
    """We sample n_targets initial states from the Gaussian distribution"""
    ground_truth_states = []
    for _ in range(n_targets):
        if population_covariance.any():
            sampled_deviation = np.linalg.cholesky(population_covariance).T @ np.random.normal(size=population_mean.shape)
            state_vector_init = population_mean + sampled_deviation
        else:
            state_vector_init = population_mean
        ground_truth_states.append(GroundTruthState(state_vector=state_vector_init, timestamp=start_time))
    return ground_truth_states


def get_groundtruth_path(initial_target_state=None, transition_model=None, timesteps=None, noise=False):
    successive_time_steps = timesteps[1:]  # dropping the very first start_time
    truth = GroundTruthPath(initial_target_state)
    for timestamp in successive_time_steps:
        interval = timestamp - truth[-1].timestamp
        truth.append(
            GroundTruthState(transition_model.function(state=truth[-1], noise=noise, time_interval=interval),
                             timestamp=timestamp)
        )
    return truth


def get_groundtruth_paths(initial_target_states=None, transition_model=None, timesteps=None, noise=False):
    groundtruth_paths = OrderedSet()
    for initial_target_state in initial_target_states:
        truth = get_groundtruth_path(initial_target_state=initial_target_state,
                                     transition_model=transition_model,
                                     timesteps=timesteps,
                                     noise=noise)
        groundtruth_paths.add(truth)
    return groundtruth_paths


def get_priors(initial_states, target_initial_covariance, timestamp):
    target_priors = []
    for initial_state in initial_states:
        mean = initial_state.state_vector
        bias = np.matrix([-22068.01433784, 69.37315652, 507.76211348, -86.74038986, -58321.63970861, 89.04789997]).T
        # bias = np.linalg.cholesky(P_0).T @ np.random.normal(size=target_state_0.shape)  # GroundTruth+chol(P0)’*randn
        prior_pdf = GaussianStatePrediction(mean + bias, target_initial_covariance, timestamp=timestamp)
        target_priors.append(prior_pdf)
    return target_priors


def get_observation_histories(truths, timesteps, measurement_model, sensor_parameters, n_mc_runs):
    """ Generates n_mc_runs observation histories. Here, a single observation is a set of measurements."""

    def get_observation_history(truths, timesteps, measurement_model, sensor_parameters):
        """ Within a single Monte Carlo run it follows the clutter generation as in
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

    observation_histories = []
    for _ in range(n_mc_runs):
        observation_history = get_observation_history(truths, timesteps, measurement_model, sensor_parameters)
        observation_histories.append(observation_history)

    return observation_histories


def do_JPDA(observation_history, data_associator):

    tracks = set()
    for prior_pdf in priors:
        tracks.add(Track([prior_pdf]))

    try:
        assert len(timesteps) == len(observation_history)  # check if the following for loop makes sense
    except AssertionError:
        print('Cardinalities should match. Possibly, no observations collected at one of the time steps.')

    for timestep, observation in zip(timesteps, observation_history):
        # NB: Observation is understood here as a set of measurements and false alarms.
        hypotheses = data_associator.associate(
            tracks, observation, timestep, allow_singular=True
        )

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


if __name__ == "__main__":

    start_time = datetime(2000, 1, 1)
    np.random.seed(1991)

    # We begin by specifying the mean state of a target population by picking a credible set of Keplerian elements, and
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

    population_covariance = np.diag([150000 ** 2, 100 ** 2, 150000 ** 2, 100 ** 2, 150000 ** 2, 100 ** 2])
    target_initial_covariance = np.diag([50000 ** 2, 100 ** 2, 50000 ** 2, 100 ** 2, 50000 ** 2, 100 ** 2])

    scenario_parameters = {
        'n_mc_runs': 1,
        'time_interval': timedelta(seconds=50),
        'n_time_steps': 10,
        'n_targets': 5,
        'population_mean': population_mean,
        'population_covariance': population_covariance,
        'target_initial_covariance': target_initial_covariance
    }

    initial_states = get_initial_states(
        n_targets=scenario_parameters['n_targets'],
        population_mean=scenario_parameters['population_mean'],
        population_covariance=scenario_parameters['population_covariance'],
        start_time=start_time
    )

    priors = get_priors(initial_states, scenario_parameters['target_initial_covariance'], start_time)  # for tracking
    timesteps = [start_time + k * scenario_parameters['time_interval'] for k in range(scenario_parameters['n_time_steps'])]
    transition_model = LinearisedDiscretisation(
        diff_equation=twoBody3d_da,
        linear_noise_coeffs=get_noise_coefficients(GM)
    )
    truths = get_groundtruth_paths(
        initial_target_states=initial_states,
        transition_model=transition_model,
        timesteps=timesteps,
        noise=True
    )

    # Generate measurements
    sigma_el, sigma_b, sigma_range = np.deg2rad(0.01), np.deg2rad(0.01), 100
    sensor_x, sensor_y, sensor_z = 0, 0, 0
    sensor_parameters = {
        'prob_detect': 0.9,
        'clutter_rate': 3,
        'clutter_spatial_density': 0.125 * 0.00001,
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
    observation_histories = get_observation_histories(
        truths, timesteps, measurement_model, sensor_parameters, scenario_parameters['n_mc_runs']
    )

    # Put together a filter
    predictor = ExtendedKalmanPredictorROBUSSTOD(transition_model)
    updater = IPLFKalmanUpdater()

    hypothesiser = PDAHypothesiser(
        predictor=predictor,
        updater=updater,
        clutter_spatial_density=sensor_parameters['clutter_spatial_density'],
        prob_detect=sensor_parameters['prob_detect']
    )
    data_associator = JPDA(hypothesiser=hypothesiser)

    tracks_JPDA_list = []
    for observation_history in observation_histories:
        tracks_JPDA = do_JPDA(observation_history, data_associator)
        tracks_JPDA_list.append(tracks_JPDA)

    plotter = Plotterly()
    mc_run_to_plot = 0
    plotter.plot_ground_truths(truths, [0, 2], line=dict(dash="dash", color='black'))
    plotter.plot_measurements(observation_histories[mc_run_to_plot], [0, 2])
    plotter.plot_tracks(tracks_JPDA_list[mc_run_to_plot], [0, 2], uncertainty=True)
    plotter.fig.show()
