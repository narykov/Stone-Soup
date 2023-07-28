import numpy as np
import torch

from stonesoup.types.detection import TrueDetection, Clutter
from stonesoup.types.array import StateVectors

from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.track import Track

from stonesoup.types.update import GaussianStateUpdate
from stonesoup.types.prediction import GaussianStatePrediction
from stonesoup.types.state import State
from scipy.stats import uniform

from stonesoup.functions import gm_reduce_single
from ordered_set import OrderedSet


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


def do_JPDA(priors, timesteps, observation_history, data_associator):

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


def get_measurement_history(truth=None, measurement_model=None):
    """ Generates a measurement history. """
    measurement_history = []
    for state in truth:
        measurement = measurement_model.function(state, noise=True)
        measurement_history.append(
            TrueDetection(measurement, timestamp=state.timestamp, measurement_model=measurement_model)
        )
    return measurement_history


def get_measurement_histories(truth=None, measurement_model=None, n_mc_runs=1):
    """ Generates n_mc_runs measurement histories. """
    measurement_histories = []
    for _ in range(n_mc_runs):
        measurement_history = get_measurement_history(truth, timesteps, measurement_model, sensor_parameters)
        measurement_histories.append(measurement_history)
    return measurement_histories
