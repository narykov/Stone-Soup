import numpy as np

from stonesoup.types.detection import TrueDetection, Clutter
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.prediction import GaussianStatePrediction
from stonesoup.types.state import State
from scipy.stats import uniform
from ordered_set import OrderedSet


def get_initial_state(state_vector=None, start_time=None):
    initial_state = GroundTruthState(state_vector=state_vector, timestamp=start_time)
    return initial_state


def get_initial_states(n_targets=None, population_mean=None, population_covariance=None, start_time=None, **kwargs):
    """We sample n_targets initial states from the Gaussian distribution"""
    ground_truth_states = []
    for _ in range(n_targets):
        sampled_deviation = np.linalg.cholesky(population_covariance).T @ np.random.normal(size=population_mean.shape)
        state_vector = population_mean + sampled_deviation
        state_vector_init = get_initial_state(state_vector=state_vector, start_time=start_time)
        ground_truth_states.append(state_vector_init)
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


def get_prior(initial_state, target_initial_covariance, timestamp, bias_fixed=True):
    mean = initial_state.state_vector
    if bias_fixed:
        bias = np.matrix([-22068.01433784, 69.37315652, 507.76211348, -86.74038986, -58321.63970861, 89.04789997]).T
    else:
        bias = np.linalg.cholesky(target_initial_covariance).T @ np.random.normal(size=mean.shape)  # GroundTruth+chol(P0)’*randn

    prior_pdf = GaussianStatePrediction(mean + bias, target_initial_covariance, timestamp=timestamp)
    return prior_pdf

def get_priors(initial_states, target_initial_covariance, timestamp):
    target_priors = []
    for initial_state in initial_states:
        prior_pdf = get_prior(initial_state, target_initial_covariance, timestamp)
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


def get_measurement_history(truth=None, measurement_model=None):
    """ Generates a measurement history. """
    measurement_history = []
    for state in truth:
        measurement = measurement_model.function(state, noise=True)
        measurement_history.append(
            TrueDetection(measurement, timestamp=state.timestamp, groundtruth_path=truth, measurement_model=measurement_model)
        )
    return measurement_history


def get_measurement_histories(truth=None, measurement_model=None, n_mc_runs=1):
    """ Generates n_mc_runs measurement histories. """
    measurement_histories = []
    for _ in range(n_mc_runs):
        measurement_history = get_measurement_history(truth=truth, measurement_model=measurement_model)
        measurement_histories.append(measurement_history)
    return measurement_histories
