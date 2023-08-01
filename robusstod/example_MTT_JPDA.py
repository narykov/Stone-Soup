#!/usr/bin/env python
import warnings

warnings.simplefilter('always', UserWarning)

import numpy as np
# import copy

from datetime import datetime, timedelta
from stonesoup.predictor.kalman import ExtendedKalmanPredictorROBUSSTOD
from stonesoup.plotter import Plotterly
from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRange
from stonesoup.types.track import Track
from stonesoup.functions import gm_reduce_single
from stonesoup.types.array import StateVectors
from stonesoup.types.update import GaussianStateUpdate

from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.dataassociator.probability import JPDA


# NEW CLASSES:
from stonesoup.updater.kalman import IPLFKalmanUpdater
from stonesoup.models.transition.nonlinear import LinearisedDiscretisation


from stonesoup.robusstod.utils import get_initial_states
from stonesoup.robusstod.utils import get_groundtruth_paths
from stonesoup.robusstod.utils import get_priors
from stonesoup.robusstod.utils import get_observation_histories

from stonesoup.robusstod.physics.constants import G, M_earth
from stonesoup.robusstod.physics.other import get_noise_coefficients

use_godot = False
if not use_godot:
    from stonesoup.robusstod.physics.godot import KeplerianToCartesian
    from stonesoup.robusstod.physics.godot import twoBody3d_da
else:
    from stonesoup.robusstod.physics.basic import KeplerianToCartesian
    from stonesoup.robusstod.physics.basic import twoBody3d_da


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


def main():
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
        tracks_JPDA = do_JPDA(priors, timesteps, observation_history, data_associator)
        tracks_JPDA_list.append(tracks_JPDA)

    plotter = Plotterly()
    mc_run_to_plot = 0
    plotter.plot_ground_truths(truths, [0, 2], line=dict(dash="dash", color='black'))
    plotter.plot_measurements(observation_histories[mc_run_to_plot], [0, 2])
    plotter.plot_tracks(tracks_JPDA_list[mc_run_to_plot], [0, 2], uncertainty=True)
    plotter.fig.show()


if __name__ == "__main__":
    main()
