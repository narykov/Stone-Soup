#!/usr/bin/env python
import warnings

warnings.simplefilter('always', UserWarning)

import numpy as np
import copy

from datetime import datetime, timedelta
from stonesoup.types.detection import Detection, TrueDetection, Clutter
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.predictor.kalman import UnscentedKalmanPredictor, KalmanPredictor, ExtendedKalmanPredictor, ExtendedKalmanPredictorROBUSSTOD
from stonesoup.updater.kalman import UnscentedKalmanUpdater, IPLFKalmanUpdater, IteratedKalmanUpdater
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.transition.nonlinear import GaussianTransitionModel
from stonesoup.models.transition.linear import LinearGaussianTransitionModel
from stonesoup.models.transition.nonlinear import GaussianTransitionModel
from stonesoup.models.base import TimeVariantModel
from stonesoup.types.array import StateVector, StateVectors, CovarianceMatrix

from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

from stonesoup.types.state import GaussianState
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track
from stonesoup.plotter import Plotterly

from stonesoup.base import Property
from stonesoup.measures import Measure, GaussianKullbackLeiblerDivergence
from astropy.constants import G, M_earth, R_earth
from scipy.linalg import expm, block_diag, inv
import torch

from stonesoup.smoother.kalman import UnscentedKalmanSmoother, KalmanSmoother
from stonesoup.types.detection import Detection
from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRange

from stonesoup.types.update import GaussianStateUpdate
from stonesoup.types.prediction import GaussianStatePrediction
from stonesoup.types.array import StateVector
from stonesoup.types.array import CovarianceMatrix
from scipy.stats import multivariate_normal
from stonesoup.measures import Euclidean
from stonesoup.types.time import TimeRange
from stonesoup.types.metric import SingleTimeMetric, TimeRangeMetric
from matplotlib import pyplot as plt
from stonesoup.types.state import State
from stonesoup.functions import jacobian as compute_jac
from scipy.stats import uniform

from stonesoup.functions import gm_reduce_single
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.dataassociator.probability import PDA

from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.dataassociator.neighbour import NearestNeighbour
from ordered_set import OrderedSet
from stonesoup.dataassociator.probability import JPDA


# LOCAL:
from stonesoup.updater.kalman import IPLFKalmanUpdater  # depends on local GaussianKullbackLeiblerDivergence
from stonesoup.models.measurement.linear import GeneralLinearGaussian
from stonesoup.models.transition.nonlinear import LinearisedDiscretisation
from stonesoup.updater.kalman import KalmanUpdater


# class LinearisedDiscretisation(GaussianTransitionModel, TimeVariantModel):
#     linear_noise_coeffs: np.ndarray = Property(
#         doc=r"The acceleration noise diffusion coefficients :math:`[q_x, \: q_y, \: q_z]^T`")
#     diff_equation: staticmethod = Property(doc=r"Differential equation describing the movement")
#
#     @property
#     def ndim_state(self):
#         """ndim_state getter method
#
#         Returns
#         -------
#         : :class:`int`
#             The number of combined model state dimensions.
#         """
#         return 6
#
#     def doLinearise(self, da, dQ, x, dt):
#
#         def getJacobian(f, state):
#             nx = len(state)
#             A = np.zeros([nx, nx])
#             state_input = [i for i in state]
#
#             jacrows = torch.autograd.functional.jacobian(f, torch.tensor(state_input))
#             for i, r in enumerate(jacrows):
#                 A[i] = r
#
#             return (A)
#
#         dA = getJacobian(da, x)
#         A = expm(dA * dt)
#         nx = len(x)
#
#         # Get \int e^{dA*s}\,ds
#         int_eA = expm(dt * np.block([[dA, np.identity(nx)], [np.zeros([nx, 2 * nx])]]))[:nx, nx:]
#
#         # Get Q
#         G = expm(dt * np.block([[-dA, dQ], [np.zeros([nx, nx]), np.transpose(dA)]]))
#         Q = np.transpose(G[nx:, nx:]) @ (G[:nx, nx:])
#         Q = (Q + np.transpose(Q)) / 2.
#
#         # Get new value of x
#         x = [i for i in x]
#         try:
#             newx = x + int_eA @ da(torch.tensor(x))
#             print('x')
#         except:
#             newx = torch.tensor(x) + int_eA @ da(torch.tensor(x))
#             print('except')
#         return newx, A, Q
#
#
#     # def matrix(self, time_interval, **kwargs):
#     #
#     #     def getJacobian(f, state):
#     #         nx = len(state)
#     #         A = np.zeros([nx, nx])
#     #         state_input = [i for i in state]
#     #
#     #         jacrows = torch.autograd.functional.jacobian(f, torch.tensor(state_input))
#     #         for i, r in enumerate(jacrows):
#     #             A[i] = r
#     #
#     #         return (A)
#     #
#     #     da = self.diff_equation
#     #     if 'state_copy' not in kwargs:
#     #         state = prior.state_vector
#     #     else:
#     #         state = kwargs['state_copy'].state_vector
#     #
#     #     print(prior.state_vector)
#     #
#     #     dA = getJacobian(da, state)
#     #     dt = time_interval.total_seconds()
#     #     A = expm(dA * dt)
#     #
#     #     return A
#
#     def jacobian(self, state, **kwargs):
#
#         def getJacobian(f, x):
#             state = x.state_vector
#             nx = len(state)
#             A = np.zeros([nx, nx])
#             state_input = [i for i in state]
#
#             jacrows = torch.autograd.functional.jacobian(f, torch.tensor(state_input))
#             for i, r in enumerate(jacrows):
#                 A[i] = r
#
#             return (A)
#
#         time_interval = kwargs['time_interval']
#         da = self.diff_equation
#
#         dA = getJacobian(da, state)
#         dt = time_interval.total_seconds()
#         A = expm(dA * dt)
#
#         return A
#
#     def function(self, state, noise=False, **kwargs) -> StateVector:
#         """Model linear function :math:`f_k(x(k),w(k)) = F_k(x_k) + w_k`
#
#         Parameters
#         ----------
#         state: :class:`~.State`
#             An input state
#         noise: :class:`numpy.ndarray` or bool
#             An externally generated random process noise sample (the default is
#             `False`, in which case no noise will be added
#             if 'True', the output of :meth:`~.Model.rvs` is added)
#
#         Returns
#         -------
#         : :class:`State`
#             The updated State with the model function evaluated.
#         """
#
#         if isinstance(noise, bool) or noise is None:
#             if noise:
#                 noise = self.rvs(**kwargs)
#             else:
#                 noise = 0
#
#         # time_interval = kwargs['time_interval'].total_seconds()
#
#         return self.jacobian(state, **kwargs) @ state.state_vector + noise
#
#     def covar(self, time_interval, **kwargs):
#
#         # if 'state_copy' not in kwargs:
#         #     breakpoint()
#
#         dt = time_interval.total_seconds()
#         # if 'state_copy' not in kwargs:
#         #     # breakpoint()
#         #     sv1 = prior.state_vector
#         # else:
#         #     sv1 = kwargs['state_copy'].state_vector
#         sv1 = kwargs['prior'].state_vector
#         da = self.diff_equation
#         q_xdot, q_ydot, q_zdot = self.linear_noise_coeffs
#         dQ = np.diag([0., q_xdot, 0., q_ydot, 0., q_zdot])
#         _, _, C = self.doLinearise(da, dQ, sv1, dt)
#
#         return CovarianceMatrix(C)



# class FromForce(LinearGaussianTransitionModel, TimeVariantModel):
#     linear_noise_coeffs: np.ndarray = Property(
#         doc=r"The acceleration noise diffusion coefficients :math:`[q_x, \: q_y, \: q_z]^T`")
#     diff_equation: staticmethod = Property(doc=r"Differential equation describing the movement")
#
#     @property
#     def ndim_state(self):
#         """ndim_state getter method
#
#         Returns
#         -------
#         : :class:`int`
#             The number of combined model state dimensions.
#         """
#         return 6
#
#
#     def matrix(self, time_interval, **kwargs):
#
#         if 'state_copy' not in kwargs:
#             state = prior.state_vector
#         else:
#             state = kwargs['state_copy'].state_vector
#
#         dA = self.jacobian(state)
#         dt = time_interval.total_seconds()
#         A = expm(dA * dt)
#
#         return A
#
#     def covar(self, time_interval, **kwargs):
#
#         def doLinearise(da, dQ, x, dt):
#             dA = self.jacobian(x)
#             A = expm(dA * dt)
#             nx = len(x)
#
#             # Get \int e^{dA*s}\,ds
#             int_eA = expm(dt * np.block([[dA, np.identity(nx)], [np.zeros([nx, 2 * nx])]]))[:nx, nx:]
#
#             # Get Q
#             G = expm(dt * np.block([[-dA, dQ], [np.zeros([nx, nx]), np.transpose(dA)]]))
#             Q = np.transpose(G[nx:, nx:]) @ (G[:nx, nx:])
#             Q = (Q + np.transpose(Q)) / 2.
#
#             # Get new value of x
#             x = [i for i in x]
#             try:
#                 newx = x + int_eA @ da(torch.tensor(x))
#                 print('x')
#             except:
#                 newx = torch.tensor(x) + int_eA @ da(torch.tensor(x))
#                 print('except')
#
#             return newx, A, Q
#
#         dt = time_interval.total_seconds()
#
#         if 'state_copy' not in kwargs:
#             # breakpoint()
#             sv1 = prior.state_vector
#         else:
#             sv1 = kwargs['state_copy'].state_vector
#
#         da = self.diff_equation
#         q_xdot, q_ydot, q_zdot = self.linear_noise_coeffs
#         dQ = np.diag([0., q_xdot, 0., q_ydot, 0., q_zdot])
#         _, _, C = doLinearise(da, dQ, sv1, dt)
#
#         return CovarianceMatrix(C)
#
#     def jacobian(self, state, **kwargs):
#         f = self.diff_equation
#         nx = len(state)
#         A = np.zeros([nx, nx])
#         state_input = [i for i in state]
#
#         jacrows = torch.autograd.functional.jacobian(f, torch.tensor(state_input))
#         for i, r in enumerate(jacrows):
#             A[i] = r
#
#         out = (A)
#
#         return out


def KeplerianToCartesian(K, E):
    # elements need to be in radians

    GM = G.value * M_earth.value

    a, e, i, w, omega, nu = K[:]

    x = a * (np.cos(E) - e)
    y = a * np.sqrt(1 - e * e) * np.sin(E)

    P = np.array([np.cos(omega) * np.cos(w) - np.sin(omega) * np.cos(i) * np.sin(w),
                  np.sin(omega) * np.cos(w) + np.cos(omega) * np.cos(i) * np.sin(w), np.sin(i) * np.sin(w)])
    Q = np.array([-np.cos(omega) * np.sin(w) - np.sin(omega) * np.cos(i) * np.cos(w),
                  np.cos(omega) * np.cos(i) * np.cos(w) - np.sin(omega) * np.sin(w), np.sin(i) * np.cos(w)])

    [X, Y, Z] = x * P + y * Q

    r = a * (1 - e * np.cos(E))

    x_dot = -((np.sqrt(a * GM)) * np.sin(E)) / float(r)
    y_dot = ((np.sqrt(a * GM)) * np.sqrt(1 - e * e) * np.cos(E)) / float(r)

    [U, V, W] = x_dot * P + y_dot * Q

    # p = [X, Y, Z, U, V, W]
    p = [X, U, Y, V, Z, W]

    return p


def twoBody3d_da(state):
    GM = G.value * M_earth.value
    try:
        (x, x_dot, y, y_dot, z, z_dot) = state
    except:
        breakpoint()
    r_pow_3 = torch.float_power(torch.linalg.vector_norm(torch.tensor((x, y, z))), 3)
    return (x_dot, -GM * x / r_pow_3, y_dot, -GM * y / r_pow_3, z_dot, -GM * z / r_pow_3)


# def doNN(all_measurements, transition_model, prior, updater):
#     """FILTER + FALSE ALARMS"""
#     predictor = KalmanPredictor(transition_model)
#     hypothesiser = DistanceHypothesiser(predictor, updater, measure=Mahalanobis(), missed_distance=5)
#     data_associator = NearestNeighbour(hypothesiser)
#
#     track = Track([prior])
#     for n, measurements in enumerate(all_measurements):
#         try:
#             timestamp = next(iter(measurements)).timestamp  # arbitary measurement at this time step (due to using sets)
#         except:
#             breakpoint()
#         hypotheses = data_associator.associate([track],
#                                                measurements,
#                                                timestamp)
#         hypothesis = hypotheses[track]
#
#         if hypothesis.measurement:
#             post = updater.update(hypothesis)
#             track.append(post)
#         else:  # When data associator says no detections are good enough, we'll keep the prediction
#             track.append(hypothesis.prediction)
#
#     del track[0]  # track = track[1:]
#
#     return track
#
#
# def doPDA(all_measurements, transition_model, prior, updater):
#     """FILTER + PDA"""
#
#     predictor = KalmanPredictor(transition_model)
#     # updater = IteratedKalmanUpdater(max_iterations=5)
#     # updater = IPLFKalmanUpdater()
#     hypothesiser = PDAHypothesiser(predictor=predictor,
#                                    updater=updater,
#                                    clutter_spatial_density=0.125 * 0.00001,
#                                    prob_detect=prob_det)
#
#     data_associator = PDA(hypothesiser=hypothesiser)
#
#     track = Track([prior])
#     for n, measurements in enumerate(all_measurements):
#         timestamp = next(iter(measurements)).timestamp
#         hypotheses = data_associator.associate([track],
#                                                measurements,
#                                                timestamp)
#
#         hypotheses = hypotheses[track]
#
#         # Loop through each hypothesis, creating posterior states for each, and merge to calculate
#         # approximation to actual posterior state mean and covariance.
#         posterior_states = []
#         posterior_state_weights = []
#         for hypothesis in hypotheses:
#             if not hypothesis:
#                 posterior_states.append(hypothesis.prediction)
#             else:
#                 posterior_state = updater.update(hypothesis)
#                 posterior_states.append(posterior_state)
#             posterior_state_weights.append(
#                 hypothesis.probability)
#
#         means = StateVectors([state.state_vector for state in posterior_states])
#         covars = np.stack([state.covar for state in posterior_states], axis=2)
#         weights = np.asarray(posterior_state_weights)
#
#         # Reduce mixture of states to one posterior estimate Gaussian.
#         post_mean, post_covar = gm_reduce_single(means, covars, weights)
#
#         # Add a Gaussian state approximation to the track.
#         track.append(GaussianStateUpdate(
#             post_mean, post_covar,
#             hypotheses,
#             hypotheses[0].measurement.timestamp))
#
#
#     del track[0]  # track = track[1:]
#
#     return track
#
#
# def doIPLS(track_iplf, transition_model, measurement_model, n_iterations=0):
#     smoother_ukf = UnscentedKalmanSmoother(transition_model)
#     track_smoothed = smoother_ukf.smooth(track_iplf)
#
#     if n_iterations >= 0:
#         predictor = KalmanPredictor(transition_model)
#         updater = IPLFKalmanUpdater()  # to use its slr computations
#         updater_kf = KalmanUpdater()  # to do forward go in the RTS smoother
#         smoother_kf = KalmanSmoother(transition_model)  # to do backward go in the RTS smoother
#
#         for n in range(n_iterations):
#
#             track_forward = Track()
#
#             for smoothed_update in track_smoothed:
#
#                 # Get prior/predicted pdf
#                 timestamp = smoothed_update.timestamp  # check if first iteration
#
#                 # Get linearization wrt a smoothed posterior
#                 slr = updater.slr_calculations(smoothed_update, measurement_model)
#                 measurement_model_linearized = GeneralLinearGaussian(
#                     ndim_state=measurement_model.ndim_state,
#                     mapping=measurement_model.mapping,
#                     meas_matrix=slr['A_l'],
#                     bias_value=slr['b_l'],
#                     noise_covar=measurement_model.noise_covar + slr['Omega_l'])
#
#                 if timestamp == track_smoothed[0].timestamp:
#                     # prediction = smoothed_update.hypothesis.prediction  # get original prior from hypothesis
#                     prediction = prior
#                 else:
#                     prediction = predictor.predict(prev_state, timestamp=timestamp)  # calculate prior from previous update
#
#
#                 # if isinstance(smoothed_update, GaussianStatePrediction):
#                 #     print('Used prediction!')
#                 #     update = smoothed_update
#                 # else:
#                 if True:
#                     if isinstance(smoothed_update.hypothesis, SingleHypothesis):
#                         # # Get the measurement plus its prediction for the above model using the predicted pdf
#                         # measurement = smoothed_update.hypothesis.measurement
#                         # print('Hello')
#                         # measurement_prediction = updater_kf.predict_measurement(predicted_state=prediction,
#                         #                                                         measurement_model=measurement_model_linearized)
#                         #
#                         # hypothesis = SingleHypothesis(prediction=prediction,
#                         #                               measurement=measurement,
#                         #                               measurement_prediction=measurement_prediction)
#                         #
#                         # update = updater_kf.update(hypothesis)
#                         breakpoint()
#                     else:
#                         posterior_states = []
#                         posterior_state_weights = []
#                         # hypotheses = smoothed_update.hypothesis
#
#                         measurements_pda = set()
#                         for hypothesis in smoothed_update.hypothesis:
#                             if hypothesis.measurement.measurement_model is not None:
#                                 measurement = copy.copy(hypothesis.measurement)
#                                 measurement.measurement_model = measurement_model_linearized
#                                 measurements_pda.add(measurement)
#
#                         hypothesiser = PDAHypothesiser(predictor=predictor,
#                                                        updater=updater_kf,
#                                                        clutter_spatial_density=0.125 * 0.00001,
#                                                        prob_detect=1)
#                         data_associator = PDA(hypothesiser=hypothesiser)
#
#                         if len(track_forward) == 0:
#                             track_forward = Track([prediction])
#
#                         hypotheses = data_associator.associate({track_forward},
#                                                                measurements_pda,
#                                                                timestamp)
#
#                         hypotheses = hypotheses[track_forward]
#
#                         for hypothesis in hypotheses:
#                             if not hypothesis:
#                                 posterior_states.append(hypothesis.prediction)
#                             else:
#                                 posterior_state = updater_kf.update(hypothesis)
#                                 posterior_states.append(posterior_state)
#                             posterior_state_weights.append(
#                                 hypothesis.probability)
#
#                         means = StateVectors([state.state_vector for state in posterior_states])
#                         covars = np.stack([state.covar for state in posterior_states], axis=2)
#                         weights = np.asarray(posterior_state_weights)
#
#                         # Reduce mixture of states to one posterior estimate Gaussian.
#                         post_mean, post_covar = gm_reduce_single(means, covars, weights)
#
#                         # Add a Gaussian state approximation to the track.
#                         update = GaussianStateUpdate(
#                             post_mean, post_covar,
#                             hypotheses,
#                             hypotheses[0].measurement.timestamp)
#
#                 track_forward.append(update)
#                 prev_state = update
#             del track_forward[0]
#             track_smoothed = smoother_kf.smooth(track_forward)
#
#     return track_smoothed


# start_time = datetime.now().replace(microsecond=0)
start_time = datetime(2000, 1, 1)
np.random.seed(1991)

"""
Setting up space ground truth
"""
# from physical considerations
GM = G.value * M_earth.value  # 4e14
# ratio = 1e-2  # 2 orders are suggested by Simon following a Lee's manuscript
ratio = 1e-14  # otherwise it `lands in Bolivia'
st_dev = ratio * GM
q = st_dev ** 2
q_xdot, q_ydot, q_zdot = q, q, q

linear_noise_coeffs = np.array([q_xdot, q_ydot, q_zdot])  # np.ndarray() of [q_x,q_y]^T
transition_model = LinearisedDiscretisation(linear_noise_coeffs=linear_noise_coeffs,
                                            diff_equation=twoBody3d_da)

a, e, i, w, Omega, nu = 9164000, 0.03, 70, 0, 0, np.radians(0)  # keplerian elements (a is in metres)
# a, e, i, w, Omega, nu = 7000, 0, 0, 0, 0, np.radians(0)  # from Alejandro
K = [a, e, np.radians(i), np.radians(w), np.radians(Omega), nu]
E = np.arctan2(np.sqrt(1 - e * e) * np.sin(nu), e + np.cos(nu))
# Next, we convert these Keplerian elements to Catertesian coordinates (using Gemma's approach)
target_state_0 = np.array(KeplerianToCartesian(K, E))



time_interval = timedelta(seconds=10)
ntimesteps = 5
n_mc_runs = 1
prob_det = 1
clutter_rate = 3
n_targets = 5
# n_targets = 1

P_b = np.diag([150000 ** 2, 100 ** 2, 150000 ** 2, 100 ** 2, 150000 ** 2, 100 ** 2])

initial_states = []
for _ in range(n_targets):
    sampled_deviation = np.linalg.cholesky(P_b).T @ np.random.normal(size=target_state_0.shape)
    state_vector_init = target_state_0+sampled_deviation
    initial_states.append(GroundTruthState(state_vector=state_vector_init,
                                           timestamp=start_time))

# """ New ground truth simulations"""
# P_b = np.diag([1000000 ** 2, 100 ** 2, 1000000 ** 2, 100 ** 2, 1000000 ** 2, 100 ** 2])
# a, e, i, w, Omega, nu = 9164000, 0.03, 70, 0, 0, np.radians(0)  # keplerian elements (a is in metres)
# scale = 1000000
# initial_states = []
# for _ in range(n_targets):
#     a_adjusted = np.random.normal(loc=a, scale=scale)
#     # a, e, i, w, Omega, nu = 7000, 0, 0, 0, 0, np.radians(0)  # from Alejandro
#     K = [a_adjusted, e, np.radians(i), np.radians(w), np.radians(Omega), nu]
#     E = np.arctan2(np.sqrt(1 - e * e) * np.sin(nu), e + np.cos(nu))
#     # Next, we convert these Keplerian elements to Catertesian coordinates (using Gemma's approach)
#     target_state_0 = np.array(KeplerianToCartesian(K, E))
#     initial_states.append(GroundTruthState(state_vector=target_state_0,
#                                            timestamp=start_time))

timesteps = [start_time + k * time_interval for k in range(ntimesteps)]

truths = OrderedSet()
for initial_state in initial_states:
    truth = GroundTruthPath(initial_state)
    for timestamp in timesteps[1:]:
        # timesteps[1:] above is to exclude the initial state
        truth.append(GroundTruthState(
            transition_model.function(state=truth[-1], noise=False, time_interval=time_interval), timestamp=timestamp))
    truths.add(truth)


print()
""" END """

# timestamp = start_time
# timesteps = [start_time]
# truth = GroundTruthPath([GroundTruthState(target_state_0, timestamp=timesteps[0])])
# for _ in range(ntimesteps):
#     timestamp += time_interval
#     timesteps.append(timestamp)
#     truth.append(GroundTruthState(
#         transition_model.function(state=truth[-1], noise=False, time_interval=time_interval, state_copy=truth[-1]),
#         timestamp=timestamp))

"""Priors for tracking"""
priors = []
P_0 = np.diag([50000 ** 2, 100 ** 2, 50000 ** 2, 100 ** 2, 50000 ** 2, 100 ** 2])
for initial_state in initial_states:
    mean = initial_state.state_vector
    bias = np.matrix([-22068.01433784, 69.37315652, 507.76211348, -86.74038986, -58321.63970861, 89.04789997]).T
    # bias = np.linalg.cholesky(P_0).T @ np.random.normal(size=target_state_0.shape)  # GroundTruth+chol(P0)’*randn
    prior_pdf = GaussianStatePrediction(mean + bias, P_0, timestamp=timesteps[0])
    priors.append(prior_pdf)


"""
MEASUREMENTS
"""

# Generate measurements.
all_measurements = []

sensor_x, sensor_y, sensor_z = 0, 0, 0
measurement_model = CartesianToElevationBearingRange(
    ndim_state=6,
    mapping=(0, 2, 4),
    noise_covar=np.diag([np.deg2rad(0.01)**2, np.deg2rad(0.01)**2, 100**2]),
    translation_offset=np.array([[sensor_x], [sensor_y], [sensor_z]]))

prob_detect = 1  # 90% chance of detection.

measurements_sets = []
for _ in range(n_mc_runs):
    all_measurements = []
    for k, timestamp in enumerate(timesteps):
        measurement_set = set()

        for truth in truths:
            # Generate actual detection from the state with a 10% chance that no detection is received.
            if np.random.rand() <= prob_detect:
                measurement = measurement_model.function(truth[k], noise=True)
                measurement_set.add(TrueDetection(state_vector=measurement,
                                                  groundtruth_path=truth,
                                                  timestamp=truth[k].timestamp,
                                                  measurement_model=measurement_model))

            # Generate clutter at this time-step
            truth_x = truth[k].state_vector[0]
            truth_y = truth[k].state_vector[2]
            truth_z = truth[k].state_vector[4]
            value = np.random.randint(clutter_rate)
            # value = 0
            for _ in range(value):
                width = 100000
                x = uniform.rvs(truth_x - 0.5 * width, width)
                y = uniform.rvs(truth_y - 0.5 * width, width)
                z = uniform.rvs(truth_z - 0.5 * width, width)
                clutter_state = State(state_vector=np.array([x, np.nan, y, np.nan, z, np.nan]),
                                      timestamp=timestamp)
                clutter_measurement = measurement_model.function(clutter_state,
                                                                 noise=False)
                clutter_detection = Clutter(state_vector=clutter_measurement,
                                            timestamp=timestamp,
                                            measurement_model=measurement_model)
                measurement_set.add(clutter_detection)

        all_measurements.append(measurement_set)

    measurements_sets.append(all_measurements)





"""
MEASUREMENTS END
"""


# predictor = KalmanPredictor(transition_model)
predictor = ExtendedKalmanPredictorROBUSSTOD(transition_model)

updater = IPLFKalmanUpdater()

hypothesiser = PDAHypothesiser(predictor=predictor,
                               updater=updater,
                               clutter_spatial_density=0.125 * 0.00001,
                               prob_detect=prob_detect)

data_associator = JPDA(hypothesiser=hypothesiser)

for all_measurements in measurements_sets:
    tracks = set()
    for prior_pdf in priors:
        tracks.add(Track([prior_pdf]))

    for n, measurements in enumerate(all_measurements):
        hypotheses = data_associator.associate(tracks,
                                               measurements,
                                               start_time + n * time_interval)

        # Loop through each track, performing the association step with weights adjusted according to
        # JPDA.
        for track in tracks:
            track_hypotheses = hypotheses[track]

            posterior_states = []
            posterior_state_weights = []
            for hypothesis in track_hypotheses:
                if not hypothesis:
                    posterior_states.append(hypothesis.prediction)
                else:
                    posterior_state = updater.update(hypothesis)
                    posterior_states.append(posterior_state)
                posterior_state_weights.append(hypothesis.probability)

            means = StateVectors([state.state_vector for state in posterior_states])
            covars = np.stack([state.covar for state in posterior_states], axis=2)
            weights = np.asarray(posterior_state_weights)

            # Reduce mixture of states to one posterior estimate Gaussian.
            post_mean, post_covar = gm_reduce_single(means, covars, weights)

            # Add a Gaussian state approximation to the track.
            track.append(GaussianStateUpdate(
                post_mean, post_covar,
                track_hypotheses,
                track_hypotheses[0].measurement.timestamp))


plotter = Plotterly()
plotter.plot_ground_truths(truths, [0, 2], line=dict(dash="dash", color='black'))
plotter.plot_measurements(all_measurements, [0, 2])
plotter.plot_tracks(tracks, [0, 2], uncertainty=True)
plotter.fig.show()


""" OLD """
iplf_outs = np.zeros((1, ntimesteps+1))
ipls_outs = np.zeros((1, ntimesteps+1))
iekf_outs = np.zeros((1, ntimesteps+1))
smoother_ukf = UnscentedKalmanSmoother(transition_model)

bias = np.linalg.cholesky(P_0).T @ np.random.normal(size=target_state_0.shape)  # GroundTruth+chol(P0)’*randn
bias = np.array([-122068.01433784,     69.37315652,    507.76211348,    -86.74038986, -58321.63970861,     89.04789997])
prior = GaussianStatePrediction(target_state_0 + bias, P_0, timestamp=timesteps[0])

for all_measurements in measurements_sets:
    track_iekf = doPDA(all_measurements, transition_model, prior, IteratedKalmanUpdater(max_iterations=5))
    track_iplf = doPDA(all_measurements, transition_model, prior, IPLFKalmanUpdater())
    # track_iekf = doIEKF(all_measurements, transition_model, prior)
    # # track_iekf = doIEKF(all_measurements, transition_model, prior)
    # track_iplf = doIPLF(all_measurements, transition_model, prior)
    # # track_iplf = doIPLF(all_measurements, transition_model, prior)
    # # track_ipls = smoother_kf.smooth(track_iplf)
    track_ipls = doIPLS(track_iplf, transition_model, measurement_model, 1)
    euclidean_iekf = [Euclidean([0, 2, 4])(*pair)**2 for pair in zip(track_iekf, truth)]
    euclidean_iplf = [Euclidean([0, 2, 4])(*pair)**2 for pair in zip(track_iplf, truth)]
    euclidean_ipls = [Euclidean([0, 2, 4])(*pair)**2 for pair in zip(track_ipls, truth)]

    iekf_outs += np.array(euclidean_iekf)/n_mc_runs
    iplf_outs += np.array(euclidean_iplf)/n_mc_runs
    ipls_outs += np.array(euclidean_ipls)/n_mc_runs


    # out_iekf = TimeRangeMetric(
    #     title='Euclidean distances IEKF',
    #     value=euclidean_iekf,
    #     time_range=TimeRange(min(truth[0].timestamp), max(truth[-1].timestamp)))
    #
    # iekf_outs.append(euclidean_iekf)
    # iplf_outs.append(euclidean_iplf)

fig, axes = plt.subplots(figsize=(10, 5))
plt.plot(np.sqrt(euclidean_ipls), label='IPLS', color='blue')
plt.plot(np.sqrt(euclidean_iplf), label='IPLF', color='green')
plt.plot(np.sqrt(euclidean_iekf), label='IEKF', color='red')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.grid(visible=None, which='major', axis='both')
plt.xlabel('Time step')
plt.ylabel('RMS position error')
plt.legend()
# plt.show()


plotter = Plotterly()
plotter.plot_tracks(Track(prior), [0, 2], uncertainty=True, track_label='Prior', marker=dict(symbol='star', color='grey'), line=dict(color='white'))
plotter.plot_tracks(truth, [0, 2], track_label='Ground truth', line=dict(dash="dash", color='black'))
plotter.plot_tracks(track_ipls, [0, 2], uncertainty=True, track_label='IPLS', line=dict(color='blue'))
plotter.plot_tracks(track_iplf, [0, 2], uncertainty=True, track_label='IPLF', line=dict(color='green'))
plotter.plot_tracks(track_iekf, [0, 2], uncertainty=True, track_label='IEKF', line=dict(color='red'))
plotter.plot_measurements(all_measurements, [0, 2])


# plotter.plot_tracks(track, [0, 2], uncertainty=True, track_label='IPLS(1)-0, i.e. UKF')
# plotter.plot_tracks(track_smoothed_ukf, [0, 2], uncertainty=True, track_label='IPLS(1)-1, i.e. U-RTS smoother')
# # plotter.plot_tracks(track_smoothed_ipls1_5, [0, 2], uncertainty=True, track_label='IPLS(1)-5')
# # plotter.plot_tracks(track_smoothed_ipls1_10, [0, 2], uncertainty=True, track_label='IPLS(1)-10')
# plotter.fig
# plotter.fig.show()
plotter.fig.show()

print()