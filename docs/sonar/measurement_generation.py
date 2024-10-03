#!/usr/bin/env python
# coding: utf-8

""" Measurement generation in 3D space with conical angles"""
import numpy as np
from datetime import datetime, timedelta

# StoneSoup imports
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity
from stonesoup.types.detection import Detection
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.state import StateVector

# Custom imports
from stonesoup.sonar.measurement import BistaticConicalAnglesDelayAzimuth

def main():
    start_time = datetime(2000, 1, 1, 0, 0, 0)
    np.random.seed(1991)

    q_x = q_y = q_z = 0.05
    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(q_x),
                                                              ConstantVelocity(q_y),
                                                              ConstantVelocity(q_z)])

    timesteps = [start_time]
    truth = GroundTruthPath([GroundTruthState([-100, 0, 300, 0, -100, 0], timestamp=timesteps[0])])

    num_steps = 20
    for k in range(1, num_steps + 1):
        timesteps.append(start_time + timedelta(seconds=k))  # add next timestep to list of timesteps
        truth.append(GroundTruthState(
            transition_model.function(truth[k - 1], noise=False, time_interval=timedelta(seconds=1)),
            timestamp=timesteps[k]))


    measurement_model = BistaticConicalAnglesDelayAzimuth(
        translation_offset=StateVector([[1000], [0], [0]]),
        translation_offset_origin=StateVector([[0], [0], [0]]),
        ndim_state=6,
        mapping=(0, 2, 4),
        noise_covar=np.diag([1e-1, 1e-1, 1, 1e-1]),
        wave_propagation_speed=1500)

    measurements = []
    for state in truth:
        measurement = measurement_model.function(state, noise=False)
        measurements.append(Detection(state_vector=measurement,
                                      measurement_model=measurement_model,
                                      timestamp=state.timestamp))

    truth_recovered = GroundTruthPath()
    for detection in measurements:
        state_inv = measurement_model.inverse_function(detection)
        truth_recovered.append(GroundTruthState(state_inv, timestamp=detection.timestamp))

    # Import plotter

    errors_x = []
    errors_y = []
    errors_z = []
    for actual, estimate in zip(truth, truth_recovered):
        error_x = abs(actual.state_vector[0] - estimate.state_vector[0])
        error_y = abs(actual.state_vector[2] - estimate.state_vector[2])
        error_z = abs(actual.state_vector[4] - estimate.state_vector[4])
        errors_x.append(error_x)
        errors_y.append(error_y)
        errors_z.append(error_z)

    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(timesteps, errors_x, label='X')
    ax.plot(timesteps, errors_y, label='Y')
    ax.plot(timesteps, errors_z, label='Z')
    ax.set_ylabel('Error')
    ax.set_xlabel('Time')
    ax.legend()
    plt.show()


    print()


if __name__ == "__main__":
    main()
