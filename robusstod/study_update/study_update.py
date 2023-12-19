#!/usr/bin/env python

"""
Study of the IPLF update.
"""
import pickle

import sys
import matplotlib.pyplot as plt
import numpy as np
from stonesoup.robusstod.stonesoup.updater import UnscentedKalmanUpdater

# ROBUSSTOD MODULES
from stonesoup.robusstod.stonesoup.updater import IPLFKalmanUpdater

# debugging
from stonesoup.robusstod.utils import true_state_metadata_extraction
from stonesoup.robusstod.utils import station_metadata_extraction
from stonesoup.robusstod.utils import get_cov_ellipsoid
from stonesoup.robusstod.utils import plot_axes

def main():

    with open('hypothesis.pickle', 'rb') as f:
        hypothesis = pickle.load(f)
    measurement = hypothesis.measurement
    mapping = measurement.measurement_model.mapping
    true_state = true_state_metadata_extraction(measurement)  # state vector and time stamp
    predicted_state = hypothesis.prediction
    station_state = station_metadata_extraction(measurement)

    updater = UnscentedKalmanUpdater(beta=2, kappa=30)
    updater = IPLFKalmanUpdater(tolerance=1e-1, max_iterations=10, beta=2, kappa=30)
    post_state = updater.update(hypothesis)

    # plotting
    fig = plt.figure(figsize=plt.figaspect(1))
    ax = fig.add_subplot(projection='3d')
    plt.plot(0, 0, marker='*', color='b')
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.plot(*true_state.state_vector[mapping, :], 'k+')
    ax.plot(*predicted_state.state_vector[mapping, :], 'b+')

    ax.plot(*post_state.state_vector[mapping, :], 'r+')
    ax.plot(*station_state.state_vector[mapping, :], 'y.')
    plot_axes(hypothesis, axis_length=1000000)

    nstd = 3
    prior_inputs = {
        'mu': predicted_state.state_vector[mapping, :],
        'cov': predicted_state.covar[mapping, :][:, mapping],
        'nstd': nstd
    }
    posterior_inputs = {
        'mu': post_state.state_vector[mapping, :],
        'cov': post_state.covar[mapping, :][:, mapping],
        'nstd': nstd
    }
    ax.plot_wireframe(*(get_cov_ellipsoid(**prior_inputs)), color='b', alpha=0.05)
    ax.plot_wireframe(*(get_cov_ellipsoid(**posterior_inputs)), color='r', alpha=0.05)
    ax.set_aspect('equal', adjustable='box')


if __name__ == "__main__":
    main()
