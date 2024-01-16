#!/usr/bin/env python

"""
Study of the filtering update based on the hypothesis in the pickled file.
The update is not working well as the posterior does not appear to be adequate.
The first figure plots the correlations between the predictive prior and the measurement values.
The second figure is the 3D plot of the predicted and updated 3-sigma ellipses.
"""
import pickle
import pandas as pd
import seaborn as sns

import sys
import matplotlib.pyplot as plt
import numpy as np
from stonesoup.types.state import State

# ROBUSSTOD MODULES
from stonesoup.robusstod.stonesoup.updater_direction_measurements import IPLFKalmanUpdater
from stonesoup.robusstod.stonesoup.updater_direction_measurements import UnscentedKalmanUpdater

# debugging
from stonesoup.robusstod.utils import true_state_metadata_extraction
from stonesoup.robusstod.utils import station_metadata_extraction
from stonesoup.robusstod.utils import get_cov_ellipsoid
from stonesoup.robusstod.utils import plot_axes

def produce_pairplot(hypothesis, num_samples=None):
    if num_samples is None:
        # Set the number of samples
        num_samples = 100

    def measurement_function(state_vectors, measurement):
        measurement_model = measurement.measurement_model

        out = None
        for state_vector in state_vectors:
            measurement = np.array(measurement_model.function(State(state_vector=state_vector))).T
            out = np.vstack([out, measurement]) if out is not None else measurement

        return [column for column in out.astype(float).T]

    measurement = hypothesis.measurement
    prediction = hypothesis.prediction

    # Set the mean and covariance matrix for the multivariate normal distribution
    mean = prediction.state_vector.ravel()
    covariance_matrix = prediction.covar

    # Generate random samples from the multivariate normal distribution
    samples = np.random.multivariate_normal(mean, covariance_matrix, num_samples)


    # Create a pandas DataFrame from the samples
    columns = ['X', 'VX', 'Y', 'VY', 'Z', 'VZ']
    df = pd.DataFrame(samples, columns=columns)
    df['El'], df['Az'], df['R'], df['RR'] = measurement_function(df[columns].values, measurement)

    for element in ['El', 'Az']:
        df[element] = df[element].apply(lambda x: np.rad2deg(x))

    sns_plot = sns.pairplot(df)
    sns_plot.figure.savefig("output.png")
    plt.show()


def produce_3dplot(post_state, hypothesis):
    # plotting
    mapping = hypothesis.measurement.measurement_model.mapping
    predicted_state = hypothesis.prediction
    true_state = true_state_metadata_extraction(hypothesis.measurement)  # state vector and time stamp
    station_state = station_metadata_extraction(hypothesis.measurement)

    fig = plt.figure(figsize=plt.figaspect(1))
    ax = fig.add_subplot(projection='3d')
    plt.plot(0, 0, marker='*', color='b', label='centre of the Earth')
    # ax.set_aspect('equal', 'box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.plot(*true_state.state_vector[mapping, :], 'k+', label='ground truth')
    ax.plot(*predicted_state.state_vector[mapping, :], 'b+', label='prior mean')
    ax.plot(*post_state.state_vector[mapping, :], 'b+', label='posterior mean')


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
    plt.legend()
    plt.show()

def main():

    with open('hypothesis.pickle', 'rb') as f:
        hypothesis = pickle.load(f)


    # updater = UnscentedKalmanUpdater(beta=2, kappa=30)
    updater = IPLFKalmanUpdater(tolerance=1e-1, max_iterations=5, beta=2, kappa=30)
    post_state = updater.update(hypothesis)

    produce_pairplot(hypothesis, num_samples=1000)
    produce_3dplot(post_state, hypothesis)



if __name__ == "__main__":
    main()
