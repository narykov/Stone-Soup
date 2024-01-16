"""Mathematical functions used within robusstod Stone Soup"""

import numpy as np
import copy

from ...types.array import StateVector, StateVectors, CovarianceMatrix
from ...functions import sigma2gauss

def slr_definition(state_pdf, prediction):
    """ Statistical linear regression (SLR), implements the definition (9)-(11) as found in
    Á. F. García-Fernández, L. Svensson and S. Särkkä, "Iterated Posterior Linearization Smoother,"
    in IEEE Transactions on Automatic Control, vol. 62, no. 4, pp. 2056-2063, April 2017, doi: 10.1109/TAC.2016.2592681.
    """

    # First two moments of the state pdf
    x_bar = state_pdf.state_vector
    p_matrix = state_pdf.covar

    # The predicted quantities wrt the state pdf (e.g. using sigma points)
    z_bar = prediction.state_vector.astype(float)
    # next quantity (psi) is naturally available in GaussianMeasurementPrediction for predicted measurements,
    # but has been introduced to AugmentedGaussianStatePrediction for state predictions
    psi = prediction.cross_covar
    phi = prediction.covar

    # Statistical linear regression parameters of a function predicted with the quantities above
    H_plus = psi.T @ np.linalg.inv(p_matrix)
    b_plus = z_bar - H_plus @ x_bar
    Omega_plus = phi - H_plus @ p_matrix @ H_plus.T

    # sanity check
    if not any(np.isclose(z_bar.astype(float), b_plus + H_plus @ x_bar, rtol=1e-05, atol=1e-08)):
        print('Issues in the SLR calculations')

    # The output is the function's SLR with respect to the state_pdf
    return H_plus, b_plus, Omega_plus


def unscented_transform(sigma_points_states, mean_weights, covar_weights,
                        fun, points_noise=None, covar_noise=None):
    """
    Apply the Unscented Transform to a set of sigma points

    Apply f to points (with secondary argument points_noise, if available),
    then approximate the resulting mean and covariance. If sigma_noise is
    available, treat it as additional variance due to additive noise.

    Parameters
    ----------
    sigma_points : :class:`~.StateVectors` of shape `(Ns, 2*Ns+1)`
        An array containing the locations of the sigma points
    mean_weights : :class:`numpy.ndarray` of shape `(2*Ns+1,)`
        An array containing the sigma point mean weights
    covar_weights : :class:`numpy.ndarray` of shape `(2*Ns+1,)`
        An array containing the sigma point covariance weights
    fun : function handle
        A (non-linear) transition function
        Must be of the form "y = fun(x,w)", where y can be a scalar or \
        :class:`numpy.ndarray` of shape `(Ns, 1)` or `(Ns,)`
    covar_noise : :class:`~.CovarianceMatrix` of shape `(Ns, Ns)`, optional
        Additive noise covariance matrix
        (default is `None`)
    points_noise : :class:`numpy.ndarray` of shape `(Ns, 2*Ns+1,)`, optional
        points to pass into f's second argument
        (default is `None`)

    Returns
    -------
    : :class:`~.StateVector` of shape `(Ns, 1)`
        Transformed mean
    : :class:`~.CovarianceMatrix` of shape `(Ns, Ns)`
        Transformed covariance
    : :class:`~.CovarianceMatrix` of shape `(Ns,Nm)`
        Calculated cross-covariance matrix
    : :class:`~.StateVectors` of shape `(Ns, 2*Ns+1)`
        An array containing the locations of the transformed sigma points
    : :class:`numpy.ndarray` of shape `(2*Ns+1,)`
        An array containing the transformed sigma point mean weights
    : :class:`numpy.ndarray` of shape `(2*Ns+1,)`
        An array containing the transformed sigma point covariance weights
    """
    # Reconstruct the sigma_points matrix
    sigma_points = StateVectors([
        sigma_points_state.state_vector for sigma_points_state in sigma_points_states])

    # Transform points through f
    if points_noise is None:
        sigma_points_t = StateVectors([
            fun(sigma_points_state) for sigma_points_state in sigma_points_states])
    else:
        sigma_points_t = StateVectors([
            fun(sigma_points_state, points_noise)
            for sigma_points_state, point_noise in zip(sigma_points_states, points_noise.T)])

    # Calculate mean and covariance approximation
    mean, covar = sigma2gauss(sigma_points_t, mean_weights, covar_weights, covar_noise)


    # Calculate cross-covariance
    cross_covar = (
        (sigma_points-sigma_points[:, 0:1]) @ np.diag(mean_weights) @ (sigma_points_t-mean).T
    ).view(CovarianceMatrix)

    measurement_model_processing = True
    try:
        fun_type = fun.__self__
        print("Crouse's measurmement prediction activated.")
        from ...robusstod.crouse import sigma2gauss as sigma2gauss_crouse
        mean_crouse, covar_crouse, cross_covar_crouse, sigma_points_t_crouse = (
            sigma2gauss_crouse(sigma_points, mean_weights, covar_weights, fun, covar_noise=covar_noise)
        )
        # print(f'Before Crouse: {mean}.')
        mean = mean_crouse
        covar = covar_crouse
        cross_covar = cross_covar_crouse
        sigma_points_t = sigma_points_t_crouse

    except AttributeError:
        measurement_model_processing = False
        print('This was the transition model.')

    if measurement_model_processing:
        import matplotlib.pyplot as plt
        # print()
        # fig, ax = plt.subplots()
        # for sigma_point in sigma_points_t:
        #     ax.plot(sigma_point[0], sigma_point[1], 'xr')
        # ax.plot(mean[0], mean[1], 'or')
        #
        # for sigma_point in sigma_points_t_crouse:
        #     ax.plot(sigma_point[0], sigma_point[1], '.b')
        # ax.plot(mean_crouse[0], mean_crouse[1], 'xb')

        # print()

    return mean, covar, cross_covar, sigma_points_t, mean_weights, covar_weights

