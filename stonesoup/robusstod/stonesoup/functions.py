"""Mathematical functions used within robusstod Stone Soup"""

import numpy as np


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
