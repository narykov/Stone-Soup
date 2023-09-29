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
    z_bar = prediction.state_vector
    # next quantity (psi) is naturally available in GaussianMeasurementPrediction for predicted measurements,
    # but has been introduced to AugmentedGaussianStatePrediction for state predictions
    psi = prediction.cross_covar
    phi = prediction.covar

    # Statistical linear regression parameters of a function predicted with the quantities above
    H_plus = psi.T @ np.linalg.inv(p_matrix)
    b_plus = z_bar - H_plus @ x_bar
    Omega_plus = phi - H_plus @ p_matrix @ H_plus.T

    # The output is the function's SLR with respect to the state_pdf
    return {'matrix': H_plus, 'vector': b_plus, 'cov_matrix': Omega_plus}


# def slr(state, predictor, model, model_type, **kwargs):
#     """
#     Statistical linear regression using sigma points.
#     Notation follows https://github.com/Agarciafernandez/IPLF/blob/main/IPLF_maneuvering.m
#     """
#
#         from .updater import UnscentedKalmanUpdater
#         from .predictor import AugmentedUnscentedKalmanPredictor
#         def _slr(state_pdf, prediction):
#         """ Statistical linear regression (SLR) """
#
#         # First two moments of the state pdf
#         x_bar = state_pdf.state_vector
#         p_capital = state_pdf.covar
#
#         # The predicted quantities wrt the state pdf (e.g. using sigma points)
#         z_bar = prediction.state_vector
#         psi = prediction.cross_covar
#         phi = prediction.covar
#
#         # Statistical linear regression parameters of a function predicted with the quantities above
#         H_plus = psi.T @ np.linalg.inv(p_capital)
#         b_plus = z_bar - H_plus @ x_bar
#         Omega_plus = phi - H_plus @ p_capital @ H_plus.T
#
#         # the output is a linear approximation
#         return {'matrix': H_plus, 'vector': b_plus, 'covariance': Omega_plus}
#
#     if 'timestamp' in kwargs:
#         timestamp = kwargs['timestamp']
#         prediction = predictor.predict(state, timestamp=timestamp)
#     elif 'measurement_model' in kwargs:
#         measurement_model = kwargs['measurement_model']
#         prediction = predictor.predict_measurement(state, measurement_model=measurement_model)
#     else:
#         print('The model in the SLR has not been set up correctly.')
#
#     return _slr(state, prediction)
