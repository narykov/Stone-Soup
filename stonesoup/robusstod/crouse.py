import numpy as np
from ..functions import sphere2cart, cart2sphere
from ..types.array import StateVector, StateVectors, CovarianceMatrix
from ..types.state import State
from . import ecef2enu, enu2ecef
from ..types.angle import Bearing, Elevation


def h_wrap(x, a=-np.pi, b=np.pi):
    """ Eq. (18) from https://ieeexplore.ieee.org/document/7266741 """
    return (x - a) % (b - a) + a


def h_wrap_sphere_single(zeta):
    out = StateVector([
        Elevation(zeta[0]),
        Bearing(h_wrap(zeta[1])),
        zeta[2],
        zeta[3]
    ])
    return out

def h_wrap_sphere_single_no_class(zeta):
    out = StateVector([
        zeta[0],
        h_wrap(zeta[1]),
        zeta[2],
        zeta[3]
    ])
    return out


def h_wrap_sphere_no_class(zetas):
    out = []
    for zeta in zetas:
        out.append([
            zeta[0],
            h_wrap(zeta[1]),
            zeta[2],
            zeta[3]
        ])
        # print(f'Before {zeta[1]} and after {h_wrap(zeta[1])}.')
    return StateVectors(out).T

def h_wrap_sphere(zetas):
    out = []
    for zeta in zetas:
        out.append([
            Elevation(zeta[0]),
            Bearing(h_wrap(zeta[1])),
            zeta[2],
            zeta[3]
        ])
        # print(f'Before {zeta[1]} and after {h_wrap(zeta[1])}.')
    return StateVectors(out).T

def meas_diff_no_class(measurement, measurement_prediction):
    # convert angles into sensor-focused x-y corrdinates
    # subtract vectors and produce the result
    # convert vector into the measurement domain again
    # do wrapping if necessary
    station_ecef = measurement.measurement_model.translation_offset
    measurement_model = measurement.measurement_model
    mapping = measurement.measurement_model.mapping

    measurements = [measurement, measurement_prediction]
    # svs = [measurement.state_vector, measurement_prediction.state_vector]
    svs = []
    for meas in measurements:
        target_ecef = measurement_model.inverse_function(meas)[mapping, :]
        target_enu = ecef2enu(target_ecef, station_ecef)
        svs.append(target_enu)

    svs_diff = svs[1] - svs[0]
    vector_diff_enu = 6 * [0]
    vector_diff_enu[::2] = svs_diff
    vector_diff_ecef = enu2ecef(svs_diff, station_ecef)
    vector_diff_ecef_6d = 6 * [0]
    vector_diff_ecef_6d[::2] = vector_diff_ecef
    meas_diff = measurement_model.function(State(state_vector=vector_diff_ecef_6d))
    r_diff = meas_diff[2]

    e_diff = meas_diff[0]
    b_diff = meas_diff[1]
    r_diff = measurement.state_vector[2] - measurement_prediction.state_vector[2]
    rr_diff = measurement.state_vector[3] - measurement_prediction.state_vector[3]

    meas_diff = StateVector([
        [e_diff],
        [b_diff],
        [r_diff],
        [rr_diff]
    ])

    return meas_diff

def sigma2gauss(sigma_points, mean_weights, covar_weights, fun, covar_noise=None):
    # sigma points (cubature points) are in the object state space

    mapping = (0, 2, 4)

    # COMPUTING THE MEAN

    # 1) Compute the transformed cubature points:
    rs = []
    rrs = []
    u_vectors = []
    for sigma_point in sigma_points:
        #TODO: introduce a check that it is the ElAzRR model
        e, b, r, rr = fun(State(state_vector=sigma_point))
        station_ecef = fun.__self__.translation_offset
        target_ecef = sigma_point[mapping, :]
        new_point = ecef2enu(target_ecef, station_ecef)
        r = np.linalg.norm(new_point)
        u = [element/r for element in new_point]
        e_new = np.arcsin(u[2] / np.linalg.norm(u))  #<- these values should be close to the original e and b
        b_new = np.arctan2(u[1], u[0])

        rs.append(r)
        rrs.append(rr)
        u_vectors.append(u)

    u_vectors = StateVectors(u_vectors).T

    # 2) The mean of the range components is found the standard way as a linear sum,
    r_mean = np.average(rs, weights=mean_weights)
    rr_mean = np.average(rrs, weights=mean_weights)

    # 3) Average 3D unit vectors for all the cubature points:
    u_mean = np.average(u_vectors, axis=1, weights=mean_weights)
    # u_mean = np.average(u_vectors, axis=1)

    e_mean = np.arcsin(u_mean[2] / np.linalg.norm(u_mean))
    b_mean = np.arctan2(u_mean[1], u_mean[0])

    # mean = StateVector([
    #     Elevation(e_mean),
    #     Bearing(b_mean),
    #     r_mean,
    #     rr_mean
    # ])
    mean = StateVector([
        [e_mean],
        [b_mean],
        [r_mean],
        [rr_mean]
    ])

    # COMPUTING THE COVARIANCE
    sigma_points_t = StateVectors([fun(State(state_vector=sigma_point)) for sigma_point in sigma_points])

    points_diff = h_wrap_sphere_no_class(sigma_points_t.astype(float)-mean)
    covar = (
        points_diff @ (np.diag(covar_weights)) @ points_diff.T
    )
    var_pred = np.zeros((4, 4))
    var_xz = np.zeros((6, 4))
    for m, w, sigma_point_t, sigma_point in zip(mean_weights, covar_weights, sigma_points_t.astype(float), sigma_points):
        sub_z_j = h_wrap_sphere_single_no_class(sigma_point_t - mean)
        var_pred = var_pred + w * (sub_z_j*sub_z_j.T)
        var_xz = var_xz + m * (sigma_point - sigma_points[:, 0:1]) * sub_z_j.T

    # covar = var_pred

    if covar_noise is not None:
        covar = covar + covar_noise
    cross_covar = (
        (sigma_points-sigma_points[:, 0:1]) @ np.diag(mean_weights) @ points_diff.T
    )
    # cross_covar = var_xz

    return mean.view(StateVector), covar.view(CovarianceMatrix), cross_covar.view(CovarianceMatrix), sigma_points_t

