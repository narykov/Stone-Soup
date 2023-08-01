import numpy as np
import torch


def get_noise_coefficients(GM):
    """ A function that returns adequate noise coefficients for Van Loan's method, ideally from physical considerations.
    We need these values that drive the process noise in the prediction step.
    """
    # ratio = 1e-2  # 2 orders are suggested by Simon following a Lee's manuscript
    ratio = 1e-14  # otherwise it `lands in Bolivia'
    st_dev = ratio * GM
    q = st_dev ** 2
    q_xdot, q_ydot, q_zdot = q, q, q
    return np.array([q_xdot, q_ydot, q_zdot])


def KeplerianToCartesian(K, GM, ndim, mapping_position, mapping_velocity):
    """ Based on KeplerianToCartesian from https://github.com/alecksphillips/SatelliteModel/blob/main/transforms.py """

    a, e, i, w, omega, nu = K[:]  # elements i, w, omega, nu need to be in radians, a in metres

    E = np.arctan2(np.sqrt(1 - e ** 2) * np.sin(nu), e + np.cos(nu))

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

    state_vector = np.zeros(ndim)
    state_vector[[mapping_position]] = [X, Y, Z]
    state_vector[[mapping_velocity]] = [U, V, W]

    return state_vector


def twoBody3d_da(state, GM=None):
    if GM is None:
        GM = 398600400000000.0

    (x, x_dot, y, y_dot, z, z_dot) = state
    r_pow_3 = torch.float_power(torch.linalg.vector_norm(torch.tensor((x, y, z))), 3)
    return (x_dot, -GM * x / r_pow_3,
            y_dot, -GM * y / r_pow_3,
            z_dot, -GM * z / r_pow_3)