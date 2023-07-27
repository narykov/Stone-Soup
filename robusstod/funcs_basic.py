import numpy as np
import torch
from astropy.constants import G, M_earth, R_earth


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