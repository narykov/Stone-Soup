import numpy as np
import torch
from godot.core import astro
from godot.core import tempo, util, autodif as ad
from godot.model import common
from godot import cosmos


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


def KeplerianToCartesian(K, mu_earth, ndim, mapping_position, mapping_velocity):
    """ The function is based on https://godot.io.esa.int/tutorials/T04_Astro/T04scv/ """

    xyz = astro.cartFromKep(K, mu_earth)

    state_vector = np.zeros(ndim)
    state_vector[[mapping_position]] = xyz[:3]
    state_vector[[mapping_velocity]] = xyz[3:]

    return state_vector


def twoBody3d_da(state, **kwargs):
    (x, x_dot, y, y_dot, z, z_dot) = state
    uni = cosmos.Universe(cosmos.util.load_yaml("universe.yml"))
    tensor1 = torch.tensor([x, y, z, x_dot, y_dot, z_dot], requires_grad=True)
    pos = ad.Vector(tensor1.detach().numpy(), 'x0')  # x, y, z, xdot, ydot, zdot

    tscale = tempo.TimeScale.TDB

    epoch = tempo.XEpoch()
    satellite = uni.frames.addPoint("Satellite", tscale)
    icrf = uni.frames.axesId('ICRF')
    earth = uni.frames.pointId('Earth')

    translation_model = common.ConstantVectorTimeEvaluable(pos)
    uni.frames.addTimeEvaluableTranslation(satellite, earth, icrf, translation_model)

    dyn = uni.dynamics.get("Earth")

    uni.frames.setAlias(dyn.point, satellite)
    uni.frames.setAlias(dyn.coi, earth)
    uni.frames.setAlias(dyn.axes, icrf)

    try:
        accel_godot = dyn.acc.eval(epoch)
    except RuntimeError as e:
        print(e)
    accel = accel_godot.value() * ((1e3) ** 3)

    return (x_dot, torch.tensor(accel[0], requires_grad=False),
            y_dot, torch.tensor(accel[1], requires_grad=False),
            z_dot, torch.tensor(accel[2], requires_grad=False))