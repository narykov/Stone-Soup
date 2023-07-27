import numpy as np
import torch
from godot.core import astro
from godot.core import tempo, util, autodif as ad
from godot.model import common
from godot import cosmos


def KeplerianToCartesian(K, mu_earth, ndim, mapping_position, mapping_velocity):
    """ The function is based on https://godot.io.esa.int/tutorials/T04_Astro/T04scv/ """

    xyz = astro.cartFromKep(K, mu_earth)

    state_vector = np.zeros(ndim)
    state_vector[[mapping_position]] = xyz[:3]
    state_vector[[mapping_velocity]] = xyz[3:]

    return state_vector


# def twoBody3d_da(state, GM=None):
#     (x, x_dot, y, y_dot, z, z_dot) = state
#     uni = cosmos.Universe(cosmos.util.load_yaml("universe.yml"))
#     # uni = cosmos.Universe(cosmos.util.load_yaml("universe_t30.yml"))
#     tensor1 = torch.tensor([x, y, z, x_dot, y_dot, z_dot], requires_grad=True)
#     pos = ad.Vector(tensor1.detach().numpy(), 'x0')  # x, y, z, xdot, ydot, zdot
#
#     tscale = tempo.TimeScale.TDB
#
#     epoch = tempo.XEpoch()
#     satellite = uni.frames.addPoint("Satellite", tscale)
#     icrf = uni.frames.axesId('ICRF')
#     earth = uni.frames.pointId('Earth')
#
#     translation_model = common.ConstantVectorTimeEvaluable(pos)
#     uni.frames.addTimeEvaluableTranslation(satellite, earth, icrf, translation_model)
#
#     # dyn = uni.dynamics.get("EarthPointMass")
#     dyn = uni.dynamics.get("Earth")
#     # dyn = uni.dynamics.get("EarthJ2")
#
#     uni.frames.setAlias(dyn.point, satellite)
#     uni.frames.setAlias(dyn.coi, earth)
#     uni.frames.setAlias(dyn.axes, icrf)
#
#     try:
#         accel_godot = dyn.acc.eval(epoch)  # -5.04456984e-09
#     except RuntimeError as e:
#         print(e)
#     accel = accel_godot.value() * ((1e3) ** 3)


def twoBody3d_da(state, GM=None):
    if GM is None:
        GM = 398600400000000.0

    (x, x_dot, y, y_dot, z, z_dot) = state
    r_pow_3 = torch.float_power(torch.linalg.vector_norm(torch.tensor((x, y, z))), 3)
    return (x_dot, -GM * x / r_pow_3, y_dot, -GM * y / r_pow_3, z_dot, -GM * z / r_pow_3)