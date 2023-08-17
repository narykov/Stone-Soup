import numpy as np
import torch
import godot
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


def diff_equation(state, **kwargs):

    if 'timestamp' in kwargs:
        timeiso = kwargs['timestamp'].isoformat(timespec='microseconds')
        timescale = 'TDB'
        t = ' '.join([timeiso, timescale])
        epoch = tempo.XEpoch(t)
    else:
        epoch = tempo.XEpoch()

    (x, x_dot, y, y_dot, z, z_dot) = state
    uni = cosmos.Universe(cosmos.util.load_yaml("universe.yml"))
    tensor1 = torch.tensor([x, y, z, x_dot, y_dot, z_dot], requires_grad=True)
    pos = ad.Vector(tensor1.detach().numpy(), 'x0')  # x, y, z, xdot, ydot, zdot

    tscale = tempo.TimeScale.TDB

    # epoch = tempo.XEpoch()
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


def rng(state, station, uni, **kwargs):
    # check if more than a single state_vector and be prepared to return more; 'statevectors' carries the relevant flag
    if kwargs['statevectors']:
        vectors = [state_vector for state_vector in state.state_vector]
    else:
        vectors = [state.state_vector]

    # obtain epoch
    timeiso = state.timestamp.isoformat(timespec='microseconds')
    timescale = 'TDB'
    t = ' '.join([timeiso, timescale])
    epoch = godot.core.tempo.XEpoch(t)

    # other ids
    icrf = uni.frames.axesId('ICRF')
    satellite = uni.frames.pointId("Satellite")

    rhos = []
    for state_vector in vectors:
        # obtain station and then range
        (x, x_dot, y, y_dot, z, z_dot) = np.array(state_vector.T)[0]
        pos = godot.core.autodif.Vector([x, y, z, x_dot, y_dot, z_dot], 'x0')
        translation_model = godot.model.common.ConstantVectorTimeEvaluable(pos)
        uni.frames.addTimeEvaluableTranslation(satellite, station, icrf, translation_model)
        rho = uni.frames.distance(station, satellite, epoch)
        rhos.append(rho.value())

    return rhos
