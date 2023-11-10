import numpy as np
from ..types.array import StateVector
from ..functions import cart2sphere, sphere2cart


def enu2ecef(target_enu, station_ecef):

    # using https://sciencing.com/convert-xy-coordinates-longitude-latitude-8449009.html
    x, y, z = station_ecef
    r = np.linalg.norm(station_ecef)
    s_lon = np.arcsin(z / r)  # also lambda
    s_lat = np.arctan2(y, x)  # also phi

    # The ENU to ECEF conversion can be easily calculated like this (https://gis.stackexchange.com/a/308452):
    # https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates
    R = np.array([[-np.sin(s_lat), -np.sin(s_lon) * np.cos(s_lat), np.cos(s_lon) * np.cos(s_lat)],
                  [np.cos(s_lat), -np.sin(s_lon) * np.sin(s_lat), np.cos(s_lon) * np.sin(s_lat)],
                  [0, np.cos(s_lon), np.sin(s_lon)]], dtype='float64')

    return R @ target_enu + station_ecef


def ecef2enu(target_ecef, station_ecef):

    # using https://sciencing.com/convert-xy-coordinates-longitude-latitude-8449009.html
    x, y, z = station_ecef
    r = np.linalg.norm(station_ecef)
    s_lon = np.arcsin(z / r)  # also lambda
    s_lat = np.arctan2(y, x)  # also phi

    # using https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates
    R = np.array([[-np.sin(s_lat), np.cos(s_lat), 0],
                  [-np.cos(s_lat)*np.sin(s_lon), -np.sin(s_lat)*np.sin(s_lon), np.cos(s_lon)],
                  [np.cos(s_lat)*np.cos(s_lon), np.sin(s_lat)*np.cos(s_lon), np.sin(s_lon)]], dtype='float64')

    enu = R @ (target_ecef - station_ecef)  # xEast, yNorth, zUp

    return StateVector(enu)


def ecef2aer(target_ecef, station_ecef):
    target_enu = ecef2enu(target_ecef, station_ecef)
    return cart2sphere(*target_enu)  # range, azimuth, elevation


def aer2ecef(target_aer, station_ecef):
    # range, azimuth, elevation
    target_rae = StateVector([target_aer[2], target_aer[0], target_aer[1]])
    target_enu = StateVector(sphere2cart(*target_rae))
    target_ecef = enu2ecef(target_enu, station_ecef)
    return target_ecef
