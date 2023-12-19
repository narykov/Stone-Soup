import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from ..types.state import State
from ..types.array import StateVector
import robusstod
import copy


def plot_axes(hypothesis, axis_length=1000000):
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

    def get_bearing(measurement, measurement_true, axis_length=None):
        measurement_bear = copy.deepcopy(measurement)
        if axis_length is None:
            measurement_bear.state_vector[2] = (
                    np.cos(measurement_bear.state_vector[0]) * measurement_bear.state_vector[2])
        else:
            # meaning, we want to plot where 0 bearing points
            measurement_bear.state_vector[2] = axis_length
            measurement_bear.state_vector[1] = 0
        measurement_bear.state_vector[0] = 0
        detection_pos = measurement_true.measurement_model.inverse_function(measurement_bear)[mapping, :]
        data = [[station_pos[i], detection_pos[i]] for i in range(3)]
        return data

    measurement = copy.deepcopy(hypothesis.measurement)
    mapping = hypothesis.measurement.measurement_model.mapping
    station_pos = station_metadata_extraction(measurement).state_vector[mapping, :]

    e_axis = axis_length * np.vstack([1, 0, 0, 0, 0, 0])
    n_axis = axis_length * np.vstack([0, 0, 1, 0, 0, 0])
    u_axis = axis_length * np.vstack([0, 0, 0, 0, 1, 0])

    point_e = enu2ecef(e_axis[mapping, :], station_pos)
    point_n = enu2ecef(n_axis[mapping, :], station_pos)
    point_u = enu2ecef(u_axis[mapping, :], station_pos)

    points = [point_e, point_n, point_u]

    for point in points:
        data = [[station_pos[i], point[i]] for i in range(3)]
        plt.gca().plot(*data, 'yo-', markersize=1)
        plt.gca().set_aspect('equal', 'box')

    # plt.gca().plot(*get_bearing(hypothesis.measurement,
    #                             hypothesis.measurement), 'r-')  # plot the bearing of original measurement
    # plt.gca().plot(*get_bearing(hypothesis.measurement_prediction,
    #                             hypothesis.measurement), 'g-')
    # plt.gca().plot(*get_bearing(hypothesis.measurement_prediction,
    #                             hypothesis.measurement, axis_length=axis_length), 'b--')
    # plot the bearing of predicted measurement
    print()

def plot_detections(detections, color=None):
    if color is None:
        color = 'r'
    xs = []
    ys = []
    for detection in detections:
        detect_state = detection.measurement_model.inverse_function(detection)
        xs.append(detect_state[0])
        ys.append(detect_state[2])
    # for detection in detections:
    #     detect_state = detection.state_vector
    #     xs.append(detect_state[0])
    #     ys.append(detect_state[1])
    plt.scatter(x=xs, y=ys, s=80, facecolors='none', edgecolors=color)

def plot_linear_detections(detections, color=None):
    if color is None:
        color = 'r'
    xs = []
    ys = []
    for detection in detections:
        detect_state = detection.measurement_model.inverse_function(detection)
        xs.append(detect_state[0])
        ys.append(detect_state[2])
    plt.scatter(x=xs, y=ys, s=80, facecolors='none', edgecolors=color)

def plot_tracks(tracks, ax=None, color=None, label=None):
    if ax is None:
        ax = plt.gca()
    if color is None:
        color = 'g'

    for track in tracks:
        x = [state.state_vector[0] for state in track.states]
        y = [state.state_vector[2] for state in track.states]
        plt.plot(x, y, linestyle='-', color=color, label=label)
        for state in track:
            plot_ellipse(state.mean[[0, 2], :], state.covar[[0, 2], :][:, [0, 2]], facecolor='none',
                         linestyle='-', edgecolor=color, ax=ax, linewidth=1)
        # plot_ellipse(track.mean[[0, 2], :], track.covar[[0, 2], :][:, [0, 2]], facecolor='none',
        # linestyle='-', edgecolor=color, ax=ax, linewidth=1)


def plot_ground_truth(tracks, ax=None, color=None, label=None):
    if ax is None:
        ax = plt.gca()
    if color is None:
        color = 'k'

    for track in tracks:
        x = [state.state_vector[0] for state in track]
        y = [state.state_vector[2] for state in track]
        plt.plot(x, y, linestyle='--', color=color, label=label)
        # for state in track:
        #     plot_ellipse(state.mean[[0, 2], :], state.covar[[0, 2], :][:, [0, 2]], facecolor='none',
        #                  linestyle='--', edgecolor=color, ax=ax, linewidth=1)
        # plot_ellipse(track.mean[[0, 2], :], track.covar[[0, 2], :][:, [0, 2]], facecolor='none',
        # linestyle='-', edgecolor=color, ax=ax, linewidth=1)


def plot_ellipse(mu, cov, nstd=3, ax=None, **kwargs):
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    width, height = 2*nstd*np.sqrt(vals)
    ellip = Ellipse(xy=mu, width=width, height=height, angle=theta,
                    alpha=0.4, **kwargs)
    ax.add_artist(ellip)
    return ellip


def get_cov_ellipsoid(cov, mu=np.zeros((3)), nstd=3):
    """
    Return the 3d points representing the covariance matrix
    cov centred at mu and scaled by the factor nstd.

    Plot on your favourite 3d axis.
    Example 1:  ax.plot_wireframe(X,Y,Z,alpha=0.1)
    Example 2:  ax.plot_surface(X,Y,Z,alpha=0.1)

    Source: https://github.com/CircusMonkey/covariance-ellipsoid/blob/master/ellipsoid.py
    """
    assert cov.shape == (3, 3)

    # Find and sort eigenvalues to correspond to the covariance matrix
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.sum(cov, axis=0).argsort()
    eigvals_temp = eigvals[idx]
    idx = eigvals_temp.argsort()
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Set of all spherical angles to draw our ellipsoid
    n_points = 100
    theta = np.linspace(0, 2 * np.pi, n_points)
    phi = np.linspace(0, np.pi, n_points)

    # Width, height and depth of ellipsoid
    rx, ry, rz = nstd * np.sqrt(eigvals)

    # Get the xyz points for plotting
    # Cartesian coordinates that correspond to the spherical angles:
    X = rx * np.outer(np.cos(theta), np.sin(phi))
    Y = ry * np.outer(np.sin(theta), np.sin(phi))
    Z = rz * np.outer(np.ones_like(theta), np.cos(phi))

    # Rotate ellipsoid for off axis alignment
    old_shape = X.shape
    # Flatten to vectorise rotation
    X, Y, Z = X.flatten(), Y.flatten(), Z.flatten()
    X, Y, Z = np.matmul(eigvecs, np.array([X, Y, Z]))
    X, Y, Z = X.reshape(old_shape), Y.reshape(old_shape), Z.reshape(old_shape)

    # Add in offsets for the mean
    X = X + mu[0]
    Y = Y + mu[1]
    Z = Z + mu[2]

    return X, Y, Z


def true_state_metadata_extraction(measurement):
    md = measurement.metadata
    true_state = State(
        state_vector=StateVector([float(md['TRUE_X']) * 1000, float(md['TRUE_VX']) * 1000,
                                  float(md['TRUE_Y']) * 1000, float(md['TRUE_VY']) * 1000,
                                  float(md['TRUE_Z']) * 1000, float(md['TRUE_VZ']) * 1000]),
        timestamp=measurement.timestamp
    )
    return true_state


def station_metadata_extraction(measurement):
    md = measurement.metadata
    true_state = State(
        state_vector=StateVector([float(md['XSTAT_X']) * 1000, float(md['XSTAT_VX']) * 1000,
                                  float(md['XSTAT_Y']) * 1000, float(md['XSTAT_VY']) * 1000,
                                  float(md['XSTAT_Z']) * 1000, float(md['XSTAT_VZ']) * 1000]),
        timestamp=measurement.timestamp
    )
    return true_state

# plot_tracks([track], color='r', label='original_track')
# plot_tracks([track_smoothed], color='g', label='UKF smoothing')
# plot_tracks([track_forward], color='m', label='forward')
# from matplotlib import pyplot as plt
#
# plt.legend()
#
# # print("Step: {} Time: {}".format(step, time))
# # plt.title(time.strftime('%Y-%m-%d %H:%M:%S'))
# # plt.gca().set_xlabel('Eastings, [m]')
# # plt.gca().set_ylabel('Northings, [m]')
# # plt.gca().set_xlim([-rng_cutoff, rng_cutoff])
# # plt.gca().set_ylim([-rng_cutoff, rng_cutoff])
# # name = 'image' + str(step).zfill(6)
# # fig.savefig('img/{}.png'.format(name), dpi=192)
# # plt.pause(0.05)
# # plt.clf()
#
# smoother_here = KalmanSmoother(transition_model=None)
# track_smoothed = smoother_here.smooth(track_forward)
# plot_tracks([track_smoothed], color='y', label='1st IPLF smoothing')