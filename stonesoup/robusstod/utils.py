import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


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