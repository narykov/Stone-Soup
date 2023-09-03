import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def plot_detections(detections):
    xs = []
    ys = []
    for detection in detections:
        detect_state = detection.measurement_model.inverse_function(detection)
        xs.append(detect_state[0])
        ys.append(detect_state[2])
    plt.scatter(x=xs, y=ys, s=80, facecolors='none', edgecolors='r')

def plot_tracks(tracks, ax=None):
    if ax is None:
        ax = plt.gca()

    for track in tracks:
        x = [state.state_vector[0] for state in track.states]
        y = [state.state_vector[2] for state in track.states]
        plt.plot(x, y, 'w-')
        plot_ellipse(track.mean[[0, 2], :], track.covar[[0, 2], :][:, [0, 2]], facecolor='none',
                         linestyle='-', edgecolor='w', ax=ax, linewidth=1)

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