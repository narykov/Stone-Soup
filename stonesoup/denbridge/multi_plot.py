import copy
import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import matplotlib.patheffects as pe


from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity
from stonesoup.simulator.simple import MultiTargetGroundTruthSimulator
from stonesoup.types.state import GaussianState

from .utils import plot_detections


# # High level settings
# # -------------------
# num_subplots = 20                   # Number of subplots to show
# grid_size = (5, 7)                  # Size of the plotting grid (rows, cols)
# main_plot_size = (3, 5)             # Size of the center plot (rows, cols)
# V_BOUNDS = np.array([[-20, 20],     # (x_min, x_max)
#                      [-10, 10]])    # (y_min, y_max)
num_subplots = 24                   # Number of subplots to show
grid_size = (7, 7)                  # Size of the plotting grid (rows, cols)
main_plot_size = (5, 5)             # Size of the center plot (rows, cols)
# rng_cutoff = 5000  # how far we wish to see with the radar
# V_BOUNDS = np.array([[-rng_cutoff, rng_cutoff],   # (x_min, x_max)
#                      [-rng_cutoff, rng_cutoff]])  # (y_min, y_max)
# V_BOUNDS = np.array([[-rng_cutoff, rng_cutoff],   # (x_min, x_max)
#                      [-rng_cutoff, rng_cutoff]])  # (y_min, y_max)


def reset_axis(ax, V_BOUNDS, main=False):
    """Reset the axis object. If main is True, set the axis limits as well.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis object to reset
    main : bool
        Whether the axis is the main plot or a subplot
    """
    ax.cla()
    ax.tick_params(
        axis='both',
        which='both',
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False)
    if main:
        ax.set_xlim(*V_BOUNDS[0])
        ax.set_ylim(*V_BOUNDS[1])


def get_subplot_idx(plot_data, track):
    """Get the subplot index for a given track. If the track is not yet plotted, assign a new
    subplot index. If all subplots are already used, return None.

    Returns
    -------
    subplot_idx : int or None
        The subplot index for the track, or None if all subplots are already used
    """
    track_to_sub_idx = plot_data['track_to_sub_idx']
    try:
        # Get the subplot index for the track. If the track is not yet plotted, this will raise a
        # KeyError, which we catch and handle below.
        subplot_idx = track_to_sub_idx[track.id]
    except KeyError:
        # Get the first available subplot index, or None if all subplots are already used
        subplot_idx = next((i for i in range(num_subplots) if i not in track_to_sub_idx.values()),
                           None)
        if subplot_idx is not None:
            # Assign the subplot index to the track
            track_to_sub_idx[track.id] = subplot_idx
    return subplot_idx


def get_subplot_coords(idx):
    """Get the row and column index for a subplot index.

    In this example, we have a 5x7 grid of subplots, and the main plot is placed in the center. The
    subplots are numbered from 0 to 19, with 0 being the top left subplot. The subplots are
    arranged in a clockwise spiral, starting from the top left corner. The subplots are numbered as
    follows:

          i  0  1  2  3  4  5  6
       j  _______________________
       0  |  0  1  2  3  4  5  6
       1  | 19 +-------------+ 7
       2  | 18 |  Main plot  | 8
       3  | 17 +-------------+ 9
       4  | 16 15 14 13 12 11 10

    The returned index for the last row and column is -1. Hence, the bottom left subplot has coords
    (-1, 0), the top right subplot (0, -1), and the bottom right subplot (-1, -1).
    """
    # if idx < 7:  # Top row (0-6)
    #     i = 0
    #     j = idx if idx < 6 else -1  # Return -1 for the last column
    # elif idx < 10:  # Right column (7-9)
    #     i = idx - 6
    #     j = -1
    # elif idx < 17:  # Bottom row (10-16)
    #     i = -1
    #     j = 16 - idx if idx > 10 else -1
    # else:  # Left column (17-19)
    #     i = 20 - idx
    #     j = 0
    # return i, j
    if idx < 7:  # Top row (0-6)
        i = 0
        j = idx if idx < 6 else -1  # Return -1 for the last column
    elif idx < 12:  # Right column (7-9)
        i = idx - 6
        j = -1
    elif idx < 19:  # Bottom row (10-16)
        i = -1
        j = 18 - idx if idx > 12 else -1
    else:  # Left column (17-19)
        i = 24 - idx
        j = 0
    return i, j


def plot_track(track, ax, zoom=False, margin=0.5):
    """Plot a track on a given axis. If `zoom` is True, zoom in on the last state of the track,
    leaving a margin of size `margin` around the state"""
    data = np.array([state.state_vector.squeeze() for state in track]).T
    linestyle = '-' if len(track) > 1 else '.'  # Use a line for tracks with more than one state
    ax.plot(data[0, :], data[2, :], f'{linestyle}', color='xkcd:white', markersize=1,
            path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()])
    if zoom:
        ax.set_xlim(data[0, -1] - margin, data[0, -1] + margin)
        ax.set_ylim(data[2, -1] - margin, data[2, -1] + margin)


def plot_tracks(tracks, plot_data, V_BOUNDS, margin=0.5*500, **kwargs):
    """Plot a list of tracks on the main plot and subplots."""

    lims = {'xlim': V_BOUNDS[0], 'ylim': V_BOUNDS[1]}

    ax = plot_data['main']['ax']
    n_tracks = len(tracks)
    props = dict(boxstyle='square', facecolor='white', alpha=0.9)
    # place a text box in upper left in axes coords
    plt.sca(ax)
    plt.imshow(kwargs['pixels'], interpolation='none', origin='lower', cmap='jet',
                    extent=[*lims['xlim'], *lims['ylim']], vmin=0, vmax=255)
    # cbar = plt.colorbar(im, orientation='vertical')
    step = kwargs['step']
    timestamp = kwargs['timestamp']
    textstr = f'{step:03d} | {timestamp.time()} | Subplot capacity: {num_subplots:02d} | Alive tracks: {n_tracks:02d}'
    plot_detections(kwargs['detector'].detections)

    ax.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
    # ax.text(0.05, 0.05, str(kwargs['timestamp']), transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)

    for track in tracks:
        # Plot on main plot
        ax = plot_data['main']['ax']
        plot_track(track, ax, zoom=False)

        # Plot on subplot
        subplot_idx = get_subplot_idx(plot_data, track)
        # If there is no subplot available, skip this track
        if subplot_idx is None:
            continue
        ax = plot_data['sub'][subplot_idx]['ax']
        reset_axis(ax, V_BOUNDS)
        plt.sca(ax)
        plt.imshow(kwargs['pixels'], interpolation='none', origin='lower', cmap='jet',
                   extent=[*lims['xlim'], *lims['ylim']], vmin=0, vmax=255)
        plot_detections(kwargs['detector'].detections)
        plot_track(track, ax, zoom=True, margin=margin)
        props = dict(boxstyle='square', facecolor='white', alpha=0.9)
        order_appear = kwargs['tracks_id_db'][track.id]
        textstr = f'{order_appear}'
        # place a text box in upper left in axes coords
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
        draw_connection(plot_data, track, subplot_idx)


def draw_connection(plot_data, track, sub_idx, color='white', linewidth=0.5, linestyle='--'):
    """Draw a connection line between the main plot and the subplot for a given track."""
    ax1 = plot_data['main']['ax']
    ax2 = plot_data['sub'][sub_idx]['ax']
    xy = track.state_vector[[0, 2]].flatten()

    # Get the row and column index for the subplot
    i, j = get_subplot_coords(sub_idx)

    # Get the coordinates of the subplot as a fraction of the axis size
    # [0, 0] is the bottom left corner, [1, 1] is the top right corner
    xy_sub = [0.5, 0.5]  # Center
    if j == 0:
        # Left
        xy_sub[0] = 1
    elif j == -1:
        # Right
        xy_sub[0] = 0
    if i == 0:
        # Top
        xy_sub[1] = 0
    elif i == -1:
        # Bottom
        xy_sub[1] = 1

    con = ConnectionPatch(xyA=xy, xyB=xy_sub, coordsA="data", coordsB="axes fraction", axesA=ax1,
                          axesB=ax2, color=color, linewidth=linewidth, linestyle=linestyle)
    ax1.add_artist(con)


def setup_plot(grid_size, main_plot_size, num_subplots, V_BOUNDS):
    """Set up the plot and return the figure and plot_data dictionary."""
    fig = plt.figure(figsize=(8, 8))

    # Initiate plotting grid with 5 rows and 7 columns
    # The center plot will span 3 rows and 5 columns, the rest will be subplots of size 1x1
    gs = fig.add_gridspec(*grid_size)

    # Create the main axis object. This will be the center plot. We assume that the main plot will
    # occupy the center 3x5 grid cells.
    margin_rows = (grid_size[0] - main_plot_size[0]) // 2
    margin_cols = (grid_size[1] - main_plot_size[1]) // 2
    ax = fig.add_subplot(gs[margin_rows:-margin_rows, margin_rows:-margin_cols])
    ax.set_facecolor('xkcd:black')
    reset_axis(ax, V_BOUNDS, main=True)

    # This dictionary will hold all the data needed for plotting. For each plot, we maintain a
    # dictionary with the axis object and a list of artists that will be updated. You can add
    # additional data to this dictionary if needed.
    #
    # The artists are the objects that will be plotted and updated. For example, for the center
    # plot, we could store artists for all tracks and detections, while for the subplots, just the
    # individual track. This can be used to speed up the plotting by only updating the artists that
    # have changed (i.e. not clearing the axis using `ax.cla()` and redrawing everything). In this
    # example, we will just clear the axis and redraw everything, but this is something to keep in
    # mind for larger simulations.
    plot_data = {
        # The center plot is the main plot
        'main': {
            'ax': ax,  # The axis object
            'arts': [],  # The matplotlib artists that will be updated
        },
        # The subplots are the smaller plots around the center plot.
        'sub': [],
        # Dictionary to keep track of which track is plotted on which subplot
        'track_to_sub_idx': dict(),
    }
    # Create the subplots
    for idx in range(num_subplots):
        # Get the row and column index for the subplot
        i, j = get_subplot_coords(idx)
        # Create the axis object and add it to the plot_data dictionary
        ax = fig.add_subplot(gs[i, j])
        ax.set_facecolor('xkcd:black')
        reset_axis(ax, V_BOUNDS)
        plot_data['sub'].append({
            'ax': ax,
            'arts': [],
        })

    # Remove the space between the subplots
    plt.tight_layout()

    return fig, plot_data


def main():
    # Create a simple multi-target ground truth simulator
    gnd_sim = MultiTargetGroundTruthSimulator(
        transition_model=CombinedLinearGaussianTransitionModel(2*[ConstantVelocity(0.0001)]),
        initial_state=GaussianState(np.array([0, 0, 0, 0]), covar=np.diag([1, 0.1, 1, 0.1]),
                                    timestamp=datetime.datetime.now()),
        timestep=datetime.timedelta(seconds=1),
        number_steps=100,
        birth_rate=0.6,
        death_probability=0.05,
        seed=1996
    )

    # Setup plot
    fig, plot_data = setup_plot(grid_size, main_plot_size, num_subplots)
    plt.ion()
    plt.pause(0.1)

    # Run simulation
    last_tracks = set()  # Store the last set of tracks, so we can handle new/deleted tracks
    for timestamp, gnd_paths in gnd_sim:

        # Print debug info
        print(f'{timestamp} - Num tracks: {len(gnd_paths)}')

        # Reset main plot
        reset_axis(plot_data['main']['ax'], main=True)

        # Clear subplots for deleted tracks
        for track in last_tracks:
            if track not in gnd_paths:
                subplot_idx = get_subplot_idx(plot_data, track)
                ax = plot_data['sub'][subplot_idx]['ax']
                reset_axis(ax)
                del plot_data['track_to_sub_idx'][track.id]

        # Plot tracks
        plot_tracks(gnd_paths, plot_data)

        # Store tracks for next iteration. It is important to make a copy here, otherwise we will
        # be storing a reference to the set stored in the gnd_sim object, which will be updated in
        # the next iteration.
        last_tracks = copy.copy(gnd_paths)

        # Update plot
        # plt.pause(0.1)


if __name__ == '__main__':
    main()
