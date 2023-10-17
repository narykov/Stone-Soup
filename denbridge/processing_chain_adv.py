import copy
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib import use
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.deleter.time import UpdateTimeDeleter
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.initiator.simple import MultiMeasurementInitiator
from stonesoup.measures import Mahalanobis
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, OrnsteinUhlenbeck
from stonesoup.predictor.kalman import UnscentedKalmanPredictor
from stonesoup.tracker.simple import MultiTargetTracker
from stonesoup.types.state import GaussianState
from stonesoup.updater.kalman import UnscentedKalmanUpdater

from stonesoup.denbridge.reader import BinaryFileReaderFrames
from stonesoup.denbridge.detector import ObjectDetector
from stonesoup.denbridge.multi_plot import get_subplot_idx, plot_tracks, reset_axis, setup_plot

# Radar coordinates '53.26.555N,3.02.426W'
# Data description
path = Path('src/sta_131023.raw')
datashape = (1024, 2048)  # 2048 cells in range, 1024 in azimuth
datatype = np.uint8  # the file stores data in 'uint8', unsigned 8-bit integer
date = datetime.strptime('2011-03-03', "%Y-%m-%d")  # in Simon L.'s vid it is 2019 for some reason
start_time = date.replace(hour=9, minute=41, second=41)
time_step = timedelta(seconds=1)
rng_min = 0
rng_instrumental_nm = 4  # in nautical miles, 1 NM = 1852 m
nm = 1852  # metres in nautical mile
rng_instrumental = rng_instrumental_nm * nm

# High level settings for viz.
# -------------------
num_subplots = 24                   # Number of subplots to show
grid_size = (7, 7)                  # Size of the plotting grid (rows, cols)
main_plot_size = (5, 5)             # Size of the center plot (rows, cols)
V_BOUNDS = np.array([[-0.5 * rng_instrumental, 0],   # (x_min, x_max)
                     [0, 0.4 * rng_instrumental]])  # (y_min, y_max)


def main():

    temporal_smooth = 8  # number of historical images are used for smoothing
    n_batches_cutoff = None  # how many track updates we wish to observe
    reader = BinaryFileReaderFrames(
        path=path,
        datashape=datashape,
        datatype=datatype,
        start_time=start_time,
        time_step=time_step,
        rng_min=rng_min,
        rng_instrumental=rng_instrumental,
        V_BOUNDS=V_BOUNDS,
        temporal_smooth=temporal_smooth,
        n_batches_cutoff=n_batches_cutoff
    )

    min_block_size = 8  # the minimum size of the detected block
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=20, nmixtures=4, backgroundRatio=0.8, noiseSigma=9.0)
    measurement_model = CartesianToBearingRange(
        ndim_state=4,
        mapping=(0, 2),
        noise_covar=np.diag([np.deg2rad(1), 100])
    )

    detector = ObjectDetector(
        sensor=reader,
        min_block_size=min_block_size,
        fgbg=fgbg,
        measurement_model=measurement_model,
        datashape=datashape,
        rng_instrumental=rng_instrumental,
        V_BOUNDS=V_BOUNDS
    )

    transition_model = CombinedLinearGaussianTransitionModel((OrnsteinUhlenbeck(0.5, 1e-4),
                                                              OrnsteinUhlenbeck(0.5, 1e-4)))
    predictor = UnscentedKalmanPredictor(transition_model)
    updater = UnscentedKalmanUpdater()
    hypothesiser = DistanceHypothesiser(predictor, updater, measure=Mahalanobis(), missed_distance=3)
    data_associator = GNNWith2DAssignment(hypothesiser)
    deleter = UpdateTimeDeleter(time_since_update=timedelta(seconds=10))
    initiator = MultiMeasurementInitiator(GaussianState(
        np.array([[0], [0], [0], [0]]), np.diag([10 ** 2, 1 ** 2, 10 ** 2, 1 ** 2])),
        measurement_model=measurement_model,
        deleter=deleter,
        data_associator=data_associator,
        updater=updater,
        min_points=3
    )
    tracker = MultiTargetTracker(
        initiator=initiator,
        deleter=deleter,
        detector=detector,
        data_associator=data_associator,
        updater=updater,
    )

    # use('Agg')  # hides the figures
    tracks = set()
    tracks_id_db = {}

    # fig, ax = plt.subplots()
    fig, plot_data = setup_plot(grid_size, main_plot_size, num_subplots, V_BOUNDS)
    fig.patch.set_facecolor('xkcd:black')
    plt.ion()

    last_tracks = set()  # Store the last set of tracks, so we can handle new/deleted tracks

    for step, (timestamp, current_tracks) in enumerate(tracker, 0):

        for track in current_tracks:
            if track.id not in tracks_id_db.keys():
                tracks_id_db[track.id] = len(tracks_id_db)  # {'track.id': order_of_appearance}

        # Print debug info
        print(f'{timestamp} - Step: {str(step).zfill(3)} - Num tracks: {len(current_tracks)}')
        # Reset main plot
        reset_axis(plot_data['main']['ax'], V_BOUNDS, main=True)

        # Clear subplots for deleted tracks
        for track in last_tracks:
            if track not in current_tracks:
                subplot_idx = get_subplot_idx(plot_data, track)
                if subplot_idx is None:
                    print('If all subplots are already used, return None')

                if subplot_idx is not None:
                    ax = plot_data['sub'][subplot_idx]['ax']
                    reset_axis(ax, V_BOUNDS)
                    del plot_data['track_to_sub_idx'][track.id]

        # Plot tracks
        pixels = reader.sensor_data.pixels
        plot_tracks(current_tracks, plot_data, V_BOUNDS,
                    tracks_id_db=tracks_id_db, pixels=pixels, timestamp=timestamp, step=step, detector=detector)
        # Store tracks for next iteration. It is important to make a copy here, otherwise we will
        # be storing a reference to the set stored in the gnd_sim object, which will be updated in
        # the next iteration.
        last_tracks = copy.copy(current_tracks)
        tracks.update(current_tracks)
        plt.title(timestamp.strftime('%Y-%m-%d %H:%M:%S'))
        plt.gca().set_xlabel('Eastings, [m]')
        plt.gca().set_ylabel('Northings, [m]')
        name = 'image' + str(step).zfill(6)
        fig.savefig('img/{}.png'.format(name), dpi=192)
        plt.pause(0.05)


if __name__ == "__main__":
    main()
