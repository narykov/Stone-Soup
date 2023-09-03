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
from stonesoup.denbridge.utils import plot_tracks, plot_detections


def main():
    # Radar coordinates '53.26.555N,3.02.426W'
    # Data description
    path = Path('src/fn2.raw')
    datashape = (4096, 4096)
    datatype = np.uint8  # the file stores data in 'uint8', unsigned 8-bit integer
    start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)    # use own time since no time in raw data
    time_step = timedelta(seconds=3)
    rng_min = 0
    rng_instrumental_nm = 32  # in nautical miles, 1 NM = 1852 m
    nm = 1852  # metres in nautical mile
    rng_instrumental = rng_instrumental_nm * nm
    rng_cutoff = 5000  # how far we wish to see with the radar
    n_batches_cutoff = None  # how many track updates we wish to observe

    temporal_smooth = 8  # number of historical images are used for smoothing
    reader = BinaryFileReaderFrames(
        path=path,
        datashape=datashape,
        datatype=datatype,
        start_time=start_time,
        time_step=time_step,
        rng_min=rng_min,
        rng_instrumental=rng_instrumental,
        rng_cutoff=rng_cutoff,
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
    rng_delta = (rng_instrumental - rng_min)/datashape[0]
    rng_pixels = np.floor(rng_cutoff/rng_delta).astype(int)  # how many

    detector = ObjectDetector(
        sensor=reader,
        min_block_size=min_block_size,
        fgbg=fgbg,
        measurement_model=measurement_model,
        rng_pixels=rng_pixels
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
    fig, ax = plt.subplots()
    for step, (time, current_tracks) in enumerate(tracker, 0):
        pixels = reader.sensor_data.pixels
        im = plt.imshow(pixels, interpolation='none', origin='lower', cmap='jet',
                   extent=[-rng_cutoff, rng_cutoff, -rng_cutoff, rng_cutoff], vmin=0, vmax=255)
        cbar = plt.colorbar(im, orientation='vertical')
        # cbar.set_label('Receiver units')
        plot_detections(detector.detections)
        tracks.update(current_tracks)
        plot_tracks(current_tracks)
        print("Step: {} Time: {}".format(step, time))
        plt.title(time.strftime('%Y-%m-%d %H:%M:%S'))
        plt.gca().set_xlabel('Eastings, [m]')
        plt.gca().set_ylabel('Northings, [m]')
        plt.gca().set_xlim([-rng_cutoff, rng_cutoff])
        plt.gca().set_ylim([-rng_cutoff, rng_cutoff])
        name = 'image' + str(step).zfill(6)
        fig.savefig('img/{}.png'.format(name), dpi=192)
        plt.pause(0.05)
        plt.clf()


    # from stonesoup.plotter import AnimatedPlotterly
    # plotter = AnimatedPlotterly(timesteps, tail_length=0.3)
    # plotter.plot_measurements(all_measurements, [0, 2])
    # plotter.plot_tracks(tracks, [0, 2], uncertainty=True)
    # plotter.fig.update_yaxes(scaleanchor="x", scaleratio=1)
    # plotter.fig.update_xaxes(range=[-rng_cutoff, rng_cutoff])
    # plotter.fig.update_yaxes(range=[-rng_cutoff, rng_cutoff])
    # plotter.fig.show()


if __name__ == "__main__":
    main()
