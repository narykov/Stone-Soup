import numpy as np
from pathlib import Path
from stonesoup.denbridge.reader import BinaryFileReaderRAW, BinaryFileReaderFrames
from stonesoup.denbridge.detector import ObjectDetector
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib
import cv2
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange



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
    rng_cutoff = 5000

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
        temporal_smooth=temporal_smooth
    )
    min_block_size = 8  # the minimum size of the detected block
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=20, nmixtures=4, backgroundRatio=0.8, noiseSigma=9.0)
    measurement_model = CartesianToBearingRange(
        ndim_state=4,
        mapping=(0, 2),
        noise_covar=np.diag([np.deg2rad(1), 10])
    )
    detector = ObjectDetector(
        sensor=reader,
        min_block_size=min_block_size,
        fgbg=fgbg,
        measurement_model=measurement_model
    )

    timesteps = []
    all_measurements = []
    for timestamp, detections in detector.detections_gen():
        # print(detections)
        timesteps.append(timestamp)
        all_measurements.append(detections)

    # from stonesoup.plotter import AnimatedPlotterly
    # plotter = AnimatedPlotterly(timesteps, tail_length=0.0001)
    #
    # plotter.plot_measurements(all_measurements, [0, 2])
    # plotter.fig.update_yaxes(scaleanchor="x", scaleratio=1)
    # plotter.fig.show()

    # Data visualisation
    matplotlib.use('Agg')
    reader = BinaryFileReaderRAW(
        path=path,
        datashape=datashape,
        datatype=datatype,
        start_time=start_time,
        time_step=time_step,
        rng_min=rng_min,
        rng_instrumental=rng_instrumental,
        rng_cutoff=rng_cutoff
    )
    chunk_generator = reader.frames_gen()

    for i in range(700):
        chunk = next(chunk_generator)
        pixels = chunk.pixels
        timestamp = chunk.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        fig = plt.figure()
        plt.imshow(pixels, interpolation='none', origin='lower', cmap='jet',
                   extent=[-rng_cutoff, rng_cutoff, -rng_cutoff, rng_cutoff], vmin=0, vmax=255)
        plt.colorbar(ax=plt.gca())

        if i > 6:
            dets = all_measurements.pop(0)
            xs, ys = [], []
            for det in dets:
                sv = measurement_model.inverse_function(det)
                xs.append(sv[0])
                ys.append(sv[2])
            plt.scatter(x=xs, y=ys, s=80, facecolors='none', edgecolors='r')

        plt.title(timestamp)
        plt.gca().set_xlim([-rng_cutoff, rng_cutoff])
        plt.gca().set_ylim([-rng_cutoff, rng_cutoff])
        plt.gca().set_xlabel('Eastings, [m]')
        plt.gca().set_ylabel('Northings, [m]')
        # plt.show()
        # plt.ion()
        name = 'image' + str(i).zfill(6)
        fig.savefig('img/{}.png'.format(name), dpi=192)
        plt.close()


if __name__ == "__main__":
    main()
