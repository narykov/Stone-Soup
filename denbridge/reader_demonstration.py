import numpy as np
from pathlib import Path
from stonesoup.denbridge.reader import BinaryFileReaderRAW
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib



def main():
    # Instrumental range

    instrumental_azimuth = 2 * np.pi

    # Radar coordinates
    radar_loc = '53.26.555N,3.02.426W'

    # Data description
    path = Path('src/fn2.raw')
    y1 = 4096
    x1 = 4096
    datashape = (y1, x1)
    datatype = np.uint8  # the file stores data in 'uint8', unsigned 8-bit integer
    start_time = datetime.now().replace(microsecond=0)  # use own time since no time in raw data
    time_step = timedelta(seconds=3)
    rng_min = 0
    rng_instrumental_nm = 32  # in nautical miles, 1 NM = 1852 m
    nm = 1852  # metres in nautical mile
    rng_instrumental = rng_instrumental_nm * nm
    rng_cutoff = 5000

    # Configuring the reader
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

    # Data visualisation
    matplotlib.use('Agg')
    chunk_generator = reader.frames_gen()

    for i in range(700):
        chunk = next(chunk_generator)
        pixels = chunk.pixels
        timestamp = chunk.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        fig = plt.figure()
        plt.imshow(pixels, interpolation='none', origin='lower', cmap='jet',
                   extent=[-rng_cutoff, rng_cutoff, -rng_cutoff, rng_cutoff], vmin=0, vmax=255)
        plt.title(timestamp)
        plt.gca().set_xlabel('Eastings, [m]')
        plt.gca().set_ylabel('Northings, [m]')
        plt.colorbar(ax=plt.gca())
        # plt.show()
        # plt.ion()
        name = 'image' + str(i).zfill(6)
        fig.savefig('img/{}.png'.format(name), dpi=192)
        plt.close()


if __name__ == "__main__":
    main()
