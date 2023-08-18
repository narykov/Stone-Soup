import numpy as np
from pathlib import Path
from stonesoup.denbridge.reader import BinaryFileReaderRAW
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


def main():
    # Data description
    path = Path('src/fn2.raw')
    y1 = 4096
    x1 = 4096
    datashape = (y1, x1)
    datatype = np.uint8  # the file stores data in 'uint8', unsigned 8-bit integer
    start_time = datetime.now().replace(microsecond=0)  # use own time since no time in raw data
    time_step = timedelta(seconds=10)

    # Configuring the reader
    reader = BinaryFileReaderRAW(
        path=path,
        datashape=datashape,
        datatype=datatype,
        start_time=start_time,
        time_step=time_step
    )

    # Data visualisation
    chunk_generator = reader.frames_gen()
    chunk = next(chunk_generator)
    pixels = chunk.pixels
    timestamp = chunk.timestamp.strftime('%Y-%m-%d %H:%M:%S')
    fig = plt.figure()
    plt.imshow(pixels, interpolation='none', origin='lower', cmap='jet', vmin=0, vmax=255)
    plt.title(timestamp)
    plt.gca().set_xlabel('Bearing info')
    plt.gca().set_ylabel('Range info')
    plt.colorbar(ax=plt.gca())
    plt.show()
    fig.savefig('img/{}.png'.format(timestamp), dpi=fig.dpi)


if __name__ == "__main__":
    main()
