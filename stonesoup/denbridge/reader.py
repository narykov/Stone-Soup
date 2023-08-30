from ..reader.file import BinaryFileReader
from ..reader.base import FrameReader
from ..buffered_generator import BufferedGenerator
from ..types.sensordata import ImageFrame
from ..base import Property
from datetime import datetime, timedelta
import numpy as np
from scipy import interpolate


class BinaryFileReaderRAW(BinaryFileReader, FrameReader):
    """ Inherits from FrameReader so, we can potentially use the CFAR and/or CCL feeders (see
    https://github.com/narykov/Stone-Soup/blob/398e31bf4c0615f54aa768c49d1a0115d381896c/stonesoup/feeder/image.py#L15)
    if we wish to. """

    datashape: tuple = Property(doc="The size of data array in y and x coordinates")
    datatype: np.dtype = Property(doc="Data type objects")
    start_time: datetime = Property(doc="Data type objects")
    time_step: datetime = Property(doc="timedelta")
    rng_min: float = Property(doc="Min range in radar data")
    rng_instrumental: float = Property(doc="Max range in radar data")
    rng_cutoff: float = Property(doc="Cutoff range (to extract a subset of data)")


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._count = np.prod(self.datashape)

    def _polar2Im(self, polar_data):
        """A method that interpolates data in range-bearing cells onto the 2D plane"""

        def _rng(x, y):
            return np.sqrt(x ** 2 + y ** 2)

        def _bearing(x, y):
            angles = np.arctan2(x, y)
            out = np.where(angles > np.pi/2, angles - 2 * np.pi, angles)  #
            return out

        b_n = self.datashape[0]
        b_delta = 2 * np.pi / b_n  # assuming the same angle is not measured twice
        r_delta = (self.rng_instrumental - self.rng_min) / b_n

        b = np.arange(-3*np.pi/2, np.pi / 2, b_delta)
        r = np.arange(self.rng_min, self.rng_instrumental, r_delta)

        # setting up a grid of query points
        cutoff_rng = self.rng_cutoff if self.rng_cutoff < self.rng_instrumental else self.rng_instrumental
        x = np.arange(-cutoff_rng, cutoff_rng, r_delta)
        bb = _bearing(x[:, None], x[None, :])  # bearings corresponding to the grid
        rr = _rng(x[:, None], x[None, :])  # ranges corresponding to the grid

        interp = interpolate.RegularGridInterpolator((b, r), np.flip(polar_data, axis=0),
                                                     bounds_error=False, fill_value=np.nan)

        return interp((bb, rr))

    @BufferedGenerator.generator_method
    def frames_gen(self):
        with self.path.open('rb') as f:
            timestamp = self.start_time
            while True:
                # noinspection PyTypeChecker
                vector = np.fromfile(f, count=self._count, dtype=self.datatype)
                if vector.size == self._count:
                    pixels = vector.reshape(self.datashape)  # equivalent of Denbridge Marine's output
                    # technically, this corresponds to the B-scope display in radar
                    # https://en.wikipedia.org/wiki/Radar_display#B-Scope
                    pixels = self._polar2Im(pixels)
                    frame = ImageFrame(pixels=pixels, timestamp=timestamp)  # turns it into Stone Soup object
                    timestamp += self.time_step
                else:
                    break

                yield frame


class BinaryFileReaderFrames(BinaryFileReaderRAW):
    temporal_smooth: int = Property(doc="The number of frames to smooth/average over")

    @BufferedGenerator.generator_method
    def frames_gen(self):
        with self.path.open('rb') as f:
            timestamp = self.start_time
            frames = []
            eof = False
            while True:
                if len(frames) > 0:
                    frames.pop(0)  # dropping the first element for the sliding window



                while len(frames) < self.temporal_smooth:
                    # noinspection PyTypeChecker
                    vector = np.fromfile(f, count=self._count, dtype=self.datatype)
                    if vector.size < self._count:
                        eof = True
                        break
                    pixels = vector.reshape(self.datashape)  # equivalent of Denbridge Marine's output
                    frame = ImageFrame(pixels=pixels, timestamp=timestamp)  # turns it into Stone Soup object
                    frames.append(frame)
                    timestamp += self.time_step

                if eof is True:
                    break

                yield frames
