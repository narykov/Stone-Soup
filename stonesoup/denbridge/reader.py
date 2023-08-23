from ..reader.file import BinaryFileReader
from ..reader.base import FrameReader
from ..buffered_generator import BufferedGenerator
from ..types.sensordata import ImageFrame
from ..base import Property
from datetime import datetime, timedelta
import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp2d, griddata

class BinaryFileReaderRAW(BinaryFileReader, FrameReader):
    """ Inherits from FrameReader so, we can potentially use the CFAR and/or CCL feeders (see
    https://github.com/narykov/Stone-Soup/blob/398e31bf4c0615f54aa768c49d1a0115d381896c/stonesoup/feeder/image.py#L15)
    if we wish to. """

    datashape: tuple = Property(doc="The size of data array in y and x coordinates")
    datatype: np.dtype = Property(doc="Data type objects")
    start_time: datetime = Property(doc="Data type objects")
    time_step: datetime = Property(doc="timedelta")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._count = np.prod(self.datashape)

    def _polar2Im(self, polar_data):
        """A method that interpolates data in range-bearing cells onto the 2D plane"""
        pass

    @BufferedGenerator.generator_method
    def frames_gen(self):
        with self.path.open('rb') as f:
            timestamp = self.start_time
            while True:
                # noinspection PyTypeChecker
                vector = np.fromfile(f, count=self._count, dtype=self.datatype)
                if vector.size == self._count:
                    pixels = vector.reshape(self.datashape).T  # equivalent of Denbridge Marine's output
                    # pixels = self._polar2Im(pixels)
                    frame = ImageFrame(pixels=pixels, timestamp=timestamp)  # turns it into Stone Soup object
                    timestamp += self.time_step
                else:
                    break

                yield frame
