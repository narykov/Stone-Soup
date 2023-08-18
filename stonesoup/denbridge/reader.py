from ..reader.file import BinaryFileReader
from ..reader.base import FrameReader
from ..buffered_generator import BufferedGenerator
from ..types.sensordata import ImageFrame
from ..base import Property
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path


class BinaryFileReaderRAW(BinaryFileReader):
    datashape: tuple = Property(doc="Shape")
    datatype: np.dtype = Property(doc="Data type objects")
    start_time: datetime = Property(doc="Data type objects")
    time_step: datetime = Property(doc="timedelta")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._count = np.prod(self.datashape)

    @BufferedGenerator.generator_method
    def frames_gen(self):
        with self.path.open('rb') as f:
            timestamp = self.start_time
            while True:
                # noinspection PyTypeChecker
                vector = np.fromfile(f, count=self._count, dtype=self.datatype)
                if vector.size == self._count:
                    pixels = vector.reshape(self.datashape).T  # equivalent of DM's output
                    frame = ImageFrame(pixels=pixels, timestamp=timestamp)  # turn it into Stone Soup object
                    timestamp += self.time_step
                else:
                    break

                yield frame
