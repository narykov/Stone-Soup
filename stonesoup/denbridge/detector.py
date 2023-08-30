import cv2
import numpy as np
from ..base import Property
from ..buffered_generator import BufferedGenerator
from ..detector.base import Detector
from ..types.array import StateVector
from ..types.detection import Detection
from ..models.measurement.nonlinear import CartesianToBearingRange
from ..denbridge.reader import BinaryFileReaderFrames


class ObjectDetector(Detector):
    sensor: BinaryFileReaderFrames = Property(doc="Source of sensor data")
    min_block_size: int = Property(doc="Minimum block size")
    fgbg: cv2.bgsegm.BackgroundSubtractorMOG = Property(doc="Background Subtractor")
    measurement_model: CartesianToBearingRange = Property(doc="Measurement model")
    @BufferedGenerator.generator_method
    def detections_gen(self):
        # noinspection PyTypeChecker
        yield from self._detections_gen()

    def _detections_gen(self):
        for frames in self.sensor:
            detections = self._get_detections_from_frames(frames)
            timestamp = frames[-1].timestamp
            yield timestamp, detections

    def _get_detections_from_frames(self, frames):
        imgs = []
        for frame in frames:
            img = frame.pixels
            img_interest = img[0:2048, 0:1000]
            img_interest = cv2.rotate(img_interest, cv2.ROTATE_90_COUNTERCLOCKWISE)
            imgs.append(img_interest[np.newaxis, :, :])

        img_history = np.concatenate(imgs, axis=0)
        img = np.uint8(np.mean(img_history, axis=0))
        img_interest_blurred = cv2.GaussianBlur(img, (7, 7), cv2.BORDER_DEFAULT)

        fgmask = self.fgbg.apply(img_interest_blurred)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, (7, 7))

        contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        valid_blocks = []
        valid_detect_pnts = []
        detections = set()
        for _i in range(len(contours)):
            if contours[_i].shape[0] >= self.min_block_size:
                valid_blocks.append(contours[_i])
                valid_detect_pnts.append((np.mean(contours[_i][:, 0, 0]), np.mean(contours[_i][:, 0, 1])))
                # store detections and time index
                detection_x = valid_detect_pnts[-1][0]
                detection_y = valid_detect_pnts[-1][1]
                d_size = contours[_i].shape[0]

                b_n = 4096
                theta = np.pi/2 - detection_x * ( 2 * np.pi / b_n)
                rho = (1000-detection_y) * (1852*32 / b_n)

                detection = Detection(
                    state_vector=StateVector([[theta], [rho]]),
                    timestamp=frames[-1].timestamp,
                    measurement_model=self.measurement_model,
                    metadata={'d_size': d_size}
                )
                detections.add(detection)

        return detections
