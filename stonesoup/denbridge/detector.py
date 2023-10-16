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
    datashape: tuple = Property(doc="The size of data array in y and x coordinates")
    rng_instrumental: int = Property(doc="Maximum number of range pixels to grab")
    lims: dict = Property(doc="Cutoff range (to extract a subset of data)")

    @BufferedGenerator.generator_method
    def detections_gen(self, **kwargs):
        # noinspection PyTypeChecker
        yield from self._detections_gen(**kwargs)

    def _detections_gen(self, **kwargs):
        for frames in self.sensor:
            detections = self._get_detections_from_frames(frames, **kwargs)
            timestamp = frames[-1].timestamp
            yield timestamp, detections

    def _get_detections_from_frames(self, frames,  **kwargs):

        imgs = []
        for frame in frames:
            img = frame.pixels
            # max_pix = self.rng_pixels
            # img_interest = img[:, 0:max_pix]
            img_interest = img
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

                #TODO: sort out this ad hoc solution
                # b_n = 4096
                b_n = self.datashape[0]
                r_n = self.datashape[1]
                theta = np.pi/2 - detection_x * ( 2 * np.pi / b_n)
                rho = (self.rng_instrumental-detection_y) * (self.rng_instrumental / r_n)
                rho = (r_n-detection_y) * (self.rng_instrumental / r_n)

                detection = Detection(
                    state_vector=StateVector([[theta], [rho]]),
                    timestamp=frames[-1].timestamp,
                    measurement_model=self.measurement_model,
                    metadata={'d_size': d_size}
                )

                target_x, target_y = self.measurement_model.inverse_function(detection)[[0, 2]]
                xlim, ylim = self.lims['xlim'], self.lims['ylim']
                detection_in_x = True if xlim[0] <= target_x <= xlim[1] else False
                detection_in_y = True if ylim[0] <= target_y <= ylim[1] else False
                if detection_in_x * detection_in_y:
                    detections.add(detection)

        return detections
