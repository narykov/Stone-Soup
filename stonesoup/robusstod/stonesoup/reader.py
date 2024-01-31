import pyproj  # this package is not in Stone Soup requirements
import copy
import csv
import numpy as np
from pathlib import Path
import os

from ...base import Property
from ...buffered_generator import BufferedGenerator
from ...models.measurement.base import MeasurementModel
from ...reader.generic import _CSVReader, DetectionReader
from ...models.measurement.nonlinear import CartesianToElevationBearingRangeRate, CartesianToElevationBearing
from ...types.angle import Bearing, Elevation
from ...types.array import StateVector, CovarianceMatrix
from ...types.detection import Detection
from ...types.state import State
from ...reader.file import FileReader

# ROBUSSTOD SPECIFIC
from ..python_libs.parsers import CCtd_Tdm as cctd

# GODOT SPECIFIC
from godot.core import util, tempo, num, astro, autodif as ad, ipfwrap as ipf
from godot import cosmos
import godot.model.common as common


class StationObject(common.VectorTimeEvaluable):
    """
    Station Info; added by Alexey.
    """
    # TODO: needs to be integrated into the TDMReader class

    def __init__(self, uni, station):
        common.VectorTimeEvaluable.__init__(self)
        self.__uni = uni
        self.__station_point = self.__uni.frames.pointId(station)
        self.__lt_center_id = self.__uni.frames.pointId("Earth")
        self.__lt_axes_id = self.__uni.frames.axesId("GCRF")

    def eval(self, epoch):
        xstat = self.__uni.frames.vector6(self.__lt_center_id, self.__station_point, self.__lt_axes_id, epoch)
        return xstat


class TDMReader(FileReader):
    """A reader class for tdm files of measurement data."""
    osdm_folder_path: Path = Property(doc="Path to file to be opened. Str will be converted to path.")
    stationMeasDict: dict = Property(doc="Describes what measurements stations take.")
    data_folder_path: Path = Property(doc="Path to file to be opened. Str will be converted to path.")
    universe_path: str = Property(doc="Path to file to be opened.")

    # TODO: - harmonise the path variables
    # TODO: - consider extraction of the ground truth for the simulated objects
    # TODO: - generate entries for metadata field of individual measurements
    # TODO: - integrate StationObject into this reader
    # TODO: - consider integration into a Feeder so as to handle multiple TDM files

    def __init__(self, path, osdm_folder_path, stationMeasDict, data_folder_path, *args, **kwargs):
        if not isinstance(path, Path):
            path = Path(path)  # Ensure Path
        if not isinstance(osdm_folder_path, Path):
            osdm_folder_path = Path(osdm_folder_path)  # Ensure Path
        if not isinstance(data_folder_path, Path):
            data_folder_path = Path(data_folder_path)  # Ensure Path
        super().__init__(path, osdm_folder_path, stationMeasDict, data_folder_path, *args, **kwargs)
        self.CCSDS_DATETIME_FORMAT = '%Y-%m-%dT%H:%M:%S.%f'
        self.EQUATORIAL_EARTH_RADIUS = 6378.13655
        self.FLATTENING_PARAMETER = 1.0 / 298.25769
        self.m_in_km = 1000
        self.mapping = (0, 2, 4)
        self.velocity_mapping = (1, 3, 5)
        self.ndim_state = 6
        self.statList = self._get_statList(self.osdm_folder_path)
        self.tdmFile = cctd.TdmFile(str(self.path))
        self.stat = self.tdmFile.tdms[0].participant_1
        self.measType = self.stationMeasDict[self.stat]
        self.measDict = self.tdmFile.measurements_to_dict()
        self.timeScaleMeas = self.tdmFile.tdms[0].time_system
        self.tdm_data = self._get_tdm_data()
        self.uni_path = 'universe.yml'
        self.uni = cosmos.Universe(cosmos.util.load_yaml("universe.yml"))
        self.station_vector_object = StationObject(self.uni, self.stat)
        self.noise_covar = self._get_noise_covar()

    @staticmethod
    def rad2millideg(ang):
        return ang * 180 / num.Pi * 1000

    @staticmethod
    def deg2rad(ang):
        return ang * num.Pi / 180

    def _covar_telescope(self):
        # TODO: make it less embarrassing
        osdmFile = str(self.osdm_folder_path) + '/' + self.stat + '.osdm'
        with open(osdmFile) as myfile:
            for line in myfile:
                if line.startswith("ANGULAR_SIGMA"):
                    angular_sigma = np.deg2rad(float(line.split()[2]))
                    sigma_el = angular_sigma
                    sigma_az = angular_sigma

        return CovarianceMatrix(np.diag([sigma_el**2,
                                         sigma_az**2]))

    def _covar_radar(self):
        # TODO: make it less embarrassing
        osdmFile = str(self.osdm_folder_path) + '/' + self.stat + '.osdm'
        with open(osdmFile) as myfile:
            for line in myfile:
                if line.startswith("ANGULAR_SIGMA"):
                    angular_sigma = np.deg2rad(float(line.split()[2]))
                    sigma_el = angular_sigma
                    sigma_az = angular_sigma
                if line.startswith("RANGE_SIGMA"):
                    sigma_r = float(line.split()[2]) * self.m_in_km
                if line.startswith("DOPPLER_SIGMA"):
                    sigma_rr = float(line.split()[2]) * self.m_in_km

        return CovarianceMatrix(np.diag([sigma_el**2,
                                         sigma_az**2,
                                         sigma_r**2,
                                         sigma_rr**2]))

    def _get_noise_covar(self):
        # this info is available in an osdm file
        # potentially, we can rely on some of the GMV parsers
        # by looking into OSDM, monostatic has the same fields as bistatic

        if self.measType == 'RADEC':
            return self._covar_telescope()
        elif self.measType in ['MONOSTATIC', 'BISTATIC']:
            return self._covar_radar()


    def _get_statList(self, osdm_folder_path):
        """This function is adapted from processTdsGodot.ipynb using pyproj."""

        statList = {}
        osdmFolder = osdm_folder_path
        osdmList = [os.path.join(osdmFolder, file) for file in os.listdir(osdmFolder)]

        for osdmFile in osdmList:
            print(osdmFile)
            osdm = {}
            with open(osdmFile) as myfile:
                for line in myfile:
                    line = line.split(" = ")
                    value = line[1].split("[")[0]
                    value = value.replace('\n', '')
                    value = value.replace('\t', '')
                    osdm[line[0].strip()] = str(value.strip())
            statList[osdm['SITE_NAME']] = {}
            if 'LAT' not in osdm.keys():
                # using pyproj and following https://stackoverflow.com/a/65048302
                datum = osdm['REF_FRAME']
                transformer = pyproj.Transformer.from_crs(
                    {"proj": 'geocent', "ellps": datum, "datum": datum},
                    {"proj": 'latlong', "ellps": datum, "datum": datum},
                )
                x = float(osdm['X']) * self.m_in_km
                y = float(osdm['Y']) * self.m_in_km
                z = float(osdm['Z']) * self.m_in_km
                lon1, lat1, alt1 = transformer.transform(x, y, z, radians=True)
                # tested on RR01 data and it closesly matches, though not exactly, e.g. alt 84 vs 83.57 m

                statList[osdm['SITE_NAME']]['GEODETIC'] = [lat1, lon1, alt1]
            else:
                statList[osdm['SITE_NAME']]['GEODETIC'] = [self.deg2rad(float(osdm['LAT'])), self.deg2rad(float(osdm['LON'])),
                                                           float(osdm['ALT']) / 1000]
        return statList

    def _get_radec(self, measDict):
        # adapted from processTdsGodot.ipynb
        epList = [ep for ep in measDict['epoch']]
        raMeasList = [float(ra) * num.Pi / 180 for ra in measDict['angle_1']]
        decMeasList = [float(dec) * num.Pi / 180 for dec in measDict['angle_2']]
        return [epList, raMeasList, decMeasList]

    def _get_monostatic(self, measDict):
        # adapted from processTdsGodot.ipynb
        epList = [ep for ep in measDict['epoch']]
        r1MeasList = [float(r1) for r1 in measDict['range']]
        azMeasList = [float(az) * num.Pi / 180 for az in measDict['angle_1']]
        elMeasList = [float(el) * num.Pi / 180 for el in measDict['angle_2']]
        rr1MeasList = [float(rr1) for rr1 in measDict['doppler_instantaneous']]
        return [epList, r1MeasList, azMeasList, elMeasList, rr1MeasList]

    def _get_bistatic(self, measDict):
        # adapted from processTdsGodot.ipynb
        epList = [ep for ep in measDict['epoch']]
        r2MeasList = [float(r2) for r2 in measDict['range']]
        azMeasList = [float(az) * num.Pi / 180 for az in measDict['angle_1']]
        elMeasList = [float(el) * num.Pi / 180 for el in measDict['angle_2']]
        rr2MeasList = [float(rr2) for rr2 in measDict['doppler_instantaneous']]
        return [epList, r2MeasList, azMeasList, elMeasList, rr2MeasList]

    def _get_tdm_data(self):
        if self.measType == 'RADEC':
            return self._get_radec(self.measDict)
        elif self.measType == 'MONOSTATIC':
            return self._get_monostatic(self.measDict)
        elif self.measType == 'BISTATIC':
            return self._get_bistatic(self.measDict)

    def _get_time(self, entry):
        # measDict already contains the date in datetime format (seems to be in TDB scale)
        return entry[0]

    def _get_epGodot(self, ep):
        # converting datetime to GODOT internal time format + taking into account the scale
        return tempo.Epoch(ep.strftime(self.CCSDS_DATETIME_FORMAT) + ' ' + self.timeScaleMeas)

    @staticmethod
    def _to_standard_angle(angle):
        # Lyu's fix of angles to be aligned with Stone Soup frames

        angle = np.pi / 2 - angle

        # Ensure the angle is within the range [0, 360)
        if angle >= 2 * np.pi:
            angle -= 2 * np.pi
        elif angle < 0:
            angle += 2 * np.pi

        return angle

    @staticmethod
    def _get_rotation_offset(station_location):
        # https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates
        x, y, z = station_location
        r = np.linalg.norm(station_location)
        lon = np.arcsin(z / r)  # also lambda
        lat = np.arctan2(y, x)
        roll, pitch, yaw = (0.5 * np.pi - lon, 0, 0.5 * np.pi + lat)
        return [roll, pitch, yaw]

    def _get_offsets(self, time):
        epGodot = self._get_epGodot(time)  # # it is in GODOT time format to query station location

        sv_station = self.station_vector_object.eval(epGodot)
        station_location = [sv_station[0] * self.m_in_km,
                            sv_station[1] * self.m_in_km,
                            sv_station[2] * self.m_in_km]
        rotation_offset = self._get_rotation_offset(station_location)
        station_velocity = [sv_station[3] * self.m_in_km,
                            sv_station[4] * self.m_in_km,
                            sv_station[5] * self.m_in_km]
        offsets = {
            'translation_offset': station_location,
            'rotation_offset': rotation_offset,
            'velocity': station_velocity
        }
        return offsets

    def _radec(self, entry, offsets):
        # unchecked telescope likelihood
        # TODO: check how to set up the offsets as RaDec is supposedly measured from the centre of Earth
        # TODO: redefine function() in CartesianToElevationBearing
        # TODO: https://github.com/dstl/Stone-Soup/blob/0a9ebe2584eb6cd3d77756fd9dfa7b98649799fd/stonesoup/models/measurement/nonlinear.py#L427
        # TODO: using tools in obsim module as distrubuted by GMV
        # TODO: this can ultimately be tested by comparing dataset measurements to those generated from the ground truth

        measurement_model = CartesianToElevationBearing(
            mapping=self.mapping,
            ndim_state=self.ndim_state,
            translation_offset=[0, 0, 0],
            rotation_offset=[0, 0, 0],
            noise_covar=self.noise_covar
        )

        azMeas = self._to_standard_angle(entry[2])
        elMeas = self._to_standard_angle(entry[3])

        state_vector = StateVector([elMeas, azMeas])

        detection = Detection(
            state_vector=state_vector,
            timestamp=self._get_time(entry),
            measurement_model=measurement_model,
            metadata={})
        return detection

    def _monostatic(self, entry, offsets):
        # TODO: redefine function() in CartesianToElevationBearingRangeRate using obSim module in GODOT

        measurement_model = CartesianToElevationBearingRangeRate(
            mapping=self.mapping,
            ndim_state=self.ndim_state,
            translation_offset=offsets['translation_offset'],
            rotation_offset=offsets['rotation_offset'],
            velocity=offsets['velocity'],
            velocity_mapping=self.velocity_mapping,
            noise_covar=self.noise_covar
        )

        r1Meas = entry[1] * self.m_in_km
        azMeas = self._to_standard_angle(entry[2])
        elMeas = self._to_standard_angle(entry[3])
        rr1Meas = entry[4] * self.m_in_km
        state_vector = StateVector([elMeas, azMeas, r1Meas, rr1Meas])

        detection = Detection(
            state_vector=state_vector,
            timestamp=self._get_time(entry),
            measurement_model=measurement_model,
            metadata={})
        return detection

    def _bistatic(self, entry, stat):
        # TODO: measurement model is not yet available in Stone Soup
        # TODO: CartesianToElevationBearingRangeRate cannot be directly applied due to 2-way measurements
        pass

    def _wrap_detection(self, entry):
        offsets = self._get_offsets(self._get_time(entry))  # this is common to all sensors

        if self.measType == 'RADEC':
            return self._radec(entry, offsets)
        elif self.measType == 'MONOSTATIC':
            return self._monostatic(entry, offsets)
        elif self.measType == 'BISTATIC':
            return self._bistatic(entry, offsets)

    def detections_gen(self):

        # we chose to return detections as sets in case the feeder will get measurements with the same timestamp
        detections = set()
        previous_time = None

        for entry in zip(*self.tdm_data):

            time = self._get_time(entry)  # it is in datetime here

            if previous_time is not None and previous_time != time:
                yield previous_time, detections
                detections = set()
            previous_time = time

            detection = self._wrap_detection(entry)
            detections.add(detection)

        # Yield remaining
        yield previous_time, detections

class CustomDetectionReader(DetectionReader, _CSVReader):
    """A detection reader for pre-processed csv files of ground-based detections.

    Parameters
    ----------
    """
    filters: dict = Property(default=None, doc='Entry')
    max_datapoints: int = Property(default=None, doc='N datapoints to consider')
    measurement_model: MeasurementModel = Property(default=None, doc='Entry')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_fields = ['TRUE_X', 'TRUE_VX', 'TRUE_Y', 'TRUE_VY', 'TRUE_Z', 'TRUE_VZ']
        self.station_fields = ['XSTAT_X', 'XSTAT_VX', 'XSTAT_Y', 'XSTAT_VY', 'XSTAT_Z', 'XSTAT_VZ']
        self.km_meas_fields = ['RANGE', 'DOPPLER_INSTANTANEOUS']
        self.km_fields = self.target_fields + self.station_fields + self.km_meas_fields
        self.m_in_km = 1000
        self.bearing_name = 'ANGLE_1'
        self.elevation_name = 'ANGLE_2'

    @staticmethod
    def _to_standard_angle(angle):
        angle = np.pi / 2 - angle

        # Ensure the angle is within the range [0, 360)
        if angle >= 2 * np.pi:
            angle -= 2 * np.pi
        elif angle < 0:
            angle += 2 * np.pi

        return angle

    @staticmethod
    def _get_translation_offset(station_ecef, mapping):
        return station_ecef[mapping, :]

    @staticmethod
    def _get_velocity(station_ecef, velocity_mapping):
        return station_ecef[velocity_mapping, :]

    @staticmethod
    def _get_rotation_offset(station_ecef, mapping):
        # https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates
        station_location = station_ecef[mapping, :]
        x, y, z = station_location
        r = np.linalg.norm(station_location)
        lon = np.arcsin(z / r)  # also lambda
        lat = np.arctan2(y, x)
        roll, pitch, yaw = (0.5 * np.pi - lon, 0, 0.5 * np.pi + lat)
        return StateVector([roll, pitch, yaw])

    def _get_measurement_adjusted(self, row):
        state_vector = []
        for col_name in self.state_vector_fields:
            value = float(row[col_name])  # csv containst strings
            value = self.m_in_km * value if col_name in self.km_fields else value  # turn into m if in km
            value = Bearing(self._to_standard_angle(value)) if col_name == self.bearing_name else value
            value = Elevation(value) if col_name == self.elevation_name else value
            state_vector.append([value])
        return StateVector(state_vector)

    def _get_measurement_model(self, row):
        measurement_model = copy.deepcopy(self.measurement_model)
        station_ecef = StateVector([self.m_in_km * float(row[key]) for key in self.station_fields])
        measurement_model.translation_offset = self._get_translation_offset(station_ecef, measurement_model.mapping)
        measurement_model.rotation_offset = self._get_rotation_offset(station_ecef, measurement_model.mapping)
        measurement_model.velocity = self._get_velocity(station_ecef, measurement_model.velocity_mapping)
        return measurement_model

    def _get_ss_measurement(self, row, measurement_model):
        target_state = []
        for col_name in self.target_fields:
            coef = self.m_in_km if col_name in self.km_fields else 1
            value = coef * float(row[col_name])
            if col_name == self.bearing_name:
                value = Bearing(self._to_standard_angle(value))
            if col_name == self.elevation_name:
                value = Elevation(value)
            target_state.append([value])

        return measurement_model.function(State(state_vector=StateVector(target_state)))


    @BufferedGenerator.generator_method
    def detections_gen(self, from_ground_truth=False):
        n_datapoints = 0
        with self.path.open(encoding=self.encoding, newline='') as csv_file:

            for row in csv.DictReader(csv_file, **self.csv_options):

                n_datapoints += 1

                if self.max_datapoints is not None:
                    if n_datapoints > self.max_datapoints:
                        break

                measurement_model = self._get_measurement_model(row)
                if from_ground_truth:
                    measurement = self._get_ss_measurement(row, measurement_model)
                else:
                    measurement = self._get_measurement_adjusted(row)

                detection = Detection(state_vector=measurement,
                                      timestamp=self._get_time(row),
                                      measurement_model=self._get_measurement_model(row),
                                      metadata=self._get_metadata(row))

                yield detection
