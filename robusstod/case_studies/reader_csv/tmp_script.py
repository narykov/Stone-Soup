""" A file used to generate csv files from the datasets"""

import scipy
from pathlib import Path
import os
from godot.core import util, tempo, num, astro, autodif as ad, ipfwrap as ipf
from godot import cosmos
import numpy as np
import datetime as dt
import pyproj
import json

from godot.core import  constants, num, autodif as ad
import godot.core.astro as astro
import godot.model.common as common
import pandas as pd
from datetime import datetime

# ROBUSSTOD SPECIFIC
from stonesoup.robusstod.python_libs.parsers import CCtd_Tdm as cctd
from stonesoup.robusstod.python_libs.parsers  import CCoe_Oem as ccoe
from stonesoup.robusstod.python_libs.obsim import obsim as obsim
from stonesoup.robusstod.python_libs.utils import Config


stationMeasDict = {
    'TR01': 'RADEC',
    'TR02': 'RADEC',
    'TR03': 'RADEC',
    'TR04': 'RADEC',
    'RR01': 'MONOSTATIC',
    'RR02': 'BISTATIC',
    'RR03': 'BISTATIC',
    'RR04': 'BISTATIC'
}  # the dictionary from the original script

stationMeasDictAdditional = {
    'MSRS': 'MONOSTATIC',
    'PFIS': 'MONOSTATIC',
    'TR01_highnoise': 'RADEC',
    'TR02_highnoise': 'RADEC',
    'TR06': 'RADEC',
    'TR07': 'RADEC',
    'RR05': 'MONOSTATIC',
    'RR06': 'MONOSTATIC',
    'RR07': 'MONOSTATIC',
    'RAN': 'RADEC',
    'SPR': 'RADEC',
    'RR03_highnoise': 'MONOSTATIC',
    'RR04_highnoise': 'MONOSTATIC'
}  # extracted by hand from ROBUSSTOD_TDS_DELIVERY_27OCT2023 subfolders
stationMeasDict.update(stationMeasDictAdditional)

CCSDS_DATETIME_FORMAT = '%Y-%m-%dT%H:%M:%S.%f'
EQUATORIAL_EARTH_RADIUS = 6378.13655
FLATTENING_PARAMETER = 1.0/298.25769

util.suppressLogger()

class AlexeyStationVector(common.VectorTimeEvaluable):
    """
    Station Info; added by Alexey
    """

    def __init__(self, uni, station):
        common.VectorTimeEvaluable.__init__(self)
        self.__uni = uni
        self.__station_point = self.__uni.frames.pointId(station)
        self.__lt_center_id = self.__uni.frames.pointId("Earth")
        self.__lt_axes_id = self.__uni.frames.axesId("GCRF")

    def eval(self, epoch):
        xstat = self.__uni.frames.vector6(self.__lt_center_id, self.__station_point, self.__lt_axes_id, epoch)
        return xstat

def rad2millideg(ang):
    return ang*180/num.Pi*1000

def deg2rad(ang):
    return ang*num.Pi/180

def main():
    dataset_package_name = 'ROBUSSTOD_TDS_DELIVERY_27OCT2023'
    version = 'full_simulation'  # 'zero_noise' may be more appropriate for debugging
    path = Path('src') / dataset_package_name / version

    tdsNames = next(os.walk(path))[1]
    for tdsName in tdsNames[8:]:
        if tdsName in ['RODDAS_OD_20_005', 'RODDAS_OD_30_010']:
            # skipping the tdsName that doesn't fit the pattern
            continue
        roddas_path = path / tdsName
        # check what ground truths available as oems
        ground_truth_paths = {}
        oemFolder = roddas_path / 'oem_sim'
        isGroundTruth = os.path.isdir(oemFolder)
        if isGroundTruth:
            # if folder with ground truths exists, list the object names
            object_names = [i.split('.')[0] for i in os.listdir(roddas_path / 'oem_sim')]

        # Get station coordinates from OSDM to generate the GroundStationDatabase.json file
        # Read OSDM for involved stations

        statList = {}
        osdmFolder = roddas_path / 'osdm'
        osdmList = [os.path.join(osdmFolder, file) for file in os.listdir(osdmFolder)]

        for osdmFile in osdmList:
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
                x = float(osdm['X']) * 1000
                y = float(osdm['Y']) * 1000
                z = float(osdm['Z']) * 1000
                lon1, lat1, alt1 = transformer.transform(x, y, z, radians=True)
                # tested on RR01 data and it closesly matches, though not exactly, e.g. alt 84 vs 83.57 m

                statList[osdm['SITE_NAME']]['GEODETIC'] = [lat1, lon1, alt1]
            else:
                statList[osdm['SITE_NAME']]['GEODETIC'] = [deg2rad(float(osdm['LAT'])), deg2rad(float(osdm['LON'])),
                                                           float(osdm['ALT']) / 1000]
        stationDataframes = {}
        for station_name in list(statList.keys()):
            stationDataframes[station_name] = pd.DataFrame()

        to_json = {}
        for stat, coordsDict in statList.items():
            coordsDict['ITRF'] = astro.cartesianFromGeodetic(coordsDict['GEODETIC'], EQUATORIAL_EARTH_RADIUS,
                                                             FLATTENING_PARAMETER).tolist()
            # entries below "coordinates" are simply copied from the original json file
            to_json[stat] = {
                "stationAlias": [stat],
                "coordinates": [str(coordsDict["ITRF"][0]) + ' km',
                                str(coordsDict["ITRF"][1]) + ' km',
                                str(coordsDict["ITRF"][2]) + ' km'],
                "plateMotion": ["0.0 km/year", "0.0 km/year", "0.0 km/year"],
                "refEpoch": "2005-01-01T00:00:00.000 TDB",
                "refCenter": "Earth",
                "refAxis": "ITRF",
                "mountingType": "AZ/EL",
                "heightAboveAntennaFoot": "0.0 m",
                "dryOffset": "0.0 m"
            }
            print(f'    "{stat}":')
            print('     {')
            print(f'        "stationAlias"                 :   ["{stat}"],')
            print(
                f'        "coordinates"                  :   ["{coordsDict["ITRF"][0]} km","{coordsDict["ITRF"][1]} km","{coordsDict["ITRF"][2]} km"],')


        filename = 'GroundStationsDatabase.json'
        filepath = Path('data/database')
        with open(filepath / filename, 'w', encoding='utf-8') as fp:
            json.dump(to_json, fp, ensure_ascii=False, indent=4)

        uni_cfg = cosmos.util.load_yaml("universe.yml")
        print(uni_cfg)
        uni = cosmos.Universe(uni_cfg)

        # List TDMs
        tdmFolder = roddas_path / 'tdm'
        tdmList = [os.path.join(tdmFolder, file) for file in os.listdir(tdmFolder)]
        tdmList.sort()

        # Process TDMs
        for tdmfile in tdmList:
            print(f'Processing this TDM: {tdmfile}')
            tdmFile = cctd.TdmFile(tdmfile)
            stat = tdmFile.tdms[0].participant_1
            participant_2 = tdmFile.tdms[0].participant_2
            objectName = participant_2

            if isGroundTruth:
                oemfile = os.path.join(oemFolder, objectName + '.oem')

                # Convert oem (remove trailing Z in epochs, change timescale to TDB)
                oemF = ccoe.OemFile(oemfile)
                oem = oemF.oems[0]
                originalTimeScale = oem.time_system
                if originalTimeScale != 'TDB':
                    oem.time_system = 'TDB'
                    oldEpochs = oem.epoch
                    svs = oem.state_vector
                    newEpochs = []

                    for ep in oldEpochs:
                        oemEpoch = tempo.Epoch(ep.strftime(CCSDS_DATETIME_FORMAT) + ' ' + originalTimeScale)
                        newEpochs.append(
                            dt.datetime.strptime(oemEpoch.calStr('TDB').split(' ')[0], CCSDS_DATETIME_FORMAT))

                    oem.body.segment.list[0].data.state_vector = ccoe.StateVectors()
                    oem.append_state_vector(newEpochs, svs)

                    start_time = tempo.Epoch(oem.start_time.strftime(CCSDS_DATETIME_FORMAT) + ' ' + originalTimeScale)
                    oem.start_time = dt.datetime.strptime(start_time.calStr('TDB').split(' ')[0], CCSDS_DATETIME_FORMAT)
                    stop_time = tempo.Epoch(oem.stop_time.strftime(CCSDS_DATETIME_FORMAT) + ' ' + originalTimeScale)
                    oem.stop_time = dt.datetime.strptime(stop_time.calStr('TDB').split(' ')[0], CCSDS_DATETIME_FORMAT)

                    oemF.write()

                # Get frames
                fr = uni.frames
                fr.addOrbitDataPoint(objectName, oemfile)


            # if participant_2 != objectName:
            #     continue
            if stat not in stationMeasDict.keys():
                print('Station not found in database')
                continue
            # print(f"Analysing TDMfile {tdmfile}")
            # print(f"Station involved - {stat}")

            # Gather measurements based on station type
            timeScaleMeas = tdmFile.tdms[0].time_system
            measType = stationMeasDict[stat]
            measDict = tdmFile.measurements_to_dict()

            station_vector_object = AlexeyStationVector(uni, stat)  # Alexey
            if isGroundTruth:
                object_vector_object = AlexeyStationVector(uni, objectName)  # Lyu

            if measType == 'RADEC':
                raMeasList = [float(ra) * num.Pi / 180 for ra in measDict['angle_1']]
                decMeasList = [float(dec) * num.Pi / 180 for dec in measDict['angle_2']]

                if isGroundTruth:
                    telRaDec = obsim.RaDec(uni, stat, objectName)

                for ep, raMeas, decMeas in zip(measDict['epoch'], raMeasList, decMeasList):
                    epGodot = tempo.Epoch(ep.strftime(CCSDS_DATETIME_FORMAT) + ' ' + timeScaleMeas)

                    timestamp, time_system = str(epGodot).split()
                    time_format = '%Y-%m-%dT%H:%M:%S.%f'
                    time_format = time_format + 'Z' if timestamp[-1] == 'Z' else time_format
                    sv_station = station_vector_object.eval(epGodot)

                    reading = {
                        'TIME': datetime.strptime(timestamp, time_format),
                        # 'TIME_GODOT': epGodot,
                        'TIME_SYSTEM': time_system,
                        'TARGET_ID': participant_2,
                        'STATION': stat,
                        'XSTAT_X': sv_station[0],
                        'XSTAT_Y': sv_station[1],
                        'XSTAT_Z': sv_station[2],
                        'XSTAT_VX': sv_station[3],
                        'XSTAT_VY': sv_station[4],
                        'XSTAT_VZ': sv_station[5],
                        'TDM': tdmfile
                    }  # general data

                    reading.update({
                        'ANGLE_1': raMeas,
                        'ANGLE_2': decMeas
                    })  # dataset

                    if isGroundTruth:
                        raGodot, decGodot = telRaDec.eval(epGodot)
                        sv_object = object_vector_object.eval(epGodot)

                        reading.update({
                            'TRUE_X': sv_object[0],
                            'TRUE_Y': sv_object[1],
                            'TRUE_Z': sv_object[2],
                            'TRUE_VX': sv_object[3],
                            'TRUE_VY': sv_object[4],
                            'TRUE_VZ': sv_object[5]
                        })

                        reading.update({
                            'ANGLE_1_GODOT': raGodot,
                            'ANGLE_2_GODOT': decGodot
                        })  # godot

                    df_the_dict = pd.DataFrame.from_dict({'values': reading.values()}, orient='index',
                                                         columns=reading.keys())
                    stationDataframes[stat] = pd.concat(
                        [stationDataframes[stat], df_the_dict], ignore_index=True)

            if measType == 'MONOSTATIC':
                r1MeasList = [float(r1) for r1 in measDict['range']]
                azMeasList = [float(az) * num.Pi / 180 for az in measDict['angle_1']]
                elMeasList = [float(el) * num.Pi / 180 for el in measDict['angle_2']]
                rr1MeasList = [float(rr1) for rr1 in measDict['doppler_instantaneous']]

                if isGroundTruth:
                    range1 = obsim.Range1(uni, stat, objectName)
                    azEl = obsim.AzEl(uni, stat, objectName)
                    rRate1 = obsim.RRate1(uni, stat, objectName)

                for ep, r1Meas, azMeas, elMeas, rr1Meas in zip(measDict['epoch'], r1MeasList, azMeasList, elMeasList,
                                                               rr1MeasList):
                    epGodot = tempo.Epoch(ep.strftime(CCSDS_DATETIME_FORMAT) + ' ' + timeScaleMeas)

                    timestamp, time_system = str(epGodot).split()
                    time_format = '%Y-%m-%dT%H:%M:%S.%f'
                    time_format = time_format + 'Z' if timestamp[-1] == 'Z' else time_format
                    sv_station = station_vector_object.eval(epGodot)

                    reading = {
                        'TIME': datetime.strptime(timestamp, time_format),
                        # 'TIME_GODOT': epGodot,
                        'TIME_SYSTEM': time_system,
                        'TARGET_ID': participant_2,
                        'STATION': stat,
                        'XSTAT_X': sv_station[0],
                        'XSTAT_Y': sv_station[1],
                        'XSTAT_Z': sv_station[2],
                        'XSTAT_VX': sv_station[3],
                        'XSTAT_VY': sv_station[4],
                        'XSTAT_VZ': sv_station[5],
                        'TDM': tdmfile
                    }  # general data

                    reading.update({
                        'ANGLE_1': azMeas,
                        'ANGLE_2': elMeas,
                        'RANGE': r1Meas,
                        'DOPPLER_INSTANTANEOUS': rr1Meas,
                    })  # dataset

                    if isGroundTruth:
                        # generate test measurements with GODOT if the ground truth is available
                        r1Godot = range1.eval(epGodot)
                        azGodot, elGodot = azEl.eval(epGodot)
                        rr1Godot = rRate1.eval(epGodot)
                        sv_object = object_vector_object.eval(epGodot)

                        reading.update({
                            'TRUE_X': sv_object[0],
                            'TRUE_Y': sv_object[1],
                            'TRUE_Z': sv_object[2],
                            'TRUE_VX': sv_object[3],
                            'TRUE_VY': sv_object[4],
                            'TRUE_VZ': sv_object[5]
                        })

                        reading.update({
                            'ANGLE_1_GODOT': azGodot,
                            'ANGLE_2_GODOT': elGodot,
                            'RANGE_GODOT': r1Godot,
                            'DOPPLER_INSTANTANEOUS_GODOT': rr1Godot
                        })  # godot


                    df_the_dict = pd.DataFrame.from_dict({'values': reading.values()}, orient='index',
                                                         columns=reading.keys())
                    stationDataframes[stat] = pd.concat(
                        [stationDataframes[stat], df_the_dict], ignore_index=True)

            if measType == 'BISTATIC':
                r2MeasList = [float(r2) for r2 in measDict['range']]
                azMeasList = [float(az) * num.Pi / 180 for az in measDict['angle_1']]
                elMeasList = [float(el) * num.Pi / 180 for el in measDict['angle_2']]
                rr2MeasList = [float(rr2) for rr2 in measDict['doppler_instantaneous']]

                if isGroundTruth:
                    range2 = obsim.Range2(uni, stat, objectName)
                    azEl = obsim.AzEl(uni, stat, objectName)
                    rRate2 = obsim.RRate2(uni, stat, objectName)

                for ep, r2Meas, azMeas, elMeas, rr2Meas in zip(measDict['epoch'], r2MeasList, azMeasList, elMeasList,
                                                               rr2MeasList):

                    epGodot = tempo.Epoch(ep.strftime(CCSDS_DATETIME_FORMAT) + ' ' + timeScaleMeas)

                    timestamp, time_system = str(epGodot).split()
                    time_format = '%Y-%m-%dT%H:%M:%S.%f'
                    time_format = time_format + 'Z' if timestamp[-1] == 'Z' else time_format
                    sv_station = station_vector_object.eval(epGodot)

                    reading = {
                        'TIME': datetime.strptime(timestamp, time_format),
                        # 'TIME_GODOT': epGodot,
                        'TIME_SYSTEM': time_system,
                        'TARGET_ID': participant_2,
                        'STATION': stat,
                        'XSTAT_X': sv_station[0],
                        'XSTAT_Y': sv_station[1],
                        'XSTAT_Z': sv_station[2],
                        'XSTAT_VX': sv_station[3],
                        'XSTAT_VY': sv_station[4],
                        'XSTAT_VZ': sv_station[5],
                        'TDM': tdmfile
                    }  # general data

                    reading.update({
                        'ANGLE_1': azMeas,
                        'ANGLE_2': elMeas,
                        'RANGE': r2Meas,
                        'DOPPLER_INSTANTANEOUS': rr2Meas,
                    })  # dataset

                    if isGroundTruth:
                        # generate test measurements with GODOT if the ground truth is available
                        r2Godot = range2.eval(epGodot)
                        azGodot, elGodot = azEl.eval(epGodot)
                        rr2Godot = rRate2.eval(epGodot)
                        sv_object = object_vector_object.eval(epGodot)

                        reading.update({
                            'TRUE_X': sv_object[0],
                            'TRUE_Y': sv_object[1],
                            'TRUE_Z': sv_object[2],
                            'TRUE_VX': sv_object[3],
                            'TRUE_VY': sv_object[4],
                            'TRUE_VZ': sv_object[5]
                        })

                        reading.update({
                            'ANGLE_1_GODOT': azGodot,
                            'ANGLE_2_GODOT': elGodot,
                            'RANGE_GODOT': r2Godot,
                            'DOPPLER_INSTANTANEOUS_GODOT': rr2Godot
                        })  # godot

                df_the_dict = pd.DataFrame.from_dict({'values': reading.values()}, orient='index',
                                                     columns=reading.keys())
                stationDataframes[stat] = pd.concat(
                    [stationDataframes[stat], df_the_dict], ignore_index=True)

            else:
                continue

        for stat, station_df in stationDataframes.items():
            cols = station_df.columns.values
            trailing = ['TIME', 'TIME_SYSTEM', 'TARGET_ID', 'STATION']
            following = sorted(list(set(cols) - set(trailing)))
            new_cols = trailing + following
            station_df = station_df[new_cols].sort_values(by=['TIME'])
            csvfilepath = Path('out') / dataset_package_name / version / tdsName
            if not os.path.exists(csvfilepath):
                os.makedirs(csvfilepath)
            filename = stat + '.csv'
            station_df.to_csv(csvfilepath / filename, index_label='TIME', index=False, mode='w+')

if __name__ == '__main__':
    main()
