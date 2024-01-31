""" A brief example of the application of a single-TDM file reader. """
from stonesoup.robusstod.stonesoup.reader import TDMReader


def main():
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

    reader = TDMReader(
        path='RODDAS_OD_00_015/tdm/00001_RR01_20230817T100415_20230817T100511.tdm',
        osdm_folder_path='RODDAS_OD_00_015/osdm',
        stationMeasDict=stationMeasDict,
        data_folder_path='robusstod/misc/reader_gmv/data',
        universe_path='universe.yml'
    )

    detections_list = []
    for detections in reader.detections_gen():
        detections_list.append(detections)

if __name__ == '__main__':
    main()
