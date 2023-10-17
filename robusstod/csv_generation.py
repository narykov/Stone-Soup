import os
import pandas as pd
from datetime import datetime

# global variables
root_dir = '/Users/alexeynarykov/PycharmProjects/Stone-Soup/robusstod/src/ROBUSSTOD_TDS/RODDAS_OD_00_015/tdm/'


def get_filenames(abs_dir):
    file_set = []
    for dir_, _, files in os.walk(abs_dir):
        for file_name in files:
            rel_dir = os.path.relpath(dir_, abs_dir)
            rel_file = os.path.join(rel_dir, file_name)
            file_set.append(rel_file)

    return sorted(file_set)


def get_data_fields(infile):
    data_fields = set()
    data_start_flag = False
    for line_full in infile:
        line = line_full.rstrip()

        if line == 'DATA_STOP':
            break

        if data_start_flag is not True:
            data_start_flag = True if line == 'DATA_START' else False
            continue

        data_fields.add(line.split(' = ')[0])

    infile.seek(0)

    return sorted(list(data_fields))

def get_data_frame(infile, metadata):
    # collect the data field names
    skipping_entries = ['META_START', 'META_STOP', 'DATA_START', 'DATA_STOP']
    data_fields = get_data_fields(infile)
    df = pd.DataFrame(columns=['TIME', 'STATION', *data_fields])
    previous_time = None
    reading = dict()
    data_start_flag = False

    for line_full in infile:
        line = line_full.strip()
        if line in skipping_entries:
            data_start_flag = True if line == 'DATA_START' else False
            continue

        key, value = line.split(' = ')
        # if we are still collecting metadata
        if not data_start_flag:
            metadata[key] = value.strip()
            continue

        # if we are actually dealing with data
        time, entry_value = value.split()
        if previous_time is not None and previous_time != time:
            time_clean = datetime.strptime(time, '%Y-%m-%dT%H:%M:%S.%fZ')
            reading['TIME'] = time_clean
            reading['STATION'] = metadata['PARTICIPANT_1']
            df_the_dict = pd.DataFrame.from_dict({'data': reading.values()}, orient='index',
                                                 columns=reading.keys())
            df = pd.concat([df, df_the_dict], ignore_index=True)
            reading = dict()
        previous_time = time

        reading[key] = float(entry_value)

    return df

def main():
    onlyfiles = get_filenames(root_dir)
    dfs = pd.DataFrame()

    for filename in onlyfiles:
        path = os.path.normpath(os.path.join(root_dir, filename))
        metadata = {'filename': filename}

        with open(path, 'r', encoding='utf-8') as infile:
            df = get_data_frame(infile, metadata)

        dfs = pd.concat([dfs, df], ignore_index=True)

    dfs = dfs.sort_values(by=['TIME']).copy()
    filename = ''.join([root_dir.split('/')[-3], '.csv'])
    dfs.to_csv('src/csv/'+filename, index=False, index_label='TIME', mode='w+')

if __name__ == "__main__":
    main()
    