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


def get_data_frame(infile):
    # collect the data field names
    metadata = {}
    skipping_entries = ['META_START', 'META_STOP', 'DATA_START', 'DATA_STOP']
    df = pd.DataFrame()
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
            metadata[key] = value
            continue

        # if we are actually dealing with data
        time, entry_value = value.split()
        if previous_time is not None and previous_time != time:
            reading['TIME'] = datetime.strptime(time, '%Y-%m-%dT%H:%M:%S.%fZ')
            reading['STATION'] = metadata['PARTICIPANT_1']
            reading['TARGET_ID'] = metadata['PARTICIPANT_2']
            df_the_dict = pd.DataFrame.from_dict({'values': reading.values()}, orient='index',
                                                 columns=reading.keys())
            df = pd.concat([df, df_the_dict], ignore_index=True)
            reading = dict()
        previous_time = time
        reading[key] = float(entry_value)

    return df, metadata

def main():
    onlyfiles = get_filenames(root_dir)
    dfs = pd.DataFrame()

    for filename in onlyfiles:
        path = os.path.normpath(os.path.join(root_dir, filename))

        with open(path, 'r', encoding='utf-8') as infile:
            df, metadata = get_data_frame(infile)

        dfs = pd.concat([dfs, df], ignore_index=True)

    cols = df.columns.values
    trailing = ['TIME', 'TARGET_ID', 'STATION']
    following = sorted(list(set(cols) - set(trailing)))
    new_cols = trailing + following
    dfs = dfs[new_cols].sort_values(by=['TIME'])
    filename = ''.join([root_dir.split('/')[-3], '.csv'])
    dfs.to_csv('src/csv/'+filename, index=False, index_label='TIME', mode='w+')

if __name__ == "__main__":
    main()
