import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
from stonesoup.types.array import StateVector
from stonesoup.robusstod import enu2ecef, ecef2enu, ecef2aer, aer2ecef
from stonesoup.types.angle import Bearing, Elevation
from datetime import datetime
import datetime as dt

participants = ['00039451', '00042063', '00055165']
participant = participants[2]
filename = 'src/ROBUSSTOD_TDS/RODDAS_OD_00_015/oem_sim/' + participant + '.oem'
# filename = 'src/ROBUSSTOD_TDS/RODDAS_OD_00_015/oem_sim/00042063.oem'
# filename = 'src/ROBUSSTOD_TDS/RODDAS_OD_00_015/oem_sim/00055165.oem'


df = pd.DataFrame()

with open(filename) as file:
    meta_ended = False
    previous_line = None
    for line in file:
        if not meta_ended:
            if previous_line == 'META_STOP':
                meta_ended = True
            else:
                previous_line = line.rstrip()
                continue
        else:
            keys = ['TIME', 'X', 'Y', 'Z', 'XDOT', 'YDOT', 'ZDOT']
            time_format = '%Y-%m-%dT%H:%M:%S.%f'
            reading = dict(zip(keys, line.split()))
            time_format = time_format + 'Z' if reading['TIME'][-1] == 'Z' else time_format
            for key in keys[1:]:
                reading[key] = float(reading[key])
            reading['TIME'] = datetime.strptime(reading['TIME'], time_format)  # UTC
            reading['TIME_TAI'] = reading['TIME'] + dt.timedelta(seconds=37)  # to TAI
            reading['OBJECT'] = participant
            df_the_dict = pd.DataFrame.from_dict({'values': reading.values()}, orient='index',
                                                 columns=reading.keys())
            df = pd.concat([df, df_the_dict], ignore_index=True)

df.to_csv('src/csv/oem_' + participant + '.csv', index=False, index_label='TIME', mode='w+')
# df.to_csv('src/csv/oem_00039451.csv', index=False, index_label='TIME', mode='w+')

datafile = 'src/csv/RR01_data_alex_good.csv'
df = pd.read_csv(datafile, parse_dates=['TIME'])
stations = sorted(list(set(df['STATION'])))

to_metres = 1000
target_locations = []
station_locations = []

fig = plt.figure()
ax = fig.add_subplot(projection='3d', computed_zorder=False)
centre = StateVector([0, 0, 0])
plt.plot(*centre, marker='d')

radius = 6371*to_metres
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = radius * np.cos(u)*np.sin(v)
y = radius * np.sin(u)*np.sin(v)
z = radius * np.cos(v)
# ax.plot_wireframe(x, y, z, color="r")
ax.plot_surface(x, y, z)



for index, row in df.iterrows():
    station_dict = {
        'XSTAT_X': row['XSTAT_X'] * to_metres,
        'XSTAT_Y': row['XSTAT_Y'] * to_metres,
        'XSTAT_Z': row['XSTAT_Z'] * to_metres,
    }
    station_ecef = StateVector([*station_dict.values()])
    station_locations.append(station_ecef)

    meas_dict = {
        'Azimuth': Bearing(row['ANGLE_1']),
        'Elevation': Elevation(row['ANGLE_2']),
        'Range': row['RANGE'] * to_metres
    }
    target_aer = StateVector([*meas_dict.values()])
    target_ecef = aer2ecef(target_aer, station_ecef)
    target_locations.append(target_ecef)

    plt.plot(*station_ecef, marker='x', color='r')
    plt.plot(*target_ecef, marker='.', color='g')
    plt.pause(1)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
centre = StateVector([0, 0, 0])
plt.plot(*centre, marker='d')

for station, target in zip(station_locations, target_locations):
    plt.plot(*station, marker='.', color='r')
    plt.plot(*target, marker='x', color='g')

df['RANGE'] = df['RANGE'] * 1000
df['ANGLE_1_GCRF'] = np.rad2deg(df['ANGLE_1_GCRF'])
df['ANGLE_2_GCRF'] = np.rad2deg(df['ANGLE_2_GCRF'])
df['ANGLE_1'] = np.rad2deg(df['ANGLE_1'])
df['ANGLE_2'] = np.rad2deg(df['ANGLE_2'])
fig0, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)

marker_standard = 'o'
marker_gcrf = 'x'

df.plot.scatter(x='TIME', y='RANGE', marker=marker_standard, color='r', ax=ax0, label='RANGE')
# df.plot.scatter(x='TIME', y='RANGE_GCRF', marker=marker_gcrf, color='g', ax=ax0, label='RANGE_GCRF')

df.plot.scatter(x='TIME', y='ANGLE_1', marker=marker_standard, color='r', ax=ax1, label='Azimuth')
# df.plot.scatter(x='TIME', y='ANGLE_1_GCRF', marker=marker_gcrf, color='g', ax=ax1, label='ANGLE_1_GCRF')

df.plot.scatter(x='TIME', y='ANGLE_2', marker=marker_standard, color='r', ax=ax2, label='Elevation')
# df.plot.scatter(x='TIME', y='ANGLE_2_GCRF', marker=marker_gcrf, color='b', ax=ax2, label='ANGLE_2_GCRF')


# df.plot.scatter(x='TIME', y='ANGLE_1_GCRF', marker='x', color='b', ax=ax, label='ANGLE_1_GCRF')
# df.plot.scatter(x='TIME', y='ANGLE_2_GCRF', marker='+', color='g', ax=ax_0, label='ANGLE_2_GCRF')

fig_1, ax_1 = plt.subplots()
color = cm.rainbow(np.linspace(0, 1, len(stations)))
df = df.sort_values(by=['TIME']).copy()
for i, station in enumerate(stations):
    # df.loc[df['STATION'] == station]['RANGE'] = df.loc[df['STATION'] == station]['RANGE'] * 1000
    df.loc[df['STATION'] == station].plot.scatter(x='TIME', y='RANGE_GODOT', marker='o', ax=ax_1, color='g', label='GODOT')
    df.loc[df['STATION'] == station].plot.scatter(x='TIME', y='RANGE_GCRF', marker='.', ax=ax_1, color='r', label='DATASET')
plt.xticks(rotation=45, ha='right')
ylimits = plt.gca().get_ylim()
plt.ylim((0, 2000))


fig_2, ax_2 = plt.subplots()
i = 1  # station to visualise
df.loc[df['STATION'] == stations[i]].plot.scatter(
    x='TIME', y='RANGE', marker='.', ax=ax_2, color=color[i], label=stations[i]
)
# plt.ylim(ylimits)
plt.ylim((0, 2000))
plt.xticks(rotation=45, ha='right')
