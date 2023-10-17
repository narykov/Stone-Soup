import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import numpy as np

datafile = 'src/csv/RODDAS_OD_00_015.csv'
df = pd.read_csv(datafile, parse_dates=['TIME'])
stations = sorted(list(set(df['STATION'])))

fig_1, ax_1 = plt.subplots()
color = cm.rainbow(np.linspace(0, 1, len(stations)))
for i, station in enumerate(stations):
    df.loc[df['STATION'] == station].plot.scatter(x='TIME', y='RANGE', marker='.', ax=ax_1, color=color[i], label=station)
plt.xticks(rotation=45, ha='right')
ylimits = plt.gca().get_ylim()


fig_2, ax_2 = plt.subplots()
i = 1
df.head(16).loc[df['STATION'] == stations[i]].plot.scatter(
    x='TIME', y='RANGE', marker='.', ax=ax_2, color=color[i], label=stations[i]
)
plt.ylim(ylimits)
plt.xticks(rotation=45, ha='right')

print()