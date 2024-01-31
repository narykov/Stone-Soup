import pandas as pd
from stonesoup.types.array import StateVector
from matplotlib import pyplot as plt

# settings
participants = ['00039451', '00042063', '00055165']
participant = participants[1]
mapping = [0, 2, 4]
to_metres = 1000

filename = '/Users/alexeynarykov/PycharmProjects/Stone-Soup/robusstod/src/csv/oem_' + participant + '.csv'
time_name = 'TIME'

def main():
    df = pd.read_csv(filename, parse_dates=[time_name])
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for index, row in df.iterrows():
        target_dict = {
            'TRUE_X': row['X'] * to_metres,
            'TRUE_VX': row['XDOT'] * to_metres,
            'TRUE_Y': row['Y'] * to_metres,
            'TRUE_VY': row['YDOT'] * to_metres,
            'TRUE_Z': row['Z'] * to_metres,
            'TRUE_VZ': row['ZDOT'] * to_metres
        }
        target_true = StateVector([target_dict[key] for key in target_dict])
        ax.plot(*target_true[mapping, :], marker='.', color='g')
        plt.pause(0.1)


if __name__ == "__main__":
    main()
