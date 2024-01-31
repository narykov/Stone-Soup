import numpy as np
from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRangeRate
from stonesoup.types.array import CovarianceMatrix
import matplotlib.pyplot as plt

# ROBUSSTOD SPECIFIC
from stonesoup.robusstod.stonesoup.reader import CustomDetectionReader


def main():

    ndim_state = 6
    mapping_location = (0, 2, 4)  # encodes location indices in the state vector
    mapping_velocity = (1, 3, 5)  # encodes velocity indices in the state vector


    # Specify sensor parameters and generate a history of measurements for the time steps (parameters for RR01)
    sigma_r = 20  # Range: 20.0 m
    sigma_a = np.deg2rad(400*0.001)  # Azimuth - elevation: 400.0 mdeg
    sigma_rr = 650.0 * 0.001  # Range-rate: 650.0 mm/s
    sigma_el, sigma_b, sigma_range, sigma_range_rate = sigma_a, sigma_a, sigma_r, sigma_rr
    noise_covar = CovarianceMatrix(np.diag([sigma_el ** 2, sigma_b ** 2, sigma_range ** 2, sigma_range_rate ** 2]))
    sensor_parameters = {
        'ndim_state': ndim_state,
        'mapping': mapping_location,
        'noise_covar': noise_covar
    }

    measurement_model = CartesianToElevationBearingRangeRate(
        ndim_state=sensor_parameters['ndim_state'],
        mapping=sensor_parameters['mapping'],
        velocity_mapping=mapping_velocity,
        noise_covar=sensor_parameters['noise_covar']
    )
    path = "measurements.csv"
    measurement_fields = ("ANGLE_2", "ANGLE_1", "RANGE", "DOPPLER_INSTANTANEOUS")

    detector = CustomDetectionReader(
        path=path,
        state_vector_fields=measurement_fields,
        time_field="TIME",
        measurement_model=measurement_model
    )

    measurements = []
    for measurement in detector.detections_gen():
        measurements.append(measurement)
    # TODO: not sure how this works, i.e., that the generator does not need to be restarted
    measurements_ss = []
    for measurement in detector.detections_gen(from_ground_truth=True):
        measurements_ss.append(measurement)

    fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1)
    ax0.set_ylabel('El, [rad]')
    ax0.get_xaxis().set_ticks([])
    ax1.set_ylabel('Az, [rad]')
    ax1.get_xaxis().set_ticks([])
    ax2.set_ylabel('R, [m]')
    ax2.get_xaxis().set_ticks([])
    ax3.set_ylabel('RR, [m/s]')
    ax3.set_xlabel('Time')

    for label in ['dataset values', 'Stone Soup model']:
        if label == 'dataset values':
            data = measurements
            plot_param = {'color': 'b', 'marker': 'x'}
        else:
            data = measurements_ss
            plot_param = {'color': 'r', 'marker': '.'}

        timestamps = [measurement.timestamp for measurement in data]
        ax0.plot(timestamps, [measurement.state_vector[0] for measurement in data], label=label, **plot_param)
        ax1.plot(timestamps, [measurement.state_vector[1] for measurement in data], label=label, **plot_param)
        ax2.plot(timestamps, [measurement.state_vector[2] for measurement in data], label=label, **plot_param)
        ax3.plot(timestamps, [measurement.state_vector[3] for measurement in data], label=label, **plot_param)

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
