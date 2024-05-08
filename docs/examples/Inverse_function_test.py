import numpy as np
from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRangeRate
from stonesoup.types.state import State, StateVector
from stonesoup.types.detection import Detection

mapping = (0, 2, 4)
velocity_mapping = (1, 3, 5)

target_state = State(state_vector=[10, 10, 10, 0, 10, 0])
sensor_state = State(state_vector=[0, 10, 0, 0, 0, 0])
# uncomment below for static sensor (note that resulting Doppler doesn't match as good as other measurement elements)
# sensor_state = State(state_vector=[0, 0, 0, 0, 0, 0])

# Measurement model
sensor_velocity = sensor_state.state_vector[velocity_mapping, :]
measurement_model = CartesianToElevationBearingRangeRate(
    ndim_state=6,
    mapping=mapping,
    noise_covar=np.array(np.diag([np.deg2rad(1)**2, np.deg2rad(1)**2, 10**2, 10**2])),
    velocity_mapping=velocity_mapping,
    velocity=sensor_velocity
)

measurement_vector = measurement_model.function(target_state, noise=False)
measurement = Detection(state_vector=measurement_vector, measurement_model=measurement_model)
target_state_recovered = State(state_vector=measurement.measurement_model.inverse_function(measurement))
measurement_vector_new = measurement_model.function(target_state_recovered, noise=False)

print(f'Measurement from the original state: \n {measurement_vector}')
print(f'Measurement from the recovered state: \n {measurement_vector_new}')
