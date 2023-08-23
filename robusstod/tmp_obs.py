from godot import cosmos
from godot.core import util

# Suppress logger
util.suppressLogger()

# Create universe
uni_config = cosmos.util.load_yaml("config/universe_obs.yml")
uni = cosmos.Universe(uni_config)

# # Create trajectory
# traj_config = cosmos.util.load_yaml("config/trajectory_obs.yml")
# traj = cosmos.Trajectory(uni, traj_config)
#
# # Compute the
# traj.compute(False)

from godot.model import common, obs
# Get reference center and axis for LT computation
icrf = uni.frames.axesId("ICRF")
ssb = uni.frames.pointId("SSB")
# Get bodies
sun_bd = uni.bodies.get("Sun")
earth_bd = uni.bodies.get("Earth")
list_bd = [sun_bd, earth_bd]
# Create Relativistic LT correction
gamma = common.ConstantScalarTimeEvaluable(1.0)
rltc = obs.RelativisticLightTimeCorrection(
    uni.frames, ssb, icrf, list_bd, gamma=gamma)

from godot.core import tempo
# Define one way light time object
owlt = obs.OneWayLightTime(uni.frames, ssb, icrf, rltc)
# Evaluate OWLT
lt_dir = obs.Direction.Backward
p1 = uni.frames.pointId("Moon")
p2 = uni.frames.pointId("New_Norcia")
print("Light Time New Norcia <-- Moon: ", owlt.eval(tempo.Epoch(), p1, p2, lt_dir), " [s]")

from godot.core import station

stn_name = "New_Norcia"
# Create station book
station_config = cosmos.util.load_json("share/test/GroundStationsDatabase.json")
station_book = station.StationBook(station_config, uni.constants)
# Get the stn object
stn_obj = station_book.get(stn_name)

# Build TDBTT conversion
tdbtt = obs.TDBTT(eph_file="share/test/de432.jpl")
# Define clock offset
clock_offset = common.ConstantScalarTimeEvaluable(0.)
# Create ground station clock
stn_clock = obs.GroundStationClock(uni.frames, stn_name, "Earth", "ICRF", tdbtt, clock_offset)

"""
Omitted are media corrections. 
"""

# Create Ground Station Participant
stn_par = obs.GroundStationParticipant( stn_name, stn_name, stn_name, station_book.get(stn_name), stn_clock)
# Set station tx frequency
stn_freq = common.ConstantScalarTimeEvaluable(8.1e9)
stn_par.setTransmitterFrequency(stn_freq)

# Define frequency band and set up biases and delays
freq_band = obs.FrequencyBand.X
const_eval = common.ConstantScalarTimeEvaluable(0.0)
stn_par.setDopplerBiases({freq_band: const_eval}, {freq_band: const_eval})
stn_par.setGroupDelays({freq_band: const_eval}, {freq_band: const_eval})
stn_par.setPhaseDelays({freq_band: const_eval}, {freq_band: const_eval})
stn_par.setRangeBiases({freq_band: const_eval}, {freq_band: const_eval})