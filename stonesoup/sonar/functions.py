import numpy as np

def mod_direction(angle):
    x = angle % (2 * np.pi)
    if x > np.pi:
        x = 2 * np.pi - x
    return x