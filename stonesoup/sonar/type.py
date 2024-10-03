from ..types.angle import Angle
from .functions import mod_direction

class Direction(Angle):
    """Recovered directional cosine angle class.

    Direction handles modulo arithmetic for adding and subtracting angles. \
    The return type for addition and subtraction is Direction.
    Multiplication or division produces a float object rather than Direction.
    """
    @staticmethod
    def mod_angle(value):
        return mod_direction(value)
