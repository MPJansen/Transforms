import numpy as np
from quaternion import *
from rotation_matrix import *

class Transform(object):
    """
    Homogeneous transform object
    :Example:
    T = Transform(np.zeros(3), Quaternion())


    >>> quaternions = [Quaternion(q/np.linalg.norm(q)) for q in np.random.rand(5,4)]
    >>> positions = [p for p in np.random.rand(100,3)]
    >>> transform_list = [Transform(position, orientation) for position, orientation in zip(positions, quaternions)]
    >>> inverted_list  = [transform.i @ transform for transform in transform_list]
    >>> all([np.allclose(transform.position, np.zeros(3)) and np.allclose(transform.orientation.q, Quaternion().q) for transform in inverted_list])
    True
    """

    def __init__(self, position=None, orientation=None):

        if position is None:
            self._position = np.zeros(3)
        else:
            self._position = position[:3]

        if orientation is None:
            self._orientation = Quaternion()
        else:
            self._orientation = Quaternion(orientation)

    def __repr__(self):
        return "Position: " + self._position.__str__() + " "+ self.orientation.__repr__()

    def __matmul__(self, other):

        orientation = self.orientation * other.orientation
        # Hamilton quaternion product
        position = ((self.orientation * Quaternion([0, *other.position]) * self.orientation.i).q
                    + np.array([0, *self.position]))[1:]
        return Transform(position, orientation)

    @property
    def i(self):
        position = -((self.orientation.i * Quaternion([0, *self.position]) * self.orientation).q[1:])
        orientation = self.orientation.i
        return Transform(position, orientation)

    @property
    def position(self):
        return self._position

    @property
    def orientation(self):
        return self._orientation
