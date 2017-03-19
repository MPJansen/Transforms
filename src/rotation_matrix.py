import numpy as np

class RotMat(object):
    def __init__(self, R=None):
        if R is None:
            self.R = np.eye(3)
        else:
            self.R = R[:3, :3]

    def __matmul__(self, other):
        return RotMat(self.R @ other.R)

    def __str__(self):
        return "RotMat:\n " + self.R.__str__()

    @property
    def i(self):
        return RotMat(self.R.T)


def rotmat_to_quaternion(rotmat):
    from quaternion import Quaternion
    trace = np.trace(rotmat.R)
    R = rotmat.R
    if trace > np.finfo(float).eps:
        root = np.sqrt(1.0 + trace)
        S = 0.5 / root
        q0 = 0.5 * root
        q1 = (R[2, 1] - R[1, 2]) * S
        q2 = (R[0, 2] - R[2, 0]) * S
        q3 = (R[1, 0] - R[0, 1]) * S

    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        root = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        S = 0.5 / root
        q0 = (R[2, 1] - R[1, 2]) * S
        q1 = 0.5 * root
        q2 = (R[0, 1] + R[1, 0]) * S
        q3 = (R[0, 2] + R[2, 0]) * S

    elif R[1, 1] > R[2, 2]:
        root = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        S = 0.5 / root
        q0 = (R[0, 2] - R[2, 0]) * S
        q1 = (R[0, 1] + R[1, 0]) * S
        q2 = 0.5 * root
        q3 = (R[1, 2] + R[2, 1]) * S

    else:
        root = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        S = 0.5 / root
        q0 = (R[1, 0] - R[0, 1]) * S
        q1 = (R[0, 2] + R[2, 0]) * S
        q2 = (R[1, 2] + R[2, 1]) * S
        q3 = 0.5 * root

    return Quaternion([q0, q1, q2, q3])



