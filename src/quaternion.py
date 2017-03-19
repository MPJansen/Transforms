import numpy as np
from rotation_matrix import *


class Quaternion():
    """
    (Normalized) Quaternion to perform quaternion algebra.
        :Example:
        q = Quaternion([0.0, 1.0, 0.0, 0.0])
        q2 = q * q
        q_inverted = q.i
        R = quaternion_to_rotation_matrix(q2)

        :param q:  quaternion element vector [w, x, y, z]
        >>> quaternions = np.random.rand(5,4)
        >>> q_list = [Quaternion(q/np.linalg.norm(q)).i * Quaternion(q/np.linalg.norm(q)) for q in quaternions]
        >>> all([np.allclose(q.q, Quaternion([1.0, 0.0, 0.0, 0.0]).q) for q in q_list])
        True
        >>> q_list_m = [rotmat_to_quaternion(quaternion_to_rotation_matrix(q)) for q in q_list]
        >>> all([np.allclose(q.q, q_m.q) for q, q_m in zip(q_list, q_list_m)])
        True
    """

    def __init__(self, q=None):

        if q is None:
            self._q = np.array([1.0, 0.0, 0.0, 0.0])
        elif isinstance(q, RotMat):
            self._q = rotmat_to_quaternion(q)
        elif isinstance(q, Quaternion):
            self._q = q.q
        else:
            self._q = np.array(q[:4], dtype=float)

    def __matmul__(self, other):
        result = Quaternion()
        result.q[0] = self.q[0] * other.q[0] - self.q[1] * other.q[1] - self.q[2] * other.q[2] - self.q[3] * other.q[3]
        result.q[1] = self.q[0] * other.q[1] + self.q[1] * other.q[0] + self.q[2] * other.q[3] - self.q[3] * other.q[2]
        result.q[2] = self.q[0] * other.q[2] + self.q[2] * other.q[0] - self.q[1] * other.q[3] + self.q[3] * other.q[1]
        result.q[3] = self.q[0] * other.q[3] + self.q[3] * other.q[0] + self.q[1] * other.q[2] - self.q[2] * other.q[1]
        return result

    def __mul__(self, other):
        result = Quaternion()
        if isinstance(other, Quaternion):
            result = self.__matmul__(other)
        elif isinstance(other, float):
            result.q = self.q * other
        else:
            raise TypeError("Unsupported operand type(s) for {}: {} and {}".format('*', type(self), type(other)))
        return result

    def __rmul__(self, other):
        if isinstance(other, float):
            result = Quaternion()
            result.q = self.q * other
            return result
        else:
            raise TypeError("Unsupported operand type(s) for {}: {} and {}".format('*', type(self), type(other)))

    def __add__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(self.q + other.q)
        else:
            raise TypeError("Unsupported operand type(s) for {}: {} and {}".format('+', type(self), type(other)))

    def __sub__(self, other):
        if other is Quaternion:
            return Quaternion(self.q - other.q)
        else:
            raise TypeError("Unsupported operand type(s) for {}: {} and {}".format('-', type(self), type(other)))

    def __repr__(self):
        return "Quaternion: " + self.q.__str__()

    def __str__(self):
        return "Quaternion: " + self.q.__str__()

    @property
    def q(self):
        return self._q

    @property
    def i(self):
        norm = self.norm()
        return Quaternion([self.q[0]/norm, -self.q[1]/norm, -self.q[2]/norm, -self.q[3]/norm])

    def norm(self, order=None):
        return np.linalg.norm(self.q, ord=order)

    @property
    def two_norm(self):
        return np.linalg.norm(self.q, ord=2)

    def exp(self):
        theta = np.sqrt(np.sum(np.square(self.q[1:])))
        sin_theta = np.sin(theta)
        q0 = np.cos(theta)
        if np.abs(theta) > np.finfo(float).eps:
            q = [q0, *(self.q[1:] * sin_theta / theta)]
            result = Quaternion(q)
        else:
            q = [q0, *(self.q[1:])]
            result = Quaternion(q)
        return result

    def log(self):

        theta = np.arccos(self.q[0])
        sin_theta = np.sin(theta)
        q0 = 0.0
        if np.abs(sin_theta) > np.finfo(float).eps:
            q = [q0, *(self.q[1:] / sin_theta * theta)]
            result = Quaternion(q)
        else:
            q = [q0, *(self.q[1:])]
            result = Quaternion(q)
        return result

    def pow(self, p):
        result = self.log()
        result.q *= p
        return result.exp()

    def normalize(self):
        norm = self.two_norm
        self._q /= self.two_norm

    @property
    def conjugate(self):
        return Quaternion([self._q[0], -self._q[1], -self._q[2], -self._q[3]])

    @property
    def w(self):
        return self._q[0]

    @property
    def x(self):
        return self._q[1]

    @property
    def y(self):
        return self._q[2]

    @property
    def z(self):
        return self._q[3]


def quaternion_to_rotation_matrix(self):
    from rotation_matrix import RotMat
    q = self._q

    n = q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2
    s = 0.0 if n == 0.0 else 2.0/n

    wx = s * q[0] * q[1]
    wy = s * q[0] * q[2]
    wz = s * q[0] * q[3]

    xx = s * q[1] * q[1]
    xy = s * q[1] * q[2]
    xz = s * q[1] * q[3]

    yy = s * q[2] * q[2]
    yz = s * q[2] * q[3]
    zz = s * q[3] * q[3]
    R = RotMat(np.array([[1 - (yy + zz), xy - wz, xz + wy],
                        [xy + wz, 1 - (xx + zz), yz - wx],
                        [xz - wy, yz + wx, 1 - (xx + yy)]]))
    return R
    """
    return RotMat(
                np.array([[2 * (q[0]**2 + q[1]**2) - 1.0     , 2*(q[1]*q[2]-q[0]*q[3])           , 2 * (q[1]*q[3] + q[0]*q[2])],
                          [2 * (q[1] * q[2] + q[0] * q[3])   , 2 * (q[0]**2 + q[2]**2) - 1.0     , 2 * (q[2] * q[3] - q[0] * q[1])],
                          [2 * (q[1] * q[2] - q[0] * q[2])   , 2 * (q[2] * q[3] + q[0] * q[1])   , 2 * (q[0]**2 + q[3]**2) - 1.0]])
                )
    """


def slerp(a, b, t):

    dot = np.dot(a.q, b.q)

    if dot >= 0.0:
        result = a * ((a.i * b).pow(t))
    else:
        result = a * ((a.i * (b * -1.0)).pow(t))

    return result


def slerp_no_inv(a, b, t):

    dot = np.dot(a.q, b.q)
    if np.abs(dot) > 0.99999:
        return a

    theta = np.arccos(dot)
    sin_theta = 1.0 / np.sin(theta)
    new_factor = np.sin(t * theta) * sin_theta
    inv_factor = np.sin((1.0 - t) * theta) * sin_theta
    out = a*inv_factor + b*new_factor
    return Quaternion(out)


def squad(p, a, b, q, t):
    sl_pq = slerp_no_inv(p, q, t)
    sl_ab = slerp_no_inv(a, b, t)
    result = sl_pq * (sl_pq.i * sl_ab).pow(2 * t * (1 - t))
    return result


def squad_interpolate(q0, q1, q2, q3, t):
    a0 = a_n(q0, q1, q2)
    a1 = a_n(q1, q2, q3)
    return squad(q1, a0, a1, q2, t)


def a_n(q_n_min, q_n, q_n_plus):
    log = ((q_n.i * q_n_plus).log() + (q_n.i * q_n_min).log()) * -0.25
    return q_n * log.exp()


class QuaternionCubicSpline():

    def __init__(self, quaternion_knots, knot_locations=None):
        if len(quaternion_knots) >= 3:
            self.__knots = quaternion_knots
        else:
            raise ValueError("Number of quaternion knots: {} should be greater than 3".format(len(quaternion_knots)))

        if knot_locations is not None:
            if len(knot_locations) == quaternion_knots:
                if len(set(knot_locations)) < len(knot_locations):
                    raise ValueError("Knot locations should be unique")
                self.__knot_locations = knot_locations
            else:
                raise ValueError("Length of knots ({}) and knot_locations ({}) should be equal".format(len(quaternion_knots), len(knot_locations)))
        else:
            self.__knot_locations = range(len(quaternion_knots))

    def evaluate_at(self, t):
        pass

