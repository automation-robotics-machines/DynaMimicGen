import math
import numpy as np
import random
import quaternion
from scipy.spatial.transform import Rotation as R


def gcd(a, b):
    if (a < b):
        return gcd(b, a)

        # base case
    if abs(b) < 1e-5:
        return a
    else:
        return gcd(b, a - math.floor(a / b) * b)


def lcm(a, b=None):
    if b is not None:
        return (a * b) / gcd(a, b)
    else:
        out = lcm(a[0], a[1])
        for i in range(2, len(a)):
            out = lcm(out, a[i])

        return out


def random_quaternion():
    z = 100000
    while z > 1.0:
        x = random.uniform(-1.0, 1.0)
        y = random.uniform(-1.0, 1.0)
        z = x * x + y * y
    w = 100000
    while w > 1.0:
        u = random.uniform(-1.0, 1.0)
        v = random.uniform(-1.0, 1.0)
        w = u * u + v * v

    s = np.sqrt((1 - z) / w)
    return quaternion.quaternion(x, y, s * u, s * v)


def normalize_vector(v):
    return v/np.linalg.norm(v)


def calculate_rotation_between_vectors(v_from, v_to):
    v_from = normalize_vector(v_from)
    v_to = normalize_vector(v_to)
    print("v from: " + str(v_from))
    print("v to: " + str(v_to))

    angle = np.arccos(np.dot(v_from, v_to) / (np.linalg.norm(v_from) * np.linalg.norm(v_to)))
    axis = np.cross(v_from, v_to)
    axis_norm = np.linalg.norm(axis)

    print("axis" + str(axis))
    print("angle" + str(angle))

    if axis_norm < 1e-5:
        if angle < 1e-5:
            print("0 deg")
            return np.identity(3)
        elif np.pi - angle < 1e-5:
            # Choose any vector orthogonal to x_world
            print("180 deg")
            # axis = np.array([0, 1, 0])
            axis = arbitrary_orthogonal_vector(v_from)
            print("new axis: " + str(axis))
            axis_norm = np.linalg.norm(axis)

    axis /= axis_norm

    r = R.from_rotvec(axis * angle)
    Rv = r.as_matrix()

    print(Rv)
    return Rv


def arbitrary_orthogonal_vector(vec):
    Ax = np.abs(vec[0])
    Ay = np.abs(vec[1])
    Az = np.abs(vec[2])
    if Ax < Ay:
        P = np.array([0, -vec[2], vec[1]]) if Ax < Az else np.array([-vec[1], vec[0], 0])
    else:
        P = np.array([vec[2], 0, -vec[0]]) if Ay < Az  else np.array([-vec[1], vec[0], 0])

    return P


def quat_to_axang(q):
    for i in range(1, len(q)):
        if np.dot(q[i].vec, q[i-1].vec) < 0:
            q[i] *= -1
    return quaternion.as_float_array(2*np.log(np.normalized(q)))[..., 1:]
