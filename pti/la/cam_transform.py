"""
    3D space transformation / pinhole camera model
    @author: Qianyue He
    @date: 2023-1-23
"""

import numpy as np
import taichi as ti
from numpy import ndarray as Arr
from scipy.spatial.transform import Rotation as Rot

__all__ = ['fov2focal', 'rotation_between']

def fov2focal(fov: float, img_size):
    fov = fov / 180. * np.pi
    return 0.5 * img_size / np.tan(.5 * fov)
    
def rotation_between(fixed: Arr, target: Arr) -> Arr:
    """
        INPUT arrays [MUST] be normalized
        Transform parsed from xml file is merely camera orientation
        Orientation should be transformed to be camera rotation matrix
        Rotation from <fixed> vector to <target> vector, defined by cross product and angle-axis
    """
    axis = np.cross(fixed, target)
    dot = np.dot(fixed, target)
    if abs(dot) > 1. - 1e-5:            # nearly parallel
        return np.sign(dot) * np.eye(3, dtype = np.float32)  
    else:
        # Not in-line, cross product is valid
        axis /= np.linalg.norm(axis)
        axis *= np.arccos(dot)
        return Rot.from_rotvec(axis).as_matrix()
