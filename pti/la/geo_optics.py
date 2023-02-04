"""
    Some geometric optics functions
    @author: Qianyue He
    @date: 2023-2-4
"""

import numpy as np
import taichi as ti
import taichi.math as tm
from taichi.math import vec3, mat3
from numpy import ndarray as Arr
from scipy.spatial.transform import Rotation as Rot

__all__ = ['inci_reflect_dir', 'exit_reflect_dir']

@ti.func
def inci_reflect_dir(ray: vec3, normal: vec3):
    dot = tm.dot(normal, ray)
    return (ray - 2 * normal * dot).normalized(), dot

@ti.func
def exit_reflect_dir(ray: vec3, normal: vec3):
    dot = tm.dot(normal, ray)
    return (2 * normal * dot - ray).normalized(), dot