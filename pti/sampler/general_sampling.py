"""
    General sampling functions for direction sampling
    sampling azimuth angle and zenith angle
    @author: Qianyue He
    @date: 2023-1-27
"""

import taichi as ti
import taichi.math as tm
from taichi.math import vec3

__all__ = ['ramped_hemisphere', 'uniform_hemisphere']

pi_inv = 1. / tm.pi

@ti.func
def ramped_hemisphere():
    """
        Zenith angle (cos theta) follows a ramped PDF (triangle like)
        Azimuth angle (itself) follows a uniform distribution
    """
    eps = ti.random(float)
    cos_theta = ti.sqrt(eps)       # zenith angle
    sin_theta = ti.sqrt(1 - cos_theta * cos_theta)
    phi = 2. * tm.pi * ti.random(float)         # uniform dist azimuth angle
    pdf = cos_theta * pi_inv        # easy to deduct, just try it
    # rotational offset w.r.t axis [0, 1, 0] & pdf
    return vec3([ti.cos(phi) * sin_theta, cos_theta, ti.sin(phi) * sin_theta]), pdf

@ti.func
def uniform_hemisphere():
    """
        Both zenith (cosine) and azimuth angle (original) are uniformly distributed
    """
    cos_theta = ti.random(float)
    sin_theta =  ti.sqrt(1 - cos_theta * cos_theta)
    phi = 2. * tm.pi * ti.random(float)
    pdf = 0.5 * pi_inv
    return vec3([ti.cos(phi) * sin_theta, cos_theta, ti.sin(phi) * sin_theta]), pdf
