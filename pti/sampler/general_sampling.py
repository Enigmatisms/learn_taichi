"""
    General sampling functions for direction sampling
    sampling azimuth angle and zenith angle
    @author: Qianyue He
    @date: 2023-1-27
"""

import taichi as ti
import taichi.math as tm
from taichi.math import vec3

__all__ = ['cosine_hemisphere', 'uniform_hemisphere']

pi_inv = 1. / tm.pi

@ti.func
def cosine_hemisphere():
    """
        Zenith angle (cos theta) follows a ramped PDF (triangle like)
        Azimuth angle (itself) follows a uniform distribution
    """
    eps = ti.random(float)
    cos_theta = ti.sqrt(eps)       # zenith angle
    sin_theta = ti.sqrt(1. - cos_theta * cos_theta)
    phi = 2. * tm.pi * ti.random(float)         # uniform dist azimuth angle
    pdf = cos_theta * pi_inv        # easy to deduct, just try it
    # rotational offset w.r.t axis [0, 1, 0] & pdf
    
    return vec3([tm.cos(phi) * sin_theta, cos_theta, tm.sin(phi) * sin_theta]), pdf

@ti.func
def uniform_hemisphere():
    """
        Both zenith (cosine) and azimuth angle (original) are uniformly distributed
    """
    cos_theta = ti.random(float)
    sin_theta =  ti.sqrt(1 - cos_theta * cos_theta)
    phi = 2. * tm.pi * ti.random(float)
    pdf = 0.5 * pi_inv
    return vec3([tm.cos(phi) * sin_theta, cos_theta, tm.sin(phi) * sin_theta]), pdf

@ti.func
def sample_triangle(dv1: vec3, dv2: vec3):
    """
        Sample on a mesh triangle
    """
    u1 = ti.random(float)
    u2 = ti.random(float)
    triangle_pt = dv1 * u1 + dv2 * u2
    if u1 + u2 > 1.0:
        triangle_pt = dv1 + dv2 - triangle_pt
    return triangle_pt