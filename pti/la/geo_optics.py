"""
    Some geometric optics functions
    @author: Qianyue He
    @date: 2023-2-4
"""

import taichi as ti
import taichi.math as tm
from taichi.math import vec3

__all__ = ['inci_reflect_dir', 'exit_reflect_dir', 'schlick_frensel']

@ti.func
def inci_reflect_dir(ray: vec3, normal: vec3):
    dot = tm.dot(normal, ray)
    return (ray - 2 * normal * dot).normalized(), dot

@ti.func
def exit_reflect_dir(ray: vec3, normal: vec3):
    dot = tm.dot(normal, ray)
    return (2 * normal * dot - ray).normalized(), dot

@ti.func
def schlick_frensel(r_s: vec3, dot_val: ti.f32):
    """ Schlick's Frensel Fraction Approximation [1993] """
    return r_s + (1 - r_s) * tm.pow(1. - dot_val, 5)
