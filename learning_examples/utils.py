"""
    Utility functions
    @date 2023.1.14
"""

import taichi as ti
import taichi.math as tm

PI_2 = tm.pi * 2.0

@ti.func
def valid_angle(ang):
    ret = ang
    if ang > tm.pi:
        ret = ang - PI_2
    elif ang < -tm.pi:
        ret = ang + PI_2
    return ret

@ti.func
def bound_rand(mini: ti.f32, maxi: ti.f32) -> ti.f32:
    diff = maxi - mini
    return diff * ti.random(ti.f32) + mini

@ti.func
def rand2d(min_x, min_y, max_x, max_y):
    return ti.Vector([bound_rand(min_x, max_x), bound_rand(min_y, max_y)])