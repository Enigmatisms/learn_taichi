
"""
    Distance Field and Signed distance field visualization for several consecutive line segement
    Signed distance field problem is way more harder than distance field problem
    Trying to be concise and fast
    @date 2023.1.5
"""

import taichi as ti
import numpy as np
import taichi.math as tm


W = 800
H = 600
T = 1000000
PNUM = 8

ti.init(arch = ti.gpu)
pixels = ti.field(dtype=float, shape=(W, H))
points_field = ti.Vector.field(2, dtype = ti.f32, shape = PNUM)
normal_field = ti.Vector.field(2, dtype = ti.f32, shape = PNUM - 1)

@ti.func
def is_in_scope(pix_pos, t1, t2):
    vec = t2 - t1
    s2p = pix_pos - t1
    e2p = pix_pos - t2
    beyond_s = tm.dot(vec, s2p) < 0
    beyond_e = tm.dot(vec, e2p) > 0
    ret = 0
    if beyond_s or beyond_e:
        ret = 1 if beyond_s else 2
    return ret

@ti.kernel
def draw_distance_field(scaler: ti.f32):
    """
        FIXME: Note that we might not be able to input field as arguments? this must be changed \\
        - Note that, we can not input field as argument, since I can not annotate it with right type...
        (Note1): kernel (if return anything), should have type annotation. For example:
        ```
        @ti.kernel
        def get_dot(v1: ti.vector(2, ti.f32), ti.vector(2, ti.f32)):
            return ti.math.dot(v1, v2)
        ```
        (Note2): ti.math.dot (and some other similar functions) can not be called in Python scope
        - This is kind of like CUDA `__device__` function

        (Question1): Can we make `ti.vector()` agnostic of vector length? How to write this template? \\
        (Question2): When do we need type annotation?

        Input:
        - normal: normalized normal vector
    """
    # line segements are represented by vector field, lsegs stores the vertices of the line segment
    line_seg_num = ti.static(points_field.shape[0]) - 1
    for i, j in pixels: # the for loop of the outer-most scope is parallel
        min_value = 1e9
        pix_pos = ti.Vector([i, j])
        for k in range(line_seg_num):
            sp = points_field[k]
            ep = points_field[k + 1]
            scope_id = is_in_scope(pix_pos, sp, ep)
            distance = 1e9
            if is_in_scope(pix_pos, sp, ep) == 0:
                distance = abs(tm.dot(pix_pos - sp, normal_field[k]))
            else:
                pt = sp if scope_id & 1 else ep
                distance = (pix_pos - pt).norm()
            min_value = min(min_value, distance)
        pixels[i, j] = 1. - min(min_value / scaler, 1.0)


def generate_random_chain(time: float, width: float, height: float, pnum: int = 5):
    # generate SDF line segments
    freq = 2 * np.pi / (width * 0.5)
    w_margin = width * 0.05
    h_margin = height * 0.05
    half_height = (height - 2 * h_margin) / 2.
    xs = np.linspace(w_margin, width - w_margin, pnum)
    ys = half_height * np.sin(freq * (xs + time)) + height / 2
    dxs = xs[1:] - xs[:-1]
    dys = -(ys[1:] - ys[:-1])
    normals = np.stack((dys, dxs), axis = 1)
    normals /= np.linalg.norm(normals, axis = -1)[:, None]
    
    for i, (x, y) in enumerate(zip(xs, ys)):
        points_field[i] = ti.Vector((x, y))
    for i in range(pnum - 1):
        normal_field[i] = ti.Vector(normals[i])

gui = ti.GUI('SDF visualize', res = (W, H))

for i in range(T):
    generate_random_chain(i, W, H, PNUM)
    draw_distance_field(100.0)
    gui.set_image(pixels)
    gui.show()