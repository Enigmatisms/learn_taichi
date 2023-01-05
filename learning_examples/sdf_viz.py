
"""
    Distance Field and Signed distance field visualization for several consecutive line segement
    Signed distance field problem is way more harder than distance field problem
    Trying to be concise and fast
    @date 2023.1.5
"""

import taichi as ti
import taichi.math as tm

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
def draw_distance_field(pixels: ti.field, lsegs: ti.Vector.field, normal = ti.Vector.field):
    """
        FIXME: Note that we might not be able to input field as arguments? this must be changed \\
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
    line_seg_num = ti.static(lsegs.shape[0]) - 1
    for i, j in pixels: # the for loop of the outer-most scope is parallel
        min_value = 1e9
        pix_pos = ti.Vector([i, j])
        for k in range(line_seg_num):
            sp = lsegs[k]
            ep = lsegs[k + 1]
            scope_id = is_in_scope(pix_pos, sp, ep)
            if is_in_scope(pix_pos, sp, ep) == 0:
                distance = abs(tm.dot(pix_pos - sp, normal[k]))
            else:
                pt = sp if scope_id & 1 else ep
                distance = (pix_pos - pt).norm()
            min_value = min(min_value, distance)
        pixels[i, j] = min_value

if __name__ == '__main__':
    ti.init(arch = ti.gpu)

    # Main logic is completed, some traits of taichi lang should be figured out