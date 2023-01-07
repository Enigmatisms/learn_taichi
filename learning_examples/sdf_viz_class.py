
"""
    Distance Field visualization for several consecutive line segement
    A class based example, since I really hate global variables, which are so ugly!
    Signed distance field problem is way more harder than distance field problem
    @date 2023.1.7
"""

import numpy as np
import taichi as ti
import taichi.math as tm

@ti.data_oriented
class TaichiSDF:
    def __init__(self, width, height, pnum = 5):
        self.w = width
        self.h = height
        self.pnum = pnum

        self.pixels = ti.field(dtype=float, shape=(width, height))
        self.points_field = ti.Vector.field(2, dtype = ti.f32, shape = pnum)
        self.normal_field = ti.Vector.field(2, dtype = ti.f32, shape = pnum - 1)

    @staticmethod
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
    def draw_distance_field(self, scaler: ti.f32):
        """
            Note that, (i, j) - pixels are computed in parallel. But for each line segment, computation is serialized \\
            Therefore, SDF logic can be easily incorporated. But, the difficult part is not in taichi   \\
            But algorithm design itself. Since my goal is to learn taichi, I will skip these difficult part.
        """
        line_seg_num = ti.static(self.points_field.shape[0]) - 1
        for i, j in self.pixels: # the for loop of the outer-most scope is parallel
            min_value = 1e9
            pix_pos = ti.Vector([i, j])
            for k in range(line_seg_num):
                sp = self.points_field[k]
                ep = self.points_field[k + 1]
                scope_id = TaichiSDF.is_in_scope(pix_pos, sp, ep)
                distance = 1e9                                      # taichi compiler requires this to be pre-defined
                # taichi scope is COMPILED! Therefore no dynamic inference
                if TaichiSDF.is_in_scope(pix_pos, sp, ep) == 0:
                    distance = abs(tm.dot(pix_pos - sp, self.normal_field[k]))
                else:
                    pt = sp if scope_id & 1 else ep
                    distance = (pix_pos - pt).norm()
                min_value = min(min_value, distance)
            self.pixels[i, j] = 1. - min(min_value / scaler, 1.0)

    def generate_random_chain(self, time: float):
        """
            We don't have to make this a taichi based function
            since the computation here is light compared to SDF calculation
        """
        freq = 2 * np.pi / (self.w * 0.5)
        w_margin = self.w * 0.05
        h_margin = self.h * 0.05
        half_height = (self.h - 2 * h_margin) / 2.
        xs = np.linspace(w_margin, self.w - w_margin, self.pnum)
        ys = half_height * np.sin(freq * (xs + time)) + self.h / 2
        dxs = xs[1:] - xs[:-1]
        dys = -(ys[1:] - ys[:-1])
        normals = np.stack((dys, dxs), axis = 1)
        normals /= np.linalg.norm(normals, axis = -1)[:, None]
        
        for i, (x, y) in enumerate(zip(xs, ys)):
            self.points_field[i] = ti.Vector((x, y))
        for i in range(self.pnum - 1):
            self.normal_field[i] = ti.Vector(normals[i])

if __name__ == '__main__':
    W = 800
    H = 600
    PNUM = 6
    T = 1000000
    ti.init(arch = ti.gpu)
    gui = ti.GUI('SDF visualize', res = (W, H))
    sdf = TaichiSDF(W, H, PNUM)

    time_t = 0
    while gui.running:
        sdf.generate_random_chain(time_t % T)
        sdf.draw_distance_field(100.0)
        gui.set_image(sdf.pixels)
        gui.show()
        time_t += 1