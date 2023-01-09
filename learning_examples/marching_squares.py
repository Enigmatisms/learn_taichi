"""
    Marching square SDF zero set visualization with taichi
    @date 2023.1.8
"""

import taichi as ti
from color_map import COLOR_MAP

"""
    Color mixing strategy: Since the border of SDF (zero-set) is drawn as pure white (black background)
    the color inside (and the convex combination of the colors) will be clipped to (30, 30, 30) - (230, 230, 230)
    As is said above, color is mixed in a convex combination way, and tnere will be a tag field (u8)
    places with positive sdf value will be marked as 0x0000, and corresponding bit will be set to 1
    if a pixel is inside the corresponding ball. The color of a negative valued pixel (or tagged with non-zero u8)
    will be the convex weighted average of the color of each participating ball.
"""

@ti.func
def bound_rand(mini: ti.f32, maxi: ti.f32) -> ti.f32:
    diff = maxi - mini
    return diff * ti.random(ti.f32) + mini

@ti.func
def rand2d(min_x, min_y, max_x, max_y):
    return ti.Vector([bound_rand(min_x, max_x), bound_rand(min_y, max_y)])

@ti.data_oriented
class MarchingSquareSDF:
    def __init__(self, width, height, ball_prop: dict):
        self.w = width
        self.h = height
        self.ball_max = ball_prop['max_radius']
        self.ball_min = ball_prop['min_radius']
        self.ball_num = ball_prop['ball_num']
        assert(self.ball_num < 16)                                              # up to 15 balls
        self.vel_bound = ball_prop['velocity_bound']

        # boundary padding for SDF map (1 pixel for bottom right direction)
        self.sdf_maps = ti.field(dtype = ti.f32, shape = (width + 1, height + 1, self.ball_num))
        self.tag_map = ti.field(dtype = ti.u16, shape = (width, height))        # tag map

        # RGB image is stored in vector.field
        self.pixels  = ti.Vector.field(3, dtype = ti.f32, shape = (width, height))     

        self.ball_pos   = ti.Vector.field(2, ti.f32, self.ball_num)             # 2 dims (px, py, vx, vy, radius)
        self.ball_vel   = ti.Vector.field(2, ti.f32, self.ball_num)             # 2 dims (px, py, vx, vy, radius)
        self.ball_radii = ti.field(dtype = ti.f32, shape = self.ball_num)       # 1 dim (px, py, vx, vy, radius)
        self.color_map  = ti.Vector.field(3, dtype = ti.f32, shape = self.ball_num)
        for i in range(self.ball_num):
            self.color_map[i] = COLOR_MAP[i]

        self.initialize_balls()

    @ti.kernel
    def initialize_balls(self):
        # initialize ball with random radius / position / speed (2D)
        for i in range(self.ball_num):
            radius = bound_rand(self.ball_min, self.ball_max)
            self.ball_radii[i] = radius
            self.ball_pos[i] = rand2d(radius, radius, self.w - radius, self.h - radius)
            self.ball_vel[i] = rand2d(-self.vel_bound, self.vel_bound, -self.vel_bound, self.vel_bound)

    def update_balls(self):
        # update ball position (boundary checks - velocity changes)
        for idx in range(self.ball_num):
            pos = self.ball_pos[idx]
            r = self.ball_radii[idx]
            if pos[0] <= r or pos[0] >= self.w - r:
                self.ball_vel[idx][0] *= -1
            if pos[1] <= r or pos[1] >= self.h - r:
                self.ball_vel[idx][1] *= -1
            self.ball_pos[idx] += self.ball_vel[idx]

    @ti.kernel
    def calculate_sdf(self):
        # Note that SDF value is calculate per pixel (top-left corner)
        for i, j, ball_id in self.sdf_maps:
            # calculate distance to ball indexed by ball_id
            # SDF calculated at pixel center
            radius      = self.ball_radii[ball_id]
            ball_pos = self.ball_pos[ball_id]
            pos = ti.Vector([i - 0.5, j - 0.5])
            self.sdf_maps[i, j, ball_id] = (pos - ball_pos).norm() - radius
    
    @ti.kernel
    def calculate_tag(self):
        for i, j, ball_id in self.sdf_maps:
            # check the surrouding SDF values (4)
            sur_vals = ti.Vector([
                self.sdf_maps[i, j, ball_id],
                self.sdf_maps[i + 1, j, ball_id],
                self.sdf_maps[i, j + 1, ball_id],
                self.sdf_maps[i + 1, j + 1, ball_id]
            ])
            all_non_pos = (sur_vals <= 0).all()
            if all_non_pos:       # inside ball indexed by ball_id
                tag = 1 << ball_id
                ti.atomic_add(self.tag_map[i, j], tag)

    def extract_indices(tag: ti.u16):
        indices = []
        for i in range(15):
            if ((1 << i) & tag):
                indices.append(i)
        return indices

    @ti.kernel
    def calculate_color(self):
        for i, j in self.pixels:
            # get tag and SDF value (add)
            tag = self.tag_map[i, j]
            # calculate color
            if tag == 0:            # black for empty spot
                self.pixels[i, j] = ti.Vector([0., 0., 0.])
            else:
                out_color = ti.Vector([0., 0., 0.])
                sum_dist = 0.
                for idx in range(15):
                    if ((1 << idx) & tag):
                        radius = self.ball_radii[idx]
                        rel_dist = (self.sdf_maps[i, j, idx] + radius) / radius
                        dist = ti.math.exp(-5.0 * rel_dist)
                        sum_dist  += dist
                        out_color += self.color_map[idx] * dist
                out_color /= sum_dist
                self.pixels[i, j] = out_color

    def reset_tag(self):
        self.tag_map.fill(0x0000)

if __name__ == "__main__":
    W = 1280
    H = 960
    T = 10000000
    BALL_NUM = 7
    ball_prop = {
        'min_radius': 50., 'max_radius': 150.0,
        'ball_num': 7, 'velocity_bound': 1.0
    }

    ti.init(random_seed = 0)
    gui = ti.GUI('SDF visualize', res = (W, H))

    marcher = MarchingSquareSDF(W, H, ball_prop)

    while gui.running:
        marcher.reset_tag()
        marcher.calculate_sdf()
        marcher.calculate_tag()
        marcher.calculate_color()
        marcher.update_balls()
        gui.set_image(marcher.pixels)
        gui.show()




    