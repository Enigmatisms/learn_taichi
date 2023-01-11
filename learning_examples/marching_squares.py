"""
    Marching square SDF zero set visualization with taichi
    @date 2023.1.8
"""
import tqdm
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
    def __init__(self, width, height, properties: dict):
        self.w = width
        self.h = height
        self.threshold = properties['threshold']
        self.ball_max  = properties['max_radius']
        self.ball_min  = properties['min_radius']
        self.ball_num  = properties['ball_num']
        self.vel_bound = properties['velocity_bound']
        assert(self.ball_num < 16)                                              # up to 15 balls

        # boundary padding for SDF map (1 pixel for bottom right direction)
        self.sdf_maps = ti.field(dtype = ti.f32, shape = (width + 1, height + 1, self.ball_num))
        self.sum_map  = ti.field(dtype = ti.f32, shape = (width + 1, height + 1))
        # RGB image is stored in vector.field
        self.pixels  = ti.Vector.field(3, dtype = ti.f32, shape = (width, height))     

        self.ball_radii = ti.field(dtype = ti.f32, shape = self.ball_num)       # 1 dim radius
        self.ball_pos   = ti.Vector.field(2, ti.f32, self.ball_num)             # 2 dims (px, py)
        self.ball_vel   = ti.Vector.field(2, ti.f32, self.ball_num)             # 2 dims (vx, vy)
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
            self.ball_pos[i]   = rand2d(radius, radius, self.w - radius, self.h - radius)
            self.ball_vel[i]   = rand2d(-self.vel_bound, -self.vel_bound, self.vel_bound, self.vel_bound)

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
            # SDF calculated at pixel center
            radius   = self.ball_radii[ball_id]
            ball_pos = self.ball_pos[ball_id]
            pos      = ti.Vector([i - 0.5, j - 0.5])
            distance = (pos - ball_pos).norm() - radius
            self.sdf_maps[i, j, ball_id] = ti.min(ti.exp(-0.1 * distance), 1.0)
    
    @ti.kernel
    def calculate_color(self):
        self.sum_map.fill(0.)
        self.pixels.fill(0.)
        for i, j, ball_id in self.sdf_maps:
            self.sum_map[i, j] += self.sdf_maps[i, j, ball_id]
        for i, j in self.sum_map:
            sur_vals = ti.Vector([self.sum_map[i, j], self.sum_map[i + 1, j], 
                    self.sum_map[i, j + 1], self.sum_map[i + 1, j + 1]]) - self.threshold
            if (sur_vals >= 0).any():       # inside ball indexed by ball_id
                out_color = ti.Vector([0., 0., 0.])
                sum_dist = 0.
                for idx in range(self.ball_num):
                    dist = self.sdf_maps[i, j, idx]
                    sum_dist  += dist
                    out_color += self.color_map[idx] * dist
                out_color /= sum_dist
                self.pixels[i, j] = out_color

if __name__ == "__main__":
    W = 800
    H = 800
    ball_prop = {
        'min_radius': 50., 'max_radius': 120.0,
        'ball_num': 12, 'velocity_bound': 7.5, 'threshold': 0.05
    }
    write_video = True

    ti.init(arch = ti.gpu, random_seed = 1)

    marcher = MarchingSquareSDF(W, H, ball_prop)

    if write_video:
        frame_rate = 25
        duration = 5.
        result_dir = "./outputs"
        video_manager = ti.tools.VideoManager(output_dir=result_dir, framerate=frame_rate, automatic_build=False)
        for i in tqdm.tqdm(range(int(duration * frame_rate))):
            marcher.calculate_sdf()
            marcher.calculate_color()
            marcher.update_balls()
            pixels_img = marcher.pixels.to_numpy()
            video_manager.write_frame(pixels_img)
    else:
        gui = ti.GUI('Marching Squares', res = (W, H))
        time_t = 0
        while gui.running:
            marcher.calculate_sdf()
            marcher.calculate_color()
            marcher.update_balls()
            gui.set_image(marcher.pixels)
            gui.show()
    