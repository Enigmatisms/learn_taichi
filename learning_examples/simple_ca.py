"""
    Very simple cellular automata that you can build within 30min
    @date 2023.1.11
"""
import tqdm
import taichi as ti
import numpy as np

@ti.data_oriented
class CellularAutomata:
    def __init__(self, width, height, pix_size, prop):
        self.w = width
        self.h = height
        self.radius         = prop['radius']
        self.disp_threshold = prop['disp_th']
        self.idle_bias      = prop['idle_bias']
        self.active_th      = prop['active_th']
        self.idle_coeff     = prop['idle_coeff']
        self.init_proba     = prop['init_proba']
        self.sigma          = prop['gauss_sigma']
        self.active_bias    = prop['active_bias']
        self.max_neighbor   = prop['max_neighbor']
        self.pixels   = ti.field(ti.f32, (width * pix_size, height * pix_size))
        self.now_vals = ti.field(ti.f32, (width, height))
        self.old_vals = ti.field(ti.f32, (width, height))
        self.initialize()

    @ti.func
    def skip(self, dx, dy, id_x, id_y):
        ret_val = False
        if dx == 0 and dy == 0: ret_val = True
        if ret_val == False and id_x < 0 or id_x >= self.w: ret_val = True
        if ret_val == False and id_y < 0 or id_y >= self.h: ret_val = True
        return ret_val

    @ti.kernel
    def update_ca(self):
        for i, j in self.old_vals:
            cnt = 0
            for dx in range(-self.radius, self.radius + 1):
                for dy in range(-self.radius, self.radius + 1):
                    id_x = i + dx
                    id_y = j + dy
                    if self.skip(dx, dy, id_x, id_y): continue
                    if self.old_vals[id_x, id_y] > self.active_th:
                        cnt += 1
            val = self.old_vals[i, j]
            now_rand = ti.randn(ti.f32)
            if cnt >= self.max_neighbor:        # force deactivate
                val = self.active_th - 1e-3
                self.now_vals[i, j] -= ti.random(ti.f32)
            # ti.static applied to range will fail
            for dx in range(-self.radius, self.radius + 1):
                for dy in range(-self.radius, self.radius + 1):
                    id_x = i + dx
                    id_y = j + dy
                    scaler = 1.0 if abs(dx) < 2 and abs(dy) < 2 else 0.5
                    if self.skip(dx, dy, id_x, id_y): continue
                    if val > self.active_th:
                        self.now_vals[id_x, id_y] += now_rand * scaler + self.active_bias
                    else:
                        self.now_vals[id_x, id_y] += now_rand * scaler * self.idle_coeff + self.idle_bias
    
    @ti.kernel
    def update_viz(self, pix_sz: ti.i32):
        for i, j in self.pixels:
            idx = i // pix_sz
            idy = j // pix_sz
            if self.now_vals[idx, idy] > self.disp_threshold:
                self.pixels[i, j] = 1
            else:
                self.pixels[i, j] = 0

    def initialize(self):
        rand_field = np.random.rand(self.w, self.h)
        rand_field[rand_field < self.init_proba] = 0.0
        self.old_vals.from_numpy(rand_field)
        self.now_vals.from_numpy(rand_field)

    def update_old(self):
        self.old_vals.copy_from(self.now_vals)

if __name__ == "__main__":
    W = 200
    H = 200
    PIX_SZ = 4
    save_image = True
    ti.init(arch = ti.gpu, random_seed = 1)

    prop = {'gauss_sigma': 1.0, 'active_th': 0.5, 'active_bias': 0.5, 'idle_coeff': 0.2, 
            'idle_bias': -0.05, 'disp_th': 0.9, 'init_proba': 0.999, 'radius': 2, 'max_neighbor': 15}

    ca = CellularAutomata(W, H, PIX_SZ, prop)

    if save_image:
        duration = 5.0
        frame_rate = 20.0
        result_dir = "./outputs"
        video_manager = ti.tools.VideoManager(output_dir=result_dir, framerate=frame_rate, automatic_build=False)
        for i in tqdm.tqdm(range(int(duration * frame_rate))):
            ca.update_ca()
            ca.update_old()
            ca.update_viz(PIX_SZ)
            pixels_img = ca.pixels.to_numpy()
            video_manager.write_frame(pixels_img)
    else:
        gui = ti.GUI('Cellular Automata', res = (W * PIX_SZ, H * PIX_SZ))
        while not gui.get_event(ti.GUI.ESCAPE):
            ca.update_ca()
            ca.update_old()
            ca.update_viz(PIX_SZ)
            gui.set_image(ca.pixels)
            gui.show()
