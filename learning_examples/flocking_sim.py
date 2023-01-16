"""
    Flocking simulation, trying to make a unified program according to the things I ve learned
    @date 2022.1.14
    - [ ] Obstacles avoidance (using Rust generated 2D maps)
"""
import numpy as np
import taichi as ti
import taichi.math as tm
from taichi import types as ttypes
from utils import *

vec2f = ttypes.vector(2, ti.f32)
vec2s = ttypes.vector(2, ti.i16)

COS_23PI = np.cos(np.pi * 2 / 3)
SIN_23PI = np.sin(np.pi * 2 / 3)
R = np.float32([[COS_23PI, SIN_23PI], [-SIN_23PI, COS_23PI]])

@ti.data_oriented
class FlockBase:
    def __init__(self, width, height, prop: dict):
        # I do not recommend boids more than 128
        self.w = width
        self.h = height
        self.boid_num   = prop['boid_num']
        # solo parameters
        self.vel_alpha  = prop['vel_alpha']
        self.ang_alpha  = prop['ang_alpha']
        self.vel_clip   = prop['vel_clip']
        self.turn_proba = prop['turn_proba']
        self.vel        = prop['vel']
        self.border     = prop['border']

        self.ang_vel    = ti.field(ti.f32, self.boid_num)
        self.angles     = ti.field(ti.f32, self.boid_num)
        self.pos        = ti.Vector.field(2, ti.f32, self.boid_num)
        self.dirs       = ti.Vector.field(2, ti.f32, self.boid_num)
        self.center     = vec2f([width, height]) / 2.

        width = self.w - self.border * 2
        height = self.h - self.border * 2
        xs = np.random.rand(self.boid_num) * width + self.border
        ys = np.random.rand(self.boid_num) * height + self.border
        np_pos = np.stack((xs, ys), axis = -1)
        self.pos.from_numpy(np_pos)
        self.initialize_boids()

    @ti.func
    def is_out_of_bounds(self, idx) -> bool:
        px, py = self.pos[idx]
        return px > self.w - self.border or px < self.border or \
                py > self.h - self.border or py < self.border

    @ti.func
    def oob_angular_acc(self, idx) -> float:
        pos = self.pos[idx]
        diff = self.center - pos
        tar_angle = ti.atan2(diff[1], diff[0])
        return tar_angle

    # "Virtual" base-class function to be overriden
    @ti.func
    def other_angular_events(self, idx, now_angle, predators, pred_num): return 0.0

    # "Virtual" base-class function to be overriden
    @ti.func
    def post_process(self, idx = None, new_pos = None): pass

    @ti.kernel
    def initialize_boids(self):
        """
            Initialize with random angle, random position is implemented in python scope
        """
        for i in range(self.boid_num):
            self.angles[i] = ti.random() * PI_2 - tm.pi

    @ti.kernel
    def boid_random_vel(self):
        """
            Random angular velocity for boid (smoothed trajectory)
        """
        for i in self.ang_vel:
            angle_vel = self.ang_vel[i]
            angle_vel = angle_vel * self.vel_alpha + tm.clamp(ti.randn(ti.f32), - self.vel_clip,  self.vel_clip) * (1. - self.vel_alpha)
            if ti.random(ti.f32) < self.turn_proba and not self.is_out_of_bounds(i):
                angle_vel = -angle_vel
            self.ang_vel[i] = angle_vel

    @ti.kernel
    def boid_pos_update(self, predators: ti.template(), pred_num: ti.i32):
        for i in self.angles:
            angle_vel = self.ang_vel[i]
            now_angle = self.angles[i]
            if self.is_out_of_bounds(i):           # out of bound angular forces
                angle_vel += 0.7 * valid_angle(self.oob_angular_acc(i) - now_angle)
            angle_vel += self.other_angular_events(i, now_angle, predators, pred_num)

            angle = valid_angle(self.angles[i] + self.ang_alpha * angle_vel)
            self.angles[i] = angle 
            self.dirs[i] = vec2f([ti.cos(angle), ti.sin(angle)])
            new_pos = ti.math.clamp(self.pos[i] + self.vel * self.dirs[i], 0.0, vec2f([self.w - 1e-3, self.h - 1e-3]))
            self.pos[i] = new_pos

            self.post_process(i, new_pos)

    def get_triangles(self):
        scaler = np.float32([[W, H]])
        pos = self.pos.to_numpy() / scaler
        dirs = self.dirs.to_numpy() / scaler * 7
        return pos, pos + dirs, pos + dirs @ R, pos + dirs @ R.T

@ti.data_oriented
class Predators(FlockBase):
    def __init__(self, width, height, prop: dict):
        super().__init__(width, height, prop)
        self.kp           = prop['predator_kp']
        self.hunt_radius  = int(prop['hunt_radius'])
        self.human_ctrl   = ti.field(ti.u8, self.boid_num)
        self.pred_ang_vel = ti.field(ti.f32, self.boid_num)

    @ti.kernel
    def prey(self, 
        pigeon_pos: ti.template(), pigeon_grid: ti.template(), 
        num_ptr: ti.template(), mouse_pos: vec2f, grid_size: ti.f32
    ):
        for i in self.pos:
            self_pos = self.pos[i]
            nearest_diff = vec2f([0., 0.])
            nearest_r    = 1e9
            if self.human_ctrl[i] > 0:
                self.ang_vel[i] = 0.0
                nearest_diff = mouse_pos - self_pos
                nearest_r = 0.0
            else:
                g_i, g_j = int(self_pos[0] / grid_size), int(self_pos[1] / grid_size)
                for j in range(g_i - 1, g_i + 2):
                    for k in range(g_j - 1, g_j + 2):
                        boid_num = num_ptr[i, j]
                        for m in range(boid_num):
                            boid_idx = int(pigeon_grid[j, k, m])
                            pos_diff = pigeon_pos[boid_idx] - self_pos
                            dist = pos_diff.norm()
                            if dist < self.hunt_radius and dist < nearest_r:
                                nearest_diff = pos_diff
                                nearest_r = dist
            if nearest_r <= 1e8:
                angle = ti.atan2(nearest_diff[1], nearest_diff[0])
                self.pred_ang_vel[i] = self.kp * valid_angle(angle - self.angles[i])
            else:
                self.pred_ang_vel[i] = 0.

    @ti.func
    def other_angular_events(self, idx, now_angle, predators, pred_num): 
        return self.pred_ang_vel[idx]

@ti.data_oriented
class Pigeons(FlockBase):
    def __init__(self, width, height, prop: dict):
        super().__init__(width, height, prop)
        # interactions
        self.coh_max    = prop['coh_max']                             # cohesion clamp weight
        self.sep_max    = -prop['sep_max']                            # separation clamp weight
        self.critical_r = prop['critical_r']
        self.alert_r    = prop['alert_r']
        self.align_w    = prop['align_w']
        self.radius     = int(prop['mate_radius'])                      # also as the grid size
        assert(width % self.radius == 0 and height % self.radius == 0)  # grid parameter requirements

        grid_x, grid_y  = width // self.radius, height // self.radius
        self.mask_grids = ti.field(ti.u8)      # storing indices (atomic_add will return the original index, which is thread-safe)
        # if bitmasked is not used here, it would consume a lot of memory
        self.sn_handle  = ti.root.pointer(ti.ij, (grid_x + 2, grid_y + 2))
        self.atom_ptr   = ti.field(ti.u8, (grid_x, grid_y))     # atomic dynamic indices for bitmask (no offset needed)
        self.sn_handle.bitmasked(ti.k, 128).place(self.mask_grids, offset = (-1, -1, 0))
        self.grid_coord = ti.Vector.field(2, ti.i16, self.boid_num)
    
    @ti.func
    def neighbor_info(self, idx):
        """
            Weight for cohesion should be constant. Yet weight for separation repelling should be variable
        """
        self_pos = self.pos[idx]
        g_i, g_j = self.grid_coord[idx]
        sum_dist = 0.0
        weighted_dir    = vec2f([0, 0])
        centroid        = vec2f([0, 0])
        valid           = False
        for i in range(g_i - 1, g_i + 2):
            for j in range(g_j - 1, g_j + 2):
                boid_num = self.atom_ptr[i, j]
                for k in range(boid_num):
                    boid_idx = int(self.mask_grids[i, j, k])
                    if boid_idx == idx: continue
                    valid = True
                    # distance calculation (universal)
                    other_pos = self.pos[boid_idx]
                    dist = ti.exp(-(other_pos - self_pos).norm() / self.radius)
                    sum_dist += dist
                    # alignment angle calculation
                    angle = self.angles[boid_idx]
                    weighted_dir += dist * vec2f([ti.cos(angle), ti.sin(angle)])
                    # centroid can be used by separation and cohesion
                    centroid += dist * other_pos
        # centroid for cohesion, weihghted_repel for separation, neighbor_angle for alignment 
        cp_angle = 0.0 
        cp_weight = 0.0 
        neighbor_angle = 0.0
        if valid:
            centroid /= sum_dist
            cohesion_force = centroid - self_pos         # negative cohesion foirce is repelling force
            cp_weight = tm.clamp(ti.tanh(cohesion_force.norm() / self.critical_r - 1.0), self.sep_max, self.coh_max)
            cohesion_force *= tm.sign(cp_weight)
            cp_weight = ti.abs(cp_weight)
            cp_angle = ti.atan2(cohesion_force[1], cohesion_force[0])
            neighbor_angle = ti.atan2(weighted_dir[1], weighted_dir[0])
        return cp_weight, cp_angle, neighbor_angle, valid

    @ti.func
    def avoid_predators(self, idx: ti.i32, now_ang: ti.f32, predators: ti.template(), pred_num: ti.i32):
        self_pos = self.pos[idx]
        sum_weight = 0.0
        avg_exit = vec2f([0, 0])        # the way out for the pigeon
        for i in range(pred_num):
            pred_pos = predators[i]
            diff = self_pos - pred_pos
            dist = diff.norm()
            if dist < self.alert_r:      # predator in sight
                weight = ti.exp(-dist / self.alert_r)
                sum_weight += weight
                avg_exit += weight * diff
        out_delta = 0.0
        if sum_weight > 1e-3:
            avg_exit /= sum_weight
            out_delta = 0.8 * valid_angle(ti.atan2(avg_exit[1], avg_exit[0]) - now_ang)
        return out_delta

    @ti.func
    def other_angular_events(self, idx, now_angle, predators = None, pred_num = None):
        angle_vel = 0.0
        cp_w, cp_ang, align_ang, is_valid = self.neighbor_info(idx)
        if is_valid:
            angle_vel += cp_w * valid_angle(cp_ang - now_angle)
            angle_vel += self.align_w * valid_angle(align_ang - now_angle)
        angle_vel += self.avoid_predators(idx, now_angle, predators, pred_num)
        return angle_vel

    @ti.func
    def post_process(self, idx, new_pos):
        id_x, id_y = int(new_pos[0] / self.radius), int(new_pos[1] / self.radius)
        index = ti.atomic_add(self.atom_ptr[id_x, id_y], 0x01)
        self.mask_grids[id_x, id_y, int(index)] = ti.u8(idx)          # store the index of the boid
        self.grid_coord[idx] = vec2s([id_x, id_y])

    def reset_status(self):
        self.sn_handle.deactivate_all()
        self.atom_ptr.fill(0)
        
if __name__ == "__main__":
    ti.init(random_seed = 1)
    W = 1040
    H = 800
    last_one_human = False
    write_video = False

    pigeon_prop = {'boid_num': 64, 'vel_alpha': 0.995, 'ang_alpha': 0.1, 
            'vel_clip': 0.6, 'turn_proba': 0.005, 'vel': 3.0, 
            'border': 50, 'sep_max': 1.0, 'coh_max': 0.4, 'critical_r': 50,
            'align_w': 0.2,'mate_radius': 80, 'alert_r': 120}

    predator_prop = {'boid_num': 4, 'vel_alpha': 0.995, 'ang_alpha': 0.1, 
            'vel_clip': 0.8, 'turn_proba': 0.001, 'vel': 2.9, 
            'border': 50, 'hunt_radius': 400, 'predator_kp': 2.0}

    pigeons   = Pigeons(W, H, pigeon_prop)
    predators = Predators(W, H, predator_prop)

    if last_one_human:
        predators.human_ctrl[-1] = 1

    gui = ti.GUI('Marching Squares', res = (W, H))

    if write_video:
        import tqdm
        frame_rate = 24
        frame_duration = 10          # interger seconds
        frames = frame_rate * frame_duration
        video_manager = ti.tools.VideoManager(output_dir="./outputs/", framerate=25, automatic_build=False)
        pbar = tqdm.tqdm(total = frame_rate * frame_duration)
    while gui.running:
        for e in gui.get_events(gui.PRESS):
            if e.key == gui.ESCAPE:
                gui.running = False
        pigeons.reset_status()
        pigeons.boid_random_vel()
        pigeons.boid_pos_update(predators.pos, predators.boid_num)
        
        cursor_x, cursor_y = gui.get_cursor_pos()
        predators.boid_random_vel()
        predators.prey(pigeons.pos, pigeons.mask_grids, pigeons.atom_ptr, vec2f([cursor_x * W, cursor_y * H]), pigeons.radius)
        predators.boid_pos_update(predators.pos, predators.boid_num)        # parameter here are meaningless (won't be used by predators)

        pg_p1, pg_p2, pg_p3, pg_p3_sym = pigeons.get_triangles()
        pd_p1, pd_p2, pd_p3, pd_p3_sym = predators.get_triangles()
        # Draw Pigeons
        gui.triangles(pg_p1, pg_p2, pg_p3, color = 0xFFFFFF)
        gui.triangles(pg_p1, pg_p2, pg_p3_sym, color = 0xFFFFFF)
        # Draw Predators
        gui.triangles(pd_p1, pd_p2, pd_p3, color = 0xFF0000)
        gui.triangles(pd_p1, pd_p2, pd_p3_sym, color = 0xFF0000)
        if write_video:
            video_manager.write_frame(gui.get_image())
            pbar.update()
            gui.clear()
            if pbar.last_print_n >= frames:
                break
        else:
            gui.show()
    if write_video:
        pbar.close()