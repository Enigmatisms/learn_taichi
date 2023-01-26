"""
    Path tracer for indirect / global illumination
    This module will be progressively built. Currently, participating media is not supported
    @author: Qianyue He
    @date: 2023.1.26
"""

import sys
sys.path.append("..")

import numpy as np
import taichi as ti
import taichi.math as tm
from taichi.math import vec3

from typing import List
from la.cam_transform import *
from emitters.point import PointSource
from emitters.rect_area import RectAreaSource

from scene.obj_desc import ObjDescriptor
from scene.xml_parser import mitsuba_parsing

@ti.data_oriented
class PathTracer:
    """
        Simple Ray tracing using Bary-centric coordinates
        This tracer can yield result with global illumination effect
    """
    def __init__(self, emitter: PointSource, objects: List[ObjDescriptor], prop: dict):
        self.w          = prop['film']['width']                              # image is a standard square
        self.h          = prop['film']['height']
        self.sample_cnt = prop['sample_count']
        self.max_bounce = prop['max_bounce']
        self.use_rr     = prop['use_rr']

        self.focal      = fov2focal(prop['fov'], min(self.w, self.h))
        self.inv_focal  = 1. / self.focal
        self.half_w     = self.w / 2
        self.half_h     = self.h / 2

        self.num_objects = len(objects)
        max_tri_num = max([obj.tri_num for obj in objects])
        self.emit_int = ti.Vector(emitter.intensity, dt = ti.f32)       

        self.cam_orient = prop['transform'][0]                              # first field is camera orientation
        self.cam_orient /= np.linalg.norm(self.cam_orient)
        self.cam_t      = ti.Vector(prop['transform'][1])
        self.cam_r      = ti.Matrix(rotation_between(np.float32([0, 0, 1]), self.cam_orient))
        
        self.aabbs      = ti.Vector.field(3, ti.f32, (self.num_objects, 2))
        self.normals    = ti.Vector.field(3, ti.f32)
        self.meshes     = ti.Vector.field(3, dtype = ti.f32)               # leveraging SSDS, shape (N, mesh_num, 3) - vector3d
        self.surf_color = ti.Vector.field(3, ti.f32, self.num_objects)
        self.pixels = ti.Vector.field(3, ti.f32, (self.w, self.h))         # output: color

        self.pdf_sum    = ti.field(ti.f32, (self.w, self.h))               # progressive update
        self.mesh_nodes = ti.root.dense(ti.i, self.num_objects)
        self.mesh_nodes.bitmasked(ti.j, max_tri_num).dense(ti.k, 3).place(self.meshes)      # for simple shapes, this would be efficient
        self.mesh_nodes.bitmasked(ti.j, max_tri_num).place(self.normals)
        self.mesh_cnt   = ti.field(ti.i32, self.num_objects)
        self.shininess  = ti.field(ti.f32, self.num_objects)

        self.initialze(objects)

    def __repr__(self):
        """
            For debug purpose
        """
        return f"bpt: number of object {self.num_objects}, w, h: ({self.w}, {self.h}), sample count: {self.sample_cnt}. Focal: {self.focal}"

    def initialze(self, objects: List[ObjDescriptor]):
        for i, obj in enumerate(objects):
            for j, (mesh, normal) in enumerate(zip(obj.meshes, obj.normals)):
                for k in range(3):
                    self.meshes[i, j, k]  = ti.Vector(mesh[k])
                self.normals[i, j] = ti.Vector(normal) 
            self.mesh_cnt[i]    = obj.tri_num
            self.shininess[i]   = obj.bsdf.shininess
            self.aabbs[i, 0]    = ti.Matrix(obj.aabb[0])       # unrolled
            self.aabbs[i, 1]    = ti.Matrix(obj.aabb[1])
            self.surf_color[i]  = ti.Vector(obj.bsdf.reflectance)

    @ti.func
    def pix2ray(self, i, j):
        """
            Convert pixel coordinate to ray direction
        """
        pi = float(i)
        pj = float(j)
        cam_dir = vec3([(pi - self.half_w + 0.5) * self.inv_focal, (pj - self.half_h + 0.5) * self.inv_focal, 1.])
        return (self.cam_r @ cam_dir).normalized()

    @ti.func
    def ray_intersect(self, ray, start_p, min_depth = -1.0):
        """
            Intersection function and mesh organization can be reused
        """
        obj_id = -1
        tri_id = -1
        if min_depth > 0.0:
            start_p += ray * 1e-3
            min_depth -= 2e-3
        else:
            min_depth = 1e7
        for aabb_idx in range(self.num_objects):
            if self.aabb_test(aabb_idx, ray, start_p) == False: continue
            tri_num = self.mesh_cnt[aabb_idx]
            for mesh_idx in range(tri_num):
                normal = self.normals[aabb_idx, mesh_idx]
                if tm.dot(ray, normal) >= 0.0: continue     # back-face culling
                # Sadly, Taichi does not support slicing. I think this restrict the use cases of Matrix field
                p1 = self.meshes[aabb_idx, mesh_idx, 0]
                vec1 = self.meshes[aabb_idx, mesh_idx, 1] - p1
                vec2 = self.meshes[aabb_idx, mesh_idx, 2] - p1
                mat = ti.Matrix.cols([vec1, vec2, -ray]).inverse()
                u, v, t = mat @ (start_p - p1)
                if u >= 0 and v >= 0 and u + v <= 1.0:
                    if t > 0 and t < min_depth:
                        min_depth = t
                        obj_id = aabb_idx
                        tri_id = mesh_idx
        return (obj_id, tri_id, min_depth)

    @ti.func
    def does_intersect(self, ray, start_p, depth = -1.0) -> bool:
        """
            Faster (greedy) checking for intersection, returns True if intersect anything within depth \\
            If depth is None (not specified), then depth range will be a large float (1e7) \\
            Taichi does not support compile-time branching. Actually it does, but not flexible, for e.g \\
            C++ supports compile-time branching via template parameter, but Taichi can not "pass" compile-time constants
        """
        if depth > 0.0:
            start_p += ray * 1e-3
            depth -= 2e-3
        else:
            depth = 1e7
        flag = False
        for aabb_idx in range(self.num_objects):
            if self.aabb_test(aabb_idx, ray, start_p) == False: continue
            tri_num = self.mesh_cnt[aabb_idx]
            for mesh_idx in range(tri_num):
                normal = self.normals[aabb_idx, mesh_idx]
                if tm.dot(ray, normal) >= 0.0: continue     # back-face culling
                p1 = self.meshes[aabb_idx, mesh_idx, 0]
                vec1 = self.meshes[aabb_idx, mesh_idx, 1] - p1
                vec2 = self.meshes[aabb_idx, mesh_idx, 2] - p1
                mat = ti.Matrix.cols([vec1, vec2, -ray]).inverse()
                u, v, t = mat @ (start_p - p1)
                if u >= 0 and v >= 0 and u + v <= 1.0:
                    if t > 0 and t < depth:
                        flag = True
                        break
            if flag == True: break
        return flag

    @ti.kernel
    def render(self, emit_pos: vec3):
        for i, j in self.pixels:
            ray = self.pix2ray(i, j)
            obj_id, tri_id, min_depth = self.ray_intersect(ray, self.cam_t)
            for bounce in range(self.max_bounce):
                """
                    To be implemented
                """
                pass

    @ti.kernel
    def reset(self):
        for i, j in self.pixels:
            self.pixels[i, j].fill(0.0)
    
    @ti.func
    def aabb_test(self, aabb_idx, ray: vec3, ray_o: vec3):
        """
            AABB used to skip some of the objects
        """
        t_min = (self.aabbs[aabb_idx, 0] - ray_o) / ray
        t_max = (self.aabbs[aabb_idx, 1] - ray_o) / ray
        t1 = ti.min(t_min, t_max)
        t2 = ti.max(t_min, t_max)
        t_near  = ti.max(ti.max(t1.x, t1.y), t1.z)
        t_far   = ti.min(ti.min(t2.x, t2.y), t2.z)
        return t_near < t_far

if __name__ == "__main__":
    profiling = False
    ti.init(kernel_profiler = profiling)
    emitter_configs, _, meshes, configs = mitsuba_parsing("../scene/test/", "test.xml")
    emitter = emitter_configs[0]
    emitter_pos = vec3(emitter.pos)
    bpt = PathTracer(emitter, meshes, configs)
    # Note that direct test the rendering time (once) is meaningless, executing for the first time
    # will be accompanied by JIT compiling, compilation time will be included.
    gui = ti.GUI('BPT', (bpt.w, bpt.h))
    while gui.running:
        for e in gui.get_events(gui.PRESS):
            if e.key == gui.ESCAPE:
                gui.running = False
            elif e.key == 'a':
                emitter_pos[0] -= 0.05
            elif e.key == 'd':
                emitter_pos[0] += 0.05
            elif e.key == gui.DOWN:
                emitter_pos[1] -= 0.05
            elif e.key == gui.UP:
                emitter_pos[1] += 0.05
            elif e.key == 'd':
                emitter_pos[0] += 0.05
            elif e.key == 's':
                emitter_pos[2] -= 0.05
            elif e.key == 'w':
                emitter_pos[2] += 0.05
        bpt.render(emitter_pos)
        gui.set_image(bpt.pixels)
        gui.show()
        if gui.running == False: break
        gui.clear()
        bpt.reset()

    if profiling:
        ti.profiler.print_kernel_profiler_info() 
    ti.tools.imwrite(bpt.pixels.to_numpy(), "./blinn-phong.png")
