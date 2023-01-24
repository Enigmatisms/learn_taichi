"""
    Rasterizer for direct lighting given a point source
    Blinn-Phong model
    @author: Qianyue He
    @date: 2023.1.22
"""

import sys
sys.path.append("..")

import numpy as np
import taichi as ti
import taichi.math as tm

from typing import List
from la.cam_transform import *
from taichi import types as ttype
from emitters.point import PointSource

from scene.obj_desc import ObjDescriptor
from scene.xml_parser import mitsuba_parsing

_Vec3 = ttype.vector(3, ti.f32)

@ti.data_oriented
class BlinnPhongRasterizer:
    """
        Rasterizer using Bary-centric coordinates
        origin + direction * t = u * PA(vec) + v * PB(vec)
        This is a rank-3 linear equation
    """
    def __init__(self, emitter: PointSource, objects: List[ObjDescriptor], prop: dict):
        # This should be extended
        self.w          = prop['film'][0]                              # image is a standard square
        self.sample_cnt = prop['sample_count']
        self.focal      = fov2focal(prop['fov'], self.w)
        self.inv_focal  = 1. / self.focal
        self.half_w     = self.w / 2

        self.num_objects = len(objects)
        max_tri_num = max([obj.tri_num for obj in objects])
        self.emit_pos = ti.Vector(emitter.pos, dt = ti.f32)                 # currently there should be only one light source
        self.emit_int = ti.Vector(emitter.intensity, dt = ti.f32)       

        self.cam_orient = prop['transform'][0]                              # first field is camera orientation
        self.cam_orient /= np.linalg.norm(self.cam_orient)
        self.cam_t      = ti.Vector(prop['transform'][1])
        self.cam_r      = ti.Matrix(rotation_between(np.float32([0, 0, 1]), self.cam_orient))
        
        self.aabbs      = ti.Vector.field(3, ti.f32, (self.num_objects, 2))
        self.normals    = ti.Vector.field(3, ti.f32)
        self.meshes     = ti.Vector.field(3, dtype = ti.f32)               # leveraging SSDS, shape (N, mesh_num, 3) - vector3d
        self.mesh_nodes = ti.root.dense(ti.i, self.num_objects)
        self.mesh_nodes.bitmasked(ti.j, max_tri_num).dense(ti.k, 3).place(self.meshes)      # for simple shapes, this would be efficient
        self.mesh_nodes.bitmasked(ti.j, max_tri_num).place(self.normals)
        self.shininess  = ti.field(ti.f32, self.num_objects)
        self.mesh_cnt   = ti.field(ti.i32, self.num_objects)
        self.depth_map  = ti.field(ti.f32, (self.w, self.w))                # gray-scale
        """
            - [] AABB check to be implemented
            Iterate through all pixels for a rasterizer
        """
        self.pixels = ti.Vector.field(3, ti.f32, (self.w, self.w))
        self.initialze(objects)

    def __repr__(self):
        """
            For debug purpose
        """
        return f"BPR: number of object {self.num_objects}, width: {self.w}, sample count: {self.sample_cnt}. Focal: {self.focal}"

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

    @ti.func
    def pix2ray(self, i, j):
        """
            Convert pixel coordinate to ray direction
        """
        pi = float(i)
        pj = float(j)
        cam_dir = _Vec3([(pi - self.half_w) * self.inv_focal, (pj - self.half_w) * self.inv_focal, 1.])
        return (self.cam_r @ cam_dir).normalized()

    @ti.func
    def ray_intersect(self, ray, start_p, min_depth = -1.0):
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

    @ti.func
    def distance_attenuate(self, x):
        return ti.min(1.0 / (1e-5 + x ** 2), 1e5)

    @ti.kernel
    def render(self):
        for i, j in self.pixels:
            ray = self.pix2ray(i, j)
            obj_id, tri_id, min_depth = self.ray_intersect(ray, self.cam_t)
            # Iterate through all the meshes and find the minimum depth
            if obj_id >= 0:
                self.depth_map[i, j] = min_depth
                # Calculate Blinn-Phong lighting model
                normal = self.normals[obj_id, tri_id]
                hit_point  = ray * min_depth + self.cam_t
                to_emitter = self.emit_pos - hit_point
                emitter_d  = to_emitter.norm()
                light_dir  = to_emitter / emitter_d
                # light_dir and ray are normalized, ray points from cam to hit point
                # the ray direction vector in half way vector should point from hit point to cam
                half_way = (0.5 * (light_dir - ray)).normalized()
                spec = tm.pow(ti.max(tm.dot(half_way, normal), 0.0), self.shininess[obj_id])
                spec *= self.distance_attenuate(emitter_d)
                if self.does_intersect(light_dir, hit_point, emitter_d):
                    spec *= 0.1
                self.pixels[i, j] = spec * self.emit_int
    
    @ti.func
    def aabb_test(self, aabb_idx, ray: _Vec3, ray_o: _Vec3):
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
    profiling = True
    ti.init(kernel_profiler = profiling)
    emitter_configs, _, meshes, configs = mitsuba_parsing("../scene/test/", "test.xml")
    bpr = BlinnPhongRasterizer(emitter_configs[0], meshes, configs)
    # Note that direct test the rendering time (once) is meaningless, executing for the first time
    # will be accompanied by JIT compiling, compilation time will be included.
    bpr.render()
    if profiling:
        ti.profiler.print_kernel_profiler_info() 
    ti.tools.imwrite(bpr.pixels.to_numpy(), "./blinn-phong.png")

    depth_map = bpr.depth_map.to_numpy()
    depth_map /= depth_map.max()
    ti.tools.imwrite(depth_map, "./depth.png")
