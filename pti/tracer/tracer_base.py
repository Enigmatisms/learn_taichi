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

from scene.obj_desc import ObjDescriptor
from scene.xml_parser import mitsuba_parsing

__eps__ = 1e-4
__inv_eps__ = 1 - __eps__ * 2.

@ti.data_oriented
class TracerBase:
    """
        Simple Ray tracing using Bary-centric coordinates
        This tracer can yield result with global illumination effect
    """
    def __init__(self, objects: List[ObjDescriptor], prop: dict):
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

        self.cam_orient = prop['transform'][0]                          # first field is camera orientation
        self.cam_orient /= np.linalg.norm(self.cam_orient)
        self.cam_t      = ti.Vector(prop['transform'][1])
        self.cam_r      = ti.Matrix(np_rotation_between(np.float32([0, 0, 1]), self.cam_orient))
        
        self.aabbs      = ti.Vector.field(3, ti.f32, (self.num_objects, 2))
        self.normals    = ti.Vector.field(3, ti.f32)
        self.meshes     = ti.Vector.field(3, ti.f32)                    # leveraging SSDS, shape (N, mesh_num, 3) - vector3d
        self.precom_vec = ti.Vector.field(3, ti.f32)
        self.pixels = ti.Vector.field(3, ti.f32, (self.w, self.h))      # output: color

        self.bitmasked_nodes = ti.root.dense(ti.i, self.num_objects).bitmasked(ti.j, max_tri_num)
        self.bitmasked_nodes.place(self.normals)
        self.bitmasked_nodes.dense(ti.k, 3).place(self.meshes)      # for simple shapes, this would be efficient
        # triangle has 3 vertices, v1, v2, v3. precom_vec stores (v2 - v1), (v3 - v1)
        # These two precom(puted) vectors can be used in ray intersection and triangle sampling (for shape-attached emitters)
        self.bitmasked_nodes.dense(ti.k, 2).place(self.precom_vec)

        self.mesh_cnt   = ti.field(ti.i32, self.num_objects)
        self.cnt        = ti.field(ti.i32, ())                          # useful in path tracer (sample counter)

    def __repr__(self):
        """
            For debug purpose
        """
        return f"tracer_base: number of object {self.num_objects}, w, h: ({self.w}, {self.h}), sample count: {self.sample_cnt}. Focal: {self.focal}"

    def initialze(self, _objects: List[ObjDescriptor]):
        pass

    @ti.func
    def pix2ray(self, i, j):
        """
            Convert pixel coordinate to ray direction
            - anti_alias: whether to use pixel sample jittering for anti-aliasing
            - str_sample: whether to use stratified sampling 
        """
        pi = float(i)
        pj = float(j)
        vx = 0.5
        vy = 0.5
        if ti.static(self.anti_alias):
            if ti.static(self.stratified_sample): # sequential stratified sampling
                mod_val = self.cnt[None] % 16
                vx = float(mod_val % 4)   * 0.25 + ti.random(float) * 0.25
                vy = float(mod_val // 4) * 0.25 + ti.random(float) * 0.25
            else:    # uniform sampling
                vx = ti.random(float) * __inv_eps__ + __eps__
                vy = ti.random(float) * __inv_eps__ + __eps__
        cam_dir = vec3([(self.half_w + vx - pi) * self.inv_focal, (pj - self.half_h + vy) * self.inv_focal, 1.])
        return (self.cam_r @ cam_dir).normalized()

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
                vec1 = self.precom_vec[aabb_idx, mesh_idx, 0]
                vec2 = self.precom_vec[aabb_idx, mesh_idx, 1]
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
                p1 = self.meshes[aabb_idx, mesh_idx, 0]
                vec1 = self.precom_vec[aabb_idx, mesh_idx, 0]
                vec2 = self.precom_vec[aabb_idx, mesh_idx, 1]
                mat = ti.Matrix.cols([vec1, vec2, -ray]).inverse()
                u, v, t = mat @ (start_p - p1)
                if u >= 0 and v >= 0 and u + v <= 1.0:
                    if t > 0 and t < depth:
                        flag = True
                        break
            if flag == True: break
        return flag

    @ti.kernel
    def render(self):
        pass

    @ti.kernel
    def reset(self):
        for i, j in self.pixels:
            self.pixels[i, j].fill(0.0)
    
if __name__ == "__main__":
    ti.init()
    _, _, meshes, configs = mitsuba_parsing("../scene/test/", "test.xml")
    base = TracerBase(meshes, configs)
