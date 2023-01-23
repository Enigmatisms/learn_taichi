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

from la.cam_transform import *
from typing import Union, List
from numpy import ndarray as Arr
from taichi import types as ttype
from scene.obj_desc import ObjDescriptor
from emitters.point import PointSource

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
        self.w          = prop['img_size']                              # image is a standard square
        self.sample_cnt = prop['sample_count']
        self.focal      = fov2focal(prop['fov'], self.w)
        self.inv_focal  = 1. / self.focal
        self.half_w     = self.w / 2

        num_objects = len(objects)
        max_tri_num = max([obj.tri_num for obj in objects])
        self.emit_pos = ti.Vector(emitter.pos, dt = ti.f32)                 # currently there should be only one light source
        self.emit_int = ti.Vector(emitter.intensity, dt = ti.f32)       

        self.cam_orient = prop['transform'][0]                              # first field is camera orientation
        self.cam_t      = prop['transform'][1]
        self.cam_r      = ti.Matrix(rotation_between(np.float32([0, 0, 1]), self.cam_orient))
        
        # TODO: attach shininess to every object (BSDF)
        self.aabbs      = ti.Vector.field(3, ti.f32, (num_objects, 2))
        self.normals    = ti.Vector.field(3, ti.f32)
        self.meshes     = ti.Vector.field(3, dtype = ti.f32)               # leveraging SSDS, shape (N, mesh_num, 3) - vector3d
        self.mesh_nodes = ti.root.dense(ti.i, num_objects)
        self.mesh_nodes.bitmasked(ti.j, max_tri_num).dense(ti.k, 3).place(self.meshes)      # for simple shapes, this would be efficient
        self.mesh_nodes.bitmasked(ti.j, max_tri_num).place(self.normals)
        self.mesh_cnt   = ti.field(ti.i32, num_objects)
        self.depth_map  = ti.field(ti.i32, (self.w, self.w))                # gray-scale
        """
            - [] Back-culling should be checked. 
            - [] AABB check to be implemented
            Iterate through all pixels for a rasterizer
        """
        self.pixels = ti.Vector.field(3, ti.f32, (self.w, self.w))
        self.initialze(objects)

    def initialze(self, objects: List[ObjDescriptor]):
        # A problem is that, whether struct-for loop can be used in a non-parallel way?
        for i, obj in enumerate(objects):
            for j, (mesh, normal) in enumerate(zip(obj.meshes, obj.normals)):
                for k in range(3):
                    self.meshes[i, j, k]  = ti.Vector(mesh[k])
                self.normals[i, j] = ti.Vector(normal) 
            self.mesh_cnt[i] = obj.tri_num
            self.aabbs[i, 0] = ti.Matrix(obj.aabb[0])       # unrolled
            self.aabbs[i, 1] = ti.Matrix(obj.aabb[1])

    @ti.func
    def pix2ray(self, i, j):
        """
            Convert pixel coordinate to ray direction
        """
        pi = float(i)
        pj = float(j)
        cam_dir = _Vec3([(pi - self.half_w) * self.inv_focal, (pj - self.half_w) * self.inv_focal, 1.])
        return (self.cam_r @ cam_dir).normalized()

    @ti.kernel
    def render(self):
        for i, j in self.pixels:
            ray = self.pix2ray(i, j)
            # Three steps: (1) AABB test (2) Back-face culling (3) bary-centric depth compute
            obj_id = -1
            tri_id = -1
            min_depth = 1e6
            for aabb_idx in self.aabbs:
                if self.aabb_test(aabb_idx, ray) == False: continue
                tri_num = self.mesh_cnt[aabb_idx]
                for mesh_idx in range(tri_num):
                    normal = self.normals[aabb_idx, mesh_idx]
                    if tm.dot(ray, normal) >= 0.0: continue     # back-face culling
                    # Sadly, Taichi does not support slicing. I think this restrict the use cases of Matrix field
                    p1 = self.meshes[aabb_idx, mesh_idx, 0]
                    vec1 = self.meshes[aabb_idx, mesh_idx, 1] - p1
                    vec2 = self.meshes[aabb_idx, mesh_idx, 2] - p1
                    mat = ti.Matrix.cols([vec1, vec2, -ray]).inverse()
                    u, v, t = mat @ self.cam_t
                    if u > 0 and v > 0 and u + v < 1.0:
                        if t > 0 and t < min_depth:
                            min_depth = t
                            obj_id = aabb_idx
                            tri_id = mesh_idx
            # Iterate through all the meshes and find the minimum depth
            if obj_id < 0:
                self.depth_map[i, j] = 0.0
                self.pixels[i, j].fill(0)
            else:
                self.depth_map = min_depth
                # Calculate Blinn-Phong lighting model
                normal = self.normals[obj_id, tri_id]
                hit_point = ray * min_depth + self.cam_t
                light_dir = (self.emit_pos - hit_point).normalized()
                # light_dir and ray are normalized, ray points from cam to hit point
                # the ray direction vector in half way vector should point from hit point to cam
                half_way = (0.5 * (light_dir - ray)).normalized()
                spec = tm.pow(ti.max(tm.dot(half_way, normal), 0.0), 1.0)

                # Iterate through all the triangles in one object
    
    @ti.func
    def aabb_test(self, aabb_idx, ray: _Vec3):
        return True

if __name__ == "__main__":
    ti.init()
    # to be implemented
