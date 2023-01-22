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

from emitters.point import PointSource

@ti.data_oriented
class PhongRasterizer:
    """
        Rasterizer using Bary-centric coordinates
        origin + direction * t = u * PA(vec) + v * PB(vec)
        This is a rank-3 linear equation
    """
    def __init__(self, emitter: PointSource, objects: list):
        # This should be extended
        num_objects = len(objects)
        max_tri_num = 16
        self.emit_pos = ti.Vector(emitter.pos, dt = ti.f32)             # currently there should be only one light source
        self.emit_int = ti.Vector(emitter.intensity, dt = ti.f32)       
        
        self.meshes = ti.Matrix.field(3, 3, dtype = ti.f32)
        self.aabbs  = ti.Matrix.field(2, 3, dtype = ti.f32)
        self.mesh_nodes = ti.root.dense(ti.i, num_objects)
        self.mesh_nodes.bitmasked(ti.j, max_tri_num).place(self.meshes)                    # for simple shapes, this would be efficient
        """
            TODO: load meshes (triangles) / normals and emitters:
            - Loading meshes: load to bitmasked (since it is spatially sparse), node that
                - meshes stored in Taichi is of shape (N objects, N triangles, 3, 3)
                - Therefore, we can cull object first, using (AABB) (since inner-for-loops are not parallel)
                - Then back culling triangles, and solve u, v, t finally.
            - Implement Blinn-Phong model
            Iterate through all pixels for a rasterizer
        """


if __name__ == "__main__":
    ti.init()
    # to be implemented

        
