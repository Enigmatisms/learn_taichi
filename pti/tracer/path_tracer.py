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
from tracer.tracer_base import TracerBase
from emitters.abtract_source import LightSource, TaichiSource

from scene.obj_desc import ObjDescriptor
from scene.xml_parser import mitsuba_parsing

@ti.data_oriented
class PathTracer(TracerBase):
    """
        Simple Ray tracing using Bary-centric coordinates
        This tracer can yield result with global illumination effect
    """
    def __init__(self, emitters: List[LightSource], objects: List[ObjDescriptor], prop: dict):
        super().__init__(objects, prop)
        """
            Implement path tracing algorithms first, then we can improve light source / BSDF / participating media
        """
        self.src_num    = len(emitters)
        self.pdf_sum    = ti.field(ti.f32, (self.w, self.h))               # progressive update
        self.shininess  = ti.field(ti.f32, self.num_objects)
        self.src_field  = TaichiSource.field()
        ti.root.dense(ti.i, self.src_num).place(self.src_field)            # Light source Taichi storage

        self.initialze(emitters, objects)

    def initialze(self, emitters: List[LightSource], objects: List[ObjDescriptor]):
        for i, obj in enumerate(objects):
            for j, (mesh, normal) in enumerate(zip(obj.meshes, obj.normals)):
                for k in range(3):
                    self.meshes[i, j, k]  = ti.Vector(mesh[k])
                self.normals[i, j] = ti.Vector(normal) 
            self.mesh_cnt[i]    = obj.tri_num
            self.shininess[i]   = obj.bsdf.shininess
            self.surf_color[i]  = ti.Vector(obj.bsdf.reflectance)
            self.aabbs[i, 0]    = ti.Matrix(obj.aabb[0])       # unrolled
            self.aabbs[i, 1]    = ti.Matrix(obj.aabb[1])
        for i, emitter in enumerate(emitters):
            self.src_field[i] = emitter.export()

    @ti.func
    def sample_light(self):
        """
            return selected light source and pdf
        """
        idx = ti.random(int) % self.src_num
        return self.src_field[idx], 1. / self.src_num

    @ti.kernel
    def render(self):
        for i, j in self.pixels:
            ray_d = self.pix2ray(i, j)
            ray_o = self.cam_t
            obj_id, tri_id, min_depth = self.ray_intersect(ray_d, ray_o)
            color = vec3([0, 0, 0])
            pdf = 1.0
            contribution = vec3([1, 1, 1])
            for _ in range(self.max_bounce):
                if contribution.max() < 5e-4: break     # contribution too small, break
                if obj_id < 0: break                    # nothing is hit, break
                normal = self.normals[obj_id, tri_id]
                hit_point  = ray_d * min_depth + ray_o
                emitter, emitter_pdf = self.sample_light()
                emit_pos, emit_int, emit_pdf = emitter.sample(hit_point)        # sample light
                to_emitter = emit_pos - hit_point
                emitter_d  = to_emitter.norm()
                light_dir  = to_emitter / emitter_d
                half_way = (0.5 * (light_dir - ray_d)).normalized()
                spec = tm.pow(ti.max(tm.dot(half_way, normal), 0.0), self.shininess[obj_id])
                # shadow ray? 
                if self.does_intersect(light_dir, hit_point, emitter_d):
                    emit_int.fill(0.0)
                color += spec * emit_int * self.surf_color[obj_id] * contribution
                contribution *= 1 - spec        # <light reflected> + <light transmitted> = 1.0
                pdf *= (emit_pdf * emitter_pdf)

                """
                    TODO: recompute ray dir and calculate new intersection point
                """
                

if __name__ == "__main__":
    profiling = False
    ti.init(kernel_profiler = profiling, default_ip = ti.i32, default_fp = ti.f32)
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
        bpt.render(emitter_pos)
        gui.set_image(bpt.pixels)
        gui.show()
        if gui.running == False: break
        gui.clear()
        bpt.reset()

    if profiling:
        ti.profiler.print_kernel_profiler_info() 
    ti.tools.imwrite(bpt.pixels.to_numpy(), "./blinn-phong.png")
