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

from sampler.general_sampling import *

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
        self.anti_alias = prop['anti_alias']

        self.cnt        = ti.field(ti.i32, ())
        self.src_num    = len(emitters)
        self.pdf_sum    = ti.field(ti.f32, (self.w, self.h))                # progressive update
        self.shininess  = ti.field(ti.f32, self.num_objects)
        self.color      = ti.Vector.field(3, ti.f32, (self.w, self.h))      # color without normalization
        self.src_field  = TaichiSource.field()
        ti.root.dense(ti.i, self.src_num).place(self.src_field)             # Light source Taichi storage

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

    @ti.func
    def sample_ray_dir(self, ctr_dir: vec3, obj_id: int):
        """

            Sample according to bsdf type. This is a simple implementation. First we just \\
            consider using shininess to determine sampling function. \\
            - ctr_dir: new direction vector is sampled around this vector
            - FIXME: BSDF should be formulated like emitters, declaring their types and parameters
        """
        shininess = self.shininess[obj_id]
        local_new_dir = vec3([0, 0, 0])
        pdf = 0.0
        # This part should be upgraded to support more (and sophisticated) BSDF
        if shininess > 1.5:
            local_new_dir, pdf = ramped_hemisphere()
        else:
            local_new_dir, pdf = uniform_hemisphere()
        R = rotation_between(vec3([0, 1, 0]), ctr_dir)
        return R @ local_new_dir, pdf

    @ti.kernel
    def render(self):
        self.cnt[None] += 1
        for i, j in self.pixels:
            ray_d = self.pix2ray(i, j, self.anti_alias)
            ray_o = self.cam_t
            obj_id, tri_id, min_depth = self.ray_intersect(ray_d, ray_o)
            color = vec3([0, 0, 0])
            contribution = vec3([1, 1, 1])
            bounce_cnt = 0
            for _ in range(self.max_bounce):
                bounce_cnt += 1
                if obj_id < 0: break                    # nothing is hit, break
                if contribution.max() < 1e-5: break     # contribution too small, break
                normal = self.normals[obj_id, tri_id]
                hit_point  = ray_d * min_depth + ray_o
                emitter, emitter_pdf = self.sample_light()
                emit_pos, emit_int, emit_pdf = emitter.sample(hit_point)        # sample light

                to_emitter = emit_pos - hit_point
                emitter_d  = to_emitter.norm()
                light_dir  = to_emitter / emitter_d
                half_way = (0.5 * (light_dir - ray_d)).normalized()
                shininess = self.shininess[obj_id]
                spec = tm.pow(ti.max(tm.dot(half_way, normal), 0.0), shininess)
                contrib_att = tm.pow(ti.max(tm.dot(-ray_d, normal), 0.0), shininess)
                ray_o = hit_point
                ray_d, ray_pdf = self.sample_ray_dir(normal, obj_id)
                # shadow ray: intersect means the light source is shadowed 
                if self.does_intersect(light_dir, hit_point, emitter_d):
                    emit_int.fill(0.0)
                contribution *= self.surf_color[obj_id]       # <light reflected> + <light transmitted> = 1.0
                color += spec * emit_int * contribution / (emitter_pdf * emit_pdf)
                contribution *= contrib_att

                obj_id, tri_id, min_depth = self.ray_intersect(ray_d, ray_o)
            self.color[i, j] += color / float(bounce_cnt)                                # PDF could be really small
            # self.pdf_sum[i, j] += 1. / pdf
            if i == 256 and j == 256:
                print(color, self.color[i, j], self.pdf_sum[i, j], self.cnt[None])
            self.pixels[i, j] = self.color[i, j] / self.cnt[None] # TODO: is this true?


if __name__ == "__main__":
    profiling = False
    ti.init(kernel_profiler = profiling, default_ip = ti.i32, default_fp = ti.f32)
    emitter_configs, _, meshes, configs = mitsuba_parsing("../scene/test/", "test.xml")
    pt = PathTracer(emitter_configs, meshes, configs)
    # Note that direct test the rendering time (once) is meaningless, executing for the first time
    # will be accompanied by JIT compiling, compilation time will be included.
    gui = ti.GUI('Path Tracing', (pt.w, pt.h))
    while gui.running:
        for e in gui.get_events(gui.PRESS):
            if e.key == gui.ESCAPE:
                gui.running = False
        pt.render()
        gui.set_image(pt.pixels)
        gui.show()
        if gui.running == False: break
        gui.clear()
        pt.reset()

    if profiling:
        ti.profiler.print_kernel_profiler_info() 
    pixels = pt.pixels.to_numpy()
    ti.tools.imwrite(pixels, "./path-tracing.png")
