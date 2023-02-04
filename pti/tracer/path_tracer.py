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
from taichi.math import vec3

from typing import List
from la.cam_transform import *
from tracer.tracer_base import TracerBase
from emitters.abtract_source import LightSource, TaichiSource

from bsdf.bsdfs import BSDF
from scene.obj_desc import ObjDescriptor
from scene.xml_parser import mitsuba_parsing

from sampler.general_sampling import *

"""
2.5 TODO:
- Refraction / BTDF implementation (according to previous implementation in Rust)
"""

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
        self.anti_alias         = prop['anti_alias']
        self.stratified_sample  = prop['stratified_sampling']   # whether to use stratified sampling
        self.num_shadow_ray     = prop['num_shadow_ray']        # number of shadow samples to trace
        self.use_mis            = prop['use_mis']               # whether to use multiple importance sampling
        assert(self.num_shadow_ray >= 1)
        self.inv_num_shadow_ray = 1. / float(self.num_shadow_ray)

        # for object with attached light source, emitter id stores the reference id to the emitter
        self.emitter_id = ti.field(ti.i32, self.num_objects)   
                     
        self.src_num    = len(emitters)
        self.color      = ti.Vector.field(3, ti.f32, (self.w, self.h))      # color without normalization
        self.src_field  = TaichiSource.field()
        self.bsdf_field = BSDF.field()
        ti.root.dense(ti.i, self.src_num).place(self.src_field)             # Light source Taichi storage
        ti.root.dense(ti.i, self.num_objects).place(self.bsdf_field)        # BSDF Taichi storage

        self.initialze(emitters, objects)

    def initialze(self, emitters: List[LightSource], objects: List[ObjDescriptor]):
        for i, obj in enumerate(objects):
            for j, (mesh, normal) in enumerate(zip(obj.meshes, obj.normals)):
                self.normals[i, j] = ti.Vector(normal) 
                for k, vec in enumerate(mesh):
                    self.meshes[i, j, k]  = ti.Vector(vec)
                if mesh.shape[0] > 2:       # not a sphere
                    self.precom_vec[i, j, 0] = self.meshes[i, j, 1] - self.meshes[i, j, 0]                    
                    self.precom_vec[i, j, 1] = self.meshes[i, j, 2] - self.meshes[i, j, 0]             
            self.mesh_cnt[i]    = obj.tri_num
            self.bsdf_field[i]  = obj.bsdf.export()
            self.aabbs[i, 0]    = ti.Matrix(obj.aabb[0])        # unrolled
            self.aabbs[i, 1]    = ti.Matrix(obj.aabb[1])
            self.emitter_id[i]  = obj.emitter_ref_id
        for i, emitter in enumerate(emitters):
            self.src_field[i] = emitter.export()

    @ti.func
    def sample_light(self, no_sample: ti.i32):
        """
            return selected light source, pdf and whether the current source is valid
            if can only sample <id = no_sample>, then the sampled source is invalid
        """
        idx = ti.random(int) % self.src_num
        pdf = 1. / self.src_num
        valid_sample = True
        if no_sample >= 0:
            if ti.static(self.src_num <= 1):
                valid_sample = False
            else:
                idx = ti.random(int) % (self.src_num - 1)
                if idx >= no_sample: idx += 1
                pdf = 1. / float(self.src_num - 1)
        return self.src_field[idx], pdf, valid_sample

    @ti.kernel
    def render(self):
        self.cnt[None] += 1
        for i, j in self.pixels:
            ray_d = self.pix2ray(i, j)
            ray_o = self.cam_t
            obj_id, normal, min_depth = self.ray_intersect(ray_d, ray_o)
            hit_light       = self.emitter_id[obj_id]   # id for hit emitter, if nothing is hit, this value will be -1
            color           = vec3([0, 0, 0])
            contribution    = vec3([1, 1, 1])
            emission_weight = 1.0
            for _i in range(self.max_bounce):
                if obj_id < 0: break                    # nothing is hit, break
                if ti.static(self.use_rr):
                    # Simple Russian Roullete ray termination
                    max_value = ti.max(contribution)
                    if ti.random(float) > max_value: break
                    else: contribution *= 1. / max_value    # unbiased calculation
                else:
                    if contribution.max() < 1e-4: break     # contribution too small, break
                hit_point   = ray_d * min_depth + ray_o
                hit_light = self.emitter_id[obj_id]

                direct_pdf  = 1.0
                emitter_pdf = 1.0

                break_flag  = False
                emit_int    = vec3([0, 0, 0])
                shadow_int  = vec3([0, 0, 0])
                direct_int  = vec3([0, 0, 0])
                direct_spec = vec3([1, 1, 1])
                for _j in range(self.num_shadow_ray):    # more shadow ray samples
                    emitter, emitter_pdf, emitter_valid = self.sample_light(hit_light)
                    light_dir = vec3([0, 0, 0])
                    # direct / emission component evaluation
                    if emitter_valid:
                        emit_pos, shadow_int, direct_pdf = emitter.         \
                            sample(self.precom_vec, self.normals, self.mesh_cnt, hit_light, hit_point)        # sample light
                        to_emitter  = emit_pos - hit_point
                        emitter_d   = to_emitter.norm()
                        light_dir   = to_emitter / emitter_d
                        direct_spec = self.bsdf_field[obj_id].eval(ray_d, light_dir, normal)
                        if self.does_intersect(light_dir, hit_point, emitter_d):        # shadow ray 
                            shadow_int.fill(0.0)
                    else:       # the only situation for being invalid, is when there is only one source and the ray hit the source
                        break_flag = True
                        break
                    light_pdf = emitter_pdf * direct_pdf
                    if ti.static(self.use_mis):
                        bsdf_pdf = self.bsdf_field[obj_id].get_pdf(light_dir, normal, ray_d)
                        mis_w    = mis_weight(light_pdf, bsdf_pdf)
                        direct_int += direct_spec * shadow_int * mis_w
                    else:
                        direct_int += direct_spec * shadow_int / light_pdf
                if not break_flag:
                    direct_int *= self.inv_num_shadow_ray

                if hit_light >= 0:
                    emit_int = self.src_field[hit_light].eval_le(hit_point - ray_o)
                
                # indirect component requires sampling 
                ray_d, indirect_spec, ray_pdf = self.bsdf_field[obj_id].sample_new_ray(ray_d, normal)
                ray_o = hit_point
                color += (direct_int + emit_int * emission_weight) * contribution
                # VERY IMPORTANT: rendering should be done according to rendering equation (approximation)
                contribution *= indirect_spec / ray_pdf
                obj_id, normal, min_depth = self.ray_intersect(ray_d, ray_o)
                # it turns out that MIS for emitter sampling does not yield a very good result
            self.color[i, j] += color
            self.pixels[i, j] = self.color[i, j] / self.cnt[None]


if __name__ == "__main__":
    profiling = False
    ti.init(arch = ti.vulkan, kernel_profiler = profiling, default_ip = ti.i32, default_fp = ti.f32)
    emitter_configs, _, meshes, configs = mitsuba_parsing("../scene/test/", "test.xml")
    pt = PathTracer(emitter_configs, meshes, configs)
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
    ti.tools.imwrite(pixels, "./outputs/path-tracing.png")
