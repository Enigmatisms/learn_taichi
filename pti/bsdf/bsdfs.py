"""
    All the bsdfs are here, note that only three kinds of simple BSDF are supported
    Diffusive / Glossy / Pure specular (then it will be participating media)
    @author: Qianyue He
    @date: 2023-1-23
"""
import sys
sys.path.append("..")

import numpy as np
import taichi as ti
import taichi.math as tm
import xml.etree.ElementTree as xet
from taichi.math import vec3

from la.cam_transform import *
from sampler.general_sampling import *
from scene.general_parser import rgb_parse

__all__ = ['BSDF_np', 'BSDF']


INV_PI = 1. / tm.pi

class BSDF_np:
    """
        BSDF base-class, 
        @author: Qianyue He
        @date: 2023-1-23
    """

    __all_albedo_name       = {"reflectance", "albedo", "k_d"}
    __all_glossiness_name   = {"glossiness", "shininess", "k_g"}
    __all_specular_name     = {"specular", "k_s"}
    __all_absorption_name   = {"absorptions", "k_a"}
    __type_mapping          = {"blinn-phong": 0, "lambertian": 1, "specular": 2}
    
    def __init__(self, elem: xet.Element):
        self.type: str = elem.get("type")
        if self.type not in BSDF_np.__type_mapping:
            raise NotImplementedError(f"Unknown BSDF type: {self.type}")
        self.type_id = BSDF_np.__type_mapping[self.type]
        self.id: str = elem.get("id")
        self.k_d = np.ones(3, np.float32)
        self.k_s = np.zeros(3, np.float32)
        self.k_g = np.ones(3, np.float32)
        self.k_a = np.zeros(3, np.float32)
        self.kd_default = True
        self.ks_default = True
        self.kg_default = True
        self.ka_default = True

        rgb_nodes = elem.findall("rgb")
        for rgb_node in rgb_nodes:
            name = rgb_node.get("name")
            if name is None: raise ValueError(f"RGB node in Blinn-phong BSDF <{elem.get('id')}> has empty name.")
            if name in BSDF_np.__all_albedo_name:
                self.k_d = rgb_parse(rgb_node)
                self.kd_default = False
            elif name in BSDF_np.__all_specular_name:
                self.k_s = rgb_parse(rgb_node)
                self.ks_default = False
            elif name in BSDF_np.__all_glossiness_name:
                self.k_g = rgb_parse(rgb_node)
                self.kg_default = False
            elif name in BSDF_np.__all_absorption_name:
                self.k_a = rgb_parse(rgb_node)
                self.ka_default = False

    def export(self):
        return BSDF(_type = self.type_id, k_d = vec3(self.k_d), k_s = vec3(self.k_s), k_g = vec3(self.k_g), k_a = vec3(self.k_a))
    
    def __repr__(self) -> str:
        return f"<{self.type.capitalize()} BSDF, default:[{int(self.kd_default), int(self.ks_default), int(self.kg_default), int(self.ka_default)}]>"


@ti.dataclass
class BSDF:
    """
        Taichi exported struct for unified BSDF storage
        FIXME: add a flag to indicate whether the BSDF is Dirac delta (for MIS)
    """
    _type:      ti.i32
    k_d:        vec3            # diffusive coefficient (albedo)
    k_s:        vec3            # specular coefficient
    k_g:        vec3            # glossiness coefficient
    k_a:        vec3            # absorption coefficient

    @ti.func
    def delocalize_rotate(self, anchor: vec3, local_dir: vec3):
        R = rotation_between(vec3([0, 1, 0]), anchor)
        return (R @ local_dir).normalized()
    
    # ======================= Blinn-Phong ========================
    @ti.func
    def eval_blinn_phong(self, ray_in: vec3, ray_out: vec3, normal: vec3):
        """
            Normally, ray in is along the opposite direction of normal
            Attention: in backward tracing, ray_in is actually out-going direction
            therefore, cosine term is related to ray_out
        """
        half_way = (ray_out - ray_in).normalized()
        dot_clamp = ti.max(0.0, tm.dot(half_way, normal))
        glossy = tm.pow(dot_clamp, self.k_g)
        cosine_term = tm.max(0.0, tm.dot(normal, ray_out))
        # When k_s for Blinn-phong is 0., Blinn-Phong degenerates to Lambertian
        return (self.k_d + self.k_s * (0.5 * (self.k_g + 2.0) * glossy)) * INV_PI * cosine_term

    @ti.func
    def sample_blinn_phong(self, incid: vec3, normal: vec3):
        local_new_dir, pdf = cosine_hemisphere()
        ray_out_d = self.delocalize_rotate(normal, local_new_dir)
        spec = self.eval_blinn_phong(incid, ray_out_d, normal)
        return ray_out_d, spec, pdf
    
    # ======================= Lambertian ========================
    @ti.func
    def eval_lambertian(self, ray_out: vec3, normal: vec3):
        cosine_term = tm.max(0.0, tm.dot(normal, ray_out))
        return self.k_d * 0.5 * INV_PI * cosine_term

    @ti.func
    def sample_lambertian(self, normal: vec3):
        local_new_dir, pdf = cosine_hemisphere()
        ray_out_d = self.delocalize_rotate(normal, local_new_dir)
        spec = self.eval_lambertian(ray_out_d, normal)
        return ray_out_d, spec, pdf

    # ======================= Mirror-Specular ========================
    @ti.func
    def eval_specular(self, incid: vec3, ray_out: vec3, normal: vec3):
        return vec3([1, 1, 1])

    @ti.func
    def sample_specular(self, incid: vec3, normal: vec3):
        return vec3([0, 1, 0]), vec3([1, 1, 1]), 1.0 

    # ================================================================

    @ti.func
    def eval(self, incid: vec3, out: vec3, normal: vec3) -> vec3:
        ret_spec = vec3([1, 1, 1])
        if self._type == 0:         # Blinn-Phong
            ret_spec = self.eval_blinn_phong(incid, out, normal)
        elif self._type == 1:       # Lambertian
            ret_spec = self.eval_lambertian(out, normal)
        elif self._type == 2:       # Specular
            ret_spec = self.eval_specular(incid, out, normal)
        else:
            print(f"Warnning: unknown or unsupported BSDF type: {self._type} during evaluation.")
        return ret_spec

    @ti.func
    def sample_new_ray(self, incid: vec3, normal: vec3):
        """
            All the sampling function will return: (1) new ray (direction) \\
            (2) rendering equation transfer term (BSDF * cos term) (3) PDF
        """
        ret_dir  = vec3([0, 1, 0])
        ret_spec = vec3([1, 1, 1])
        pdf      = 1.0
        if self._type == 0:         # Blinn-Phong
            ret_dir, ret_spec, pdf = self.sample_blinn_phong(incid, normal)
        elif self._type == 1:       # Lambertian
            ret_dir, ret_spec, pdf = self.sample_lambertian(normal)
        elif self._type == 2:       # Specular
            ret_dir, ret_spec, pdf = self.sample_specular(incid, normal)
        else:
            print(f"Warnning: unknown or unsupported BSDF type: {self._type} during evaluation.")
        return ret_dir, ret_spec, pdf

    @ti.func
    def get_pdf(self, outdir: vec3, normal: vec3, incid: vec3):
        """ 
            Solid angle PDF for a specific incident direction - BSDF sampling
            Some PDF has nothing to do with backward incid (from eye to the surface), like diffusive 
            This PDF is actually the PDF of cosine-weighted term * BSDF function value
            FIXME: to be more completed
        """
        pdf = 0.0
        if self._type == 0:
            pdf = tm.max(tm.dot(normal, outdir), 0.0) * INV_PI      # dot is cosine term
        elif self._type == 1:
            pdf = tm.max(tm.dot(normal, outdir), 0.0) * INV_PI
        return pdf