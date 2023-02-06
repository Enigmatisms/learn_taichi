"""
    Bidirectional Scattering Distribution Function
    This is more general than BRDF, since it combines BRDF and BTDF
    I... actually know nothing of this before...
    @author: Qianyue He
    @date: 2023-2-5
"""

import sys
sys.path.append("..")

import numpy as np
import taichi as ti
import taichi.math as tm
import xml.etree.ElementTree as xet
from taichi.math import vec3, vec4, mat3

from la.geo_optics import *
from la.cam_transform import *
from sampler.general_sampling import *
from scene.general_parser import rgb_parse, get
from bxdf.brdf import BRDF_np

__all__ = ['BSDF_np', 'BSDF']

INV_PI = 1. / tm.pi

class BSDF_np(BRDF_np):
    """
        BSDF base-class, 
        @author: Qianyue He
        @date: 2023-2-5
    """
    def __init__(self, elem: xet.Element):
        super().__init__(elem, True)
        self.ior = 1.0
        all_float_elems = elem.findall("float")
        for float_elem in all_float_elems:
            name = float_elem.get("name")
            if name == "ior":
                self.ior = get(float_elem, "value")

    def setup(self):
        pass

    def export(self):       # TODO: override
        return BSDF(
            _type = self.type_id, is_delta = self.is_delta, k_d = vec3(self.k_d), 
            k_s = vec3(self.k_s), k_g = vec3(self.k_g), k_a = vec3(self.k_a), ior = self.ior
        )
        
    def __repr__(self) -> str:
        return f"<{self.type.capitalize()} BSDF with ior: {self.ior:.3f} default:[{int(self.kd_default), int(self.ks_default), int(self.kg_default)}]>"
    

@ti.dataclass
class BSDF:
    """
        TODO: 
        - implement simple BSDF first (simple refraction and mirror surface / glossy surface / lambertian surface)
        - transmission and reflection have independent distribution, yet transmission can be stochastic 
    """
    _type:      ti.i32
    is_delta:   ti.i32          # whether the BRDF is Dirac-delta-like
    k_d:        vec3            # diffusive coefficient (albedo)
    k_s:        vec3            # specular coefficient
    k_g:        vec3            # glossiness coefficient
    k_a:        vec3            # absorption coefficient
    k_t:        vec3            # transmission coeffcient
    ior:        ti.f32

    def sample_det_refraction():
        """ Deterministic refraction sampling - Surface reflection model: 
            A homespun dot powered reflection (hemisphere dot power)
        """
        pass


    