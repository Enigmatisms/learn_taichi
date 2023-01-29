"""
    Light source abstraction
    FIXME: sampling is not yet implemented currently (2023.1.20 version)
    @date: 2023.1.20
    @author: Qianyue He
"""

import sys
sys.path.append("..")

import numpy as np
import xml.etree.ElementTree as xet
from scene.general_parser import rgb_parse

import taichi as ti
from taichi.math import vec3

@ti.dataclass
class TaichiSource:
    """
        This implementation is not elegant, too obese. Of course, there are better ways \\
        For example, for every 'variable field', we can place them to bit-masked: \\
        node = pointer -> bitmasked \\
        node.place(pos); node.place(dirv); node.place(base_1); node.place(base_2); \\
        The following implementation is much simpler, and light source will not consume too much memory
    """
    _type:      ti.i32      # 0 Point, 1 Area, 2 Spot, 3 Directional
    intensity:  vec3
    pos:        vec3
    dirv:       vec3
    base_1:     vec3
    base_2:     vec3
    l1:         ti.f32
    l2:         ti.f32

    @ti.func
    def distance_attenuate(self, x: vec3):
        return ti.min(1.0 / (1e-5 + x.norm_sqr()), 1e5)

    @ti.func
    def sample(self, hit_pos: vec3):
        """
            A unified sampling function, choose sampling strategy according to _type \\
            input ray hit point \\
            returns <sampled source point> <souce intensity> <sample pdf> 
        """
        ret_int = self.intensity
        ret_pos = self.pos
        ret_pdf = 1.0
        if self._type == 0:     # point source
            ret_int *= self.distance_attenuate(hit_pos - ret_pos)
        elif self._type == 1:   # area source
            ret_pdf = 1. / (self.l1 * self.l2)
            dot_light = ti.math.dot(hit_pos - self.pos, self.dirv)
            if dot_light <= 0.0:
                ret_int = vec3([0, 0, 0])
                ret_pdf = 1.0
            else:
                rand_axis1 = ti.random(float) - 0.5
                rand_axis2 = ti.random(float) - 0.5

                v_axis1 = self.base_1 * self.l1 * rand_axis1
                v_axis2 = self.base_2 * self.l2 * rand_axis2
                ret_pos += (v_axis1 + v_axis2)
                ret_int *= (self.distance_attenuate(ret_pos - hit_pos) * dot_light)
        return ret_pos, ret_int, ret_pdf

class LightSource:
    """
        Sampling function is implemented in Taichi source. Currently:
        Point / Area / Spot / Directional are to be supported
    """
    def __init__(self, base_elem: xet.Element = None, _type: str = "point"):
        if base_elem is not None:
            self.intensity = rgb_parse(base_elem.find("rgb"))
        else:
            print("Warning: default intializer should only be used in testing.")
            self.intensity = np.ones(3, np.float32)
        self.type: str = _type

    def export(self) -> TaichiSource:
        """
            Export to taichi
        """
        raise NotImplementedError("Can not call virtual method to be overridden.")

    def __repr__(self):
        return f"<{self.type.capitalize()} light source. Intensity: {self.intensity}>"
