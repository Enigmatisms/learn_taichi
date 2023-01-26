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
    _type:  ti.i32      # 0 Point, 1 Area, 2 Spot, 3 Directional
    pos:    vec3
    dirv:   vec3
    base_1: vec3
    base_2: vec3
    l1:     ti.f32
    l2:     ti.f32

    def sample(self):
        """
            A unified sampling function, choose sampling strategy according to _type
        """
        if self._type == 0:     # point source
            print("Shit 1")
        elif self._type == 1:
            print("Shit 2")

    """
        structs = TaichiSource.field()
        ti.root.pointer(ti.i, 8).place(structs)
        print(structs)
    """

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
