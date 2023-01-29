"""
    TODO: Rectangluar area light source
"""

import sys
sys.path.append("..")

import numpy as np
from taichi.math import vec3
import xml.etree.ElementTree as xet

from emitters.abtract_source import LightSource, TaichiSource
from scene.general_parser import vec3d_parse

class RectAreaSource(LightSource):
    def __init__(self, elem: xet.Element):
        """
            This is more complex than other light sources
            The problem is how to export different kinds of light source to Taichi
        """
        super().__init__(elem, "rect_area")
        point_elems = elem.findall("point")
        assert(len(point_elems))

        self.ctr_pos = None
        self.base_1 = np.float32([1, 0, 0])         # x-axis by default
        self.base_2 = np.float32([0, 0, 1])         # z-axis by default
        for point in point_elems:
            name = point.get("name")
            if name == "center":
                self.ctr_pos: np.ndarray = vec3d_parse(point)
            elif name == "base_1":
                self.base_1 = vec3d_parse(point)
            elif name == "base_2":
                self.base_2 = vec3d_parse(point)
        self.normal = np.cross(self.base_1, self.base_2)
        bool_elem = elem.find("boolean")
        if bool_elem is not None and bool_elem.get("value").lower() == "true":
            self.normal *= -1
        assert(self.ctr_pos is not None)

        self.l1 = -1.0
        self.l2 = -1.0
        float_elems = elem.findall("float")
        for float_elem in float_elems:
            if float_elem.get("name") in ("width", "l1"):
                self.l1 = float(float_elem.get("value"))
            elif float_elem.get("name") in ("height", "l2"):
                self.l2 = float(float_elem.get("value"))
        assert(self.l1 > 0.0 and self.l2 > 0.0)

    def export(self) -> TaichiSource:
        return TaichiSource(
            _type = 1, intensity = vec3(self.intensity), pos = vec3(self.ctr_pos), dirv = vec3(self.normal),
            base_1 = vec3(self.base_1), base_2 = vec3(self.base_2), l1 = self.l1, l2 = self.l2
        )
        