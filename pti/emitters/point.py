"""
    The simplest light source: point source
    @date: 2023.1.20
    @author: Qianyue He
    sampling is not yet implemented (2023.1.20 version)
"""

import sys
sys.path.append("..")

import numpy as np
import xml.etree.ElementTree as xet

from emitters.abtract_source import LightSource
from scene.general_parser import vec3d_parse


class PointSource(LightSource):
    def __init__(self, elem: xet.Element = None):
        super().__init__(elem, "point")
        if elem is not None:
            pos_elem = elem.find("point")
            assert(pos_elem is not None)
            self.pos: np.ndarray = vec3d_parse(pos_elem)
        else:
            self.pos = np.zeros(3, np.float32)
        all_felems = elem.findall("float")
        self.half_w = 4.0
        for float_elem in all_felems:
            if float_elem.get("name") == "half_decay":
                self.half_w = float(float_elem.get("value"))

    def sample(self):
        raise NotImplementedError("To be implemented.")
