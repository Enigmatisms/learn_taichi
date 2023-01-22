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
from scene.general_parser import rgb_parse, vec3d_parse


class PointSource(LightSource):
    def __init__(self, elem: xet.Element):
        super().__init__(elem, "point")
        pos_elem = elem.find("point")
        print(pos_elem, elem.tag)
        assert(pos_elem is not None)
        self.pos: np.ndarray = vec3d_parse(pos_elem)

    def sample(self):
        raise NotImplementedError("To be implemented.")
