"""
    TODO: Rectangluar area light source
"""

import sys
sys.path.append("..")

import numpy as np
import xml.etree.ElementTree as xet

from emitters.abtract_source import LightSource
from scene.general_parser import vec3d_parse

class RectAreaSource(LightSource):
    def __init__(self, elem: xet.Element):
        """
            This is more complex than other light sources
        """
        super().__init__(elem, "rect_area")
        raise NotImplementedError("Maybe in the future, mate.")