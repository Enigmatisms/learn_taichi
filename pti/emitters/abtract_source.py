"""
    Light source abstraction
    FIXME: sampling is not yet implemented currently (2023.1.20 version)
    @date: 2023.1.20
    @author: Qianyue He
"""

import sys
sys.path.append("..")

import xml.etree.ElementTree as xet
from scene.general_parser import rgb_parse

class LightSource:
    """
        FIXME: Too young, too simple. Sometimes naive. I'm angry. 
    """
    def __init__(self, base_elem: xet.Element, _type: str = "point"):
        intensity_elem = base_elem.find("rgb")
        self.intensity = rgb_parse(intensity_elem.get("value"))
        self.type: str = _type

    def sample(self):
        raise NotImplementedError("Can not call virtual method to be overridden.")

    def __repr__(self):
        return f"<{self.type.capitalize()} light source. Intensity: {self.intensity}>"
