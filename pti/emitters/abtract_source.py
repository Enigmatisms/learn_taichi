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

class LightSource:
    """
        FIXME: Too young, too simple. Sometimes naive. I'm angry. 
    """
    def __init__(self, base_elem: xet.Element = None, _type: str = "point"):
        if base_elem is not None:
            intensity_elem = base_elem.find("rgb")
            self.intensity = rgb_parse(intensity_elem.get("value"))
        else:
            print("Warning: default intializer should only be used in testing.")
            self.intensity = np.ones(3, np.float32)
        self.type: str = _type

    def sample(self):
        raise NotImplementedError("Can not call virtual method to be overridden.")

    def __repr__(self):
        return f"<{self.type.capitalize()} light source. Intensity: {self.intensity}>"
