"""
    The simplest light source: point source
    @date: 2023.1.20
    @author: Qianyue He
    sampling is not yet implemented (2023.1.20 version)
"""

import numpy as np
from abtract_source import LightSource

class PointSource(LightSource):
    def __init__(self, position, intensity):
        super().__init__(intensity, "point")
        self.pos = position

    def sample(self):
        raise NotImplementedError("To be implemented.")
