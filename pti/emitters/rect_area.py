"""
    TODO: Rectangluar area light source
"""

from abtract_source import LightSource

class RectAreaSource(LightSource):
    def __init__(self, intensity):
        super().__init__(intensity, "rect_area")
        raise NotImplementedError("Maybe in the future, mate.")