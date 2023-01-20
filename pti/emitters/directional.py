"""
    TODO: Directional light source (global paralleled ray light source)
"""

from abtract_source import LightSource

class DirectionalSource(LightSource):
    def __init__(self, intensity):
        super().__init__(intensity, "directional")
        raise NotImplementedError("Maybe in the future, mate.")