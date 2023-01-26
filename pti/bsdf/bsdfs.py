"""
    All the bsdfs are here, note that only three kinds of simple BSDF are supported
    Diffusive / Glossy / Pure specular (then it will be participating media)
    @author: Qianyue He
    @date: 2023-1-23
"""
import numpy as np

__all__ = ["BlinnPhong"]

class BSDF:
    """
        BSDF base-class, 
        @author: Qianyue He
        @date: 2023-1-23

        TODO: other reflectance models to be implemented
        for tracers, BSDF base class is enough
    """
    def __init__(self, rflct: np.ndarray, _type: str):
        self.type: str = _type
        self.reflectance = rflct if rflct is not None else np.ones(3, np.float32)
    
    def __repr__(self) -> str:
        return f"<{self.type.capitalize()} BSDF with reflectance {self.reflectance}>"

class BlinnPhong(BSDF):
    """
        "BSDF" used in Blinn-Phong tracer, not an actual bsdf
        This class is extremely simple
        @author: Qianyue He
        @date: 2023-1-23
    """
    def __init__(self, reflectance, shininess):
        super().__init__(reflectance, "blinn-phong")
        assert(shininess >= 1.0)    
        self.shininess = shininess

    def __repr__(self) -> str:
        return f"<Blinn-Phong BSDF. Shininess = {self.shininess:.4f}. reflectance = {self.reflectance}>"
