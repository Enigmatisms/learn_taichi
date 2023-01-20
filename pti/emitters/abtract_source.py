"""
    Light source abstraction
    FIXME: sampling is not yet implemented currently (2023.1.20 version)
    @date: 2023.1.20
    @author: Qianyue He
"""

class LightSource:
    """
        FIXME: Too young, too simple. Sometimes naive. I'm angry. 
    """
    def __init__(self, intensity, _type: str = "point"):
        self.intensity = intensity
        self.type: str = _type

    def sample(self):
        raise NotImplementedError("Can not call virtual method to be overridden.")

    def __repr__(self):
        return f"<{self.type.capitalize()} light source. Intensity: {self.intensity}>"
