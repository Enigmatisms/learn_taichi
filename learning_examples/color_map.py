
import taichi as ti

__all__ = ['COLOR_MAP']

COLOR_MAP = [
    ti.Vector([0.82745, 0.24706, 0.28627]),     # Bright red
    ti.Vector([0.95686, 0.82745, 0.36863]),     # Yellow
    ti.Vector([0.29020, 0.34510, 0.60000]),     # Liberty blue
    ti.Vector([0.61176, 0.92549, 0.35686]),     # Light green
    ti.Vector([0.27451, 0.13333, 0.33333]),     # Voilet

    ti.Vector([0.28627, 0.06667, 0.10980]),     # Dark red
    ti.Vector([0.93333, 0.58824, 0.29412]),     # Sandy orange
    ti.Vector([0.35686, 0.75294, 0.74510]),     # Blue green
    ti.Vector([0.77255, 0.90196, 0.65098]),     # Tea green  
    ti.Vector([0.56863, 0.30196, 0.46275]),     # Purple plum

    ti.Vector([0.97647, 0.34118, 0.21961]),     # Redish orange
    ti.Vector([0.92941, 0.87059, 0.64314]),     # Champagne yellow
    ti.Vector([0.19608, 0.38431, 0.45098]),     # Sapphire blue
    ti.Vector([0.64706, 0.58039, 0.97647]),     # Blue purple
    ti.Vector([0.66275, 0.57255, 0.49020]),     # Light brown
]  # create color map for up to 16 colors 