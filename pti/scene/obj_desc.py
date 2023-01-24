"""
    Object descriptor (simple)
    @date 2023.1.20
"""

import numpy as np
from numpy import ndarray as Arr

def get_aabb(meshes: Arr) -> Arr:
    """
        Axis-aligned bounding box for one object
        input: meshes of shape (N, 3, 3), output: two 3D point describing an AABB
    """
    mini = meshes.min(axis = 1).min(axis = 0)
    maxi = meshes.max(axis = 1).max(axis = 0)
    return np.float32((mini, maxi))

class ObjDescriptor:
    def __init__(self, meshes, normals, bsdf, R = None, t = None):
        """
            Inputs are objects on which transformations have been applied already
        """
        self.tri_num = meshes.shape[0]
        self.meshes = meshes
        self.normals = normals
        self.R = R
        self.t = t
        self.aabb = get_aabb(meshes)        # of shape (2, 3)
        self.bsdf = bsdf

    def __repr__(self):
        centroid = (self.aabb[0] + self.aabb[1]) / 2
        return f"<object with {self.meshes.shape[0]} triangles centered at {centroid}.\n Transformed: {self.R is not None or self.t is not None}>"


