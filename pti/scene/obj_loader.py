"""
    wavefront obj file loader
    @author Qianyue He
    @date 2023.1.19
"""
__all__ = ["load_obj_file", "parse_transform", "apply_transform"]

import numpy as np
import pywavefront as pwf
import xml.etree.ElementTree as xet

from typing import Tuple
from numpy import ndarray as Arr
from scipy.spatial.transform import Rotation as Rot

# supported_rot_type = ("euler", "quaternion", "angle-axis")

def get(node: xet.Element, name: str, _type = float):
    return _type(node.get(name))

def load_obj_file(path: str, precomp_normal = True, verbose = False):
    """
        Meshes (output) are of shape (N_faces, 3, 3)
        Normals should be of shape (N_faces, 3)
    """
    obj         = pwf.Wavefront(path, collect_faces = True)
    vertices    = np.float32(obj.vertices)
    faces       = []
    for mesh in obj.mesh_list:
        for face_idx in mesh.faces:
            faces.append(vertices[face_idx])
    mesh_faces = np.stack(faces, axis = 0)
    normals = None
    if precomp_normal:
        normals = []
        for face_mesh in mesh_faces:
            dp1 = face_mesh[1] - face_mesh[0]
            dp2 = face_mesh[2] - face_mesh[1]
            normal = np.cross(dp1, dp2)         # preserves right-hand order
            normal /= np.linalg.norm(normal)
            normals.append(normal)
        normals = np.stack(normals, axis = 0)
    if verbose:
        print(f"Mesh loaded from {path}, output shape: {mesh_faces.shape}. Normal output: {precomp_normal}")
    return mesh_faces, normals

def transform_parse(transform_elem: xet.Element) -> Tuple[Arr, Arr]:
    """
        Note that: extrinsic rotation is not supported, 
        meaning that we can only rotate around the object centroid,
        which is [intrinsic rotation]. Yet, extrinsic rotation can be composed
        by intrinsic rotation and translation
    """
    trans_r, trans_t = None, None
    for child in transform_elem:
        if child.tag == "translate":
            trans_t = np.float32([get(child, "x"), get(child, "y"), get(child, "z")])
        elif child.tag == "rotate":
            rot_type = child.get("type")
            if rot_type == "euler":
                r_angle = get(child, "r")   # roll
                p_angle = get(child, "p")   # pitch
                y_angle = get(child, "y")   # yaw
                trans_r = Rot.from_euler("xyz", (r_angle, p_angle, y_angle), degrees = True).as_matrix()
            elif rot_type == "quaternion":
                trans_r = Rot.from_quat([get(child, "x"), get(child, "y"), get(child, "z"), get(child, "w")])
            elif rot_type == "angle-axis":
                axis: Arr = np.float32([get(child, "x"), get(child, "y"), get(child, "z")])
                axis /= np.linalg.norm(axis) * get(child, "angle") / 180. * np.pi
                trans_r = Rot.from_rotvec(axis)
            else:
                raise ValueError(f"Unsupported rotation representation '{rot_type}'")
        else:
            raise ValueError(f"Unsupported transformation representation '{child.tag}'")
    # Note that, trans_r (rotation) is defualt to be intrinsic (apply under the centroid coordinate)
    # Therefore, do not use trans_r unless you know how to correctly transform objects with it
    return trans_r, trans_t         # trans_t and trans_r could be None, if <transform> is not defined in the object

def apply_transform(meshes: Arr, normals: Arr, trans_r: Arr, trans_t: Arr) -> Arr:
    """
        - input normals are of shape (N, 3)
        - input meshes are of shape (N, 3, 3), and for the last two dims
            - 3(front): entry index for the vertices of triangles
            - 3(back): entry index for (x, y, z)
    """
    if trans_r is not None:
        center  = meshes.mean(axis = 1).mean(axis = 0)
        meshes -= center                # decentralize
        meshes = meshes @ trans_r      # right multiplication
        if normals is not None: 
            normals = normals @ trans_r # unit normn is preserved
        meshes += center
    if trans_t is not None:
        meshes += trans_t
    return meshes, normals