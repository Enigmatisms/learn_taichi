"""
    Various kinds of low level element parsers
    @date: 2023.1.20
    @author: Qianyue He
"""

import numpy as np
from typing import Tuple
from numpy import ndarray as Arr
import xml.etree.ElementTree as xet
from scipy.spatial.transform import Rotation as Rot

def get(node: xet.Element, name: str, _type = float):
    return _type(node.get(name))

def parse_str(val_str: str) -> Arr:
    splitter = (',', ' ')
    for split in splitter:
        if split in val_str:
            all_parts = val_str.split(split)
            return np.float32([float(part.strip()) for part in all_parts])
    else:       # single scalar marked as RGB
        return np.float32([float(val_str.strip())] * 3)

def rgb_parse(val_str: str):
    if val_str.startswith("#"): # html-like hexidecimal RGB
        rgb = np.zeros(3, dtype = np.float32)
        for i in range(3):
            base = 1 + (i << 1)
            rgb[i] = int(val_str[base:base + 2], 16) / 255.
        return rgb
    else:
        return parse_str(val_str)

def vec3d_parse(elem: xet.Element):
    if elem.tag == "point" and elem.get("name") in ("position", "direction"):
        return np.float32([get(elem, "x"), get(elem, "y"), get(elem, "z")])

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
        elif child.tag.lower() == "lookat":
            target_point = parse_str(child.get("target"))
            origin_point = parse_str(child.get("origin"))
            direction = target_point - origin_point
            dir_norm = np.linalg.norm(direction)
            if dir_norm < 1e-5:
                raise ValueError("Normal length too small: Target and origin seems to be the same point")
            # up in XML field is not usefull (polarization), trans_r being a vector means directional vector
            trans_r = direction / dir_norm
            trans_t = origin_point
        else:
            raise ValueError(f"Unsupported transformation representation '{child.tag}'")
    # Note that, trans_r (rotation) is defualt to be intrinsic (apply under the centroid coordinate)
    # Therefore, do not use trans_r unless you know how to correctly transform objects with it
    return trans_r, trans_t         # trans_t and trans_r could be None, if <transform> is not defined in the object
    