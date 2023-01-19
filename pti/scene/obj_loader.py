"""
    wavefront obj file loader
    @author Qianyue He
    @date 2023.1.19
"""

import numpy as np
from typing import Tuple
from numpy import ndarray as Arr
import xml.etree.ElementTree as xet
from scipy.spatial.transform import Rotation as Rot

def load_obj_file(path: str):
    print(path)

def parse_transform(transform_elem: xet.Element) -> Tuple[Arr, Arr]:
    return None, None

def apply_transform(obj: Arr, trans_r: Arr, trans_t: Arr) -> Arr:
    return obj