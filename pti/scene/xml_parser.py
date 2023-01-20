"""
    Parsing scene XML file (simple one)
    currently, only BSDF, path tracer settings, emitter settings and .obj file
    configurations are supported
    @author Qianyue He (Enigmatisms)
    @date 2023.1.18
"""

import os
import numpy as np
import xml.etree.ElementTree as xet

from typing import List
from obj_loader import *
from obj_desc import ObjDescriptor
from numpy import ndarray as Arr

"""
    Actually I think in Taichi, we can leverage SSDS:
    AABB and triangles are on the same level (attached to the same node)
    level 1    AABB1 (tri1 tri2 tri3)     AABB2 (tri4 tri5 tri6)     AABB3 (tri4 tri5 tri6)
"""

def parse_emitters(em_elem: list):
    """
        Parsing scene emitters from list of xml nodes \\
        only [Point], [Area], [Directional] are supported \\
        TODO: Simple emitter parser for direct illumination can be implemented
        - for example, point light source is the simplest of call
        - rasterizer should have two functionalities: 
            - point source direct illumination (Phong model)
            - depth map rasterization
    """
    for elem in em_elem:
        emitter_type = elem.get("type")
        if emitter_type == "point":
            pass
    return []

def parse_wavefront(directory: str, obj_list: List[xet.Element]) -> List[Arr]:
    """
        Parsing wavefront obj file (filename) from list of xml nodes    
        TODO: the first function to implement (rasterizer)
    """
    all_objs = []
    for elem in obj_list:
        trans_r, trans_t = None, None                           # transform
        for children in elem:
            if children.tag == "string":
                meshes, normals = load_obj_file(os.path.join(directory, children.get("value")))
            elif children.tag == "transform":
                trans_r, trans_t = parse_transform(children)
        meshes, normals = apply_transform(meshes, normals, trans_r, trans_t)
        # AABB calculation should be done after transformation
        all_objs.append(ObjDescriptor(meshes, normals, trans_r, trans_t))
    return all_objs

def parse_bsdf(obj_list: list):
    """
        Parsing wavefront obj file (filename) from list of xml nodes    
        note that participating medium is complex, therefore will not be added in the early stage
        FIXME: bsdf is of the lowest priority, only after rasterizer is completed can this gets implemented
    """
    return []

def parse_global_sensor(sensor_elem: xet.Element):
    """
        Parsing sensor (there can only be one sensor)
        Other global configs related to film, etc. are loaded here
    """
    return dict()

def empty_or_front(lst: list, aux: str = "sensors"):
    if not lst:
        raise ValueError(f"List contains {aux} should not be empty")
    return lst[0]

def mitsuba_parsing(directory: str, file: str):
    xml_file = os.path.join(directory, file)
    node_tree = xet.parse(xml_file)
    root_node = node_tree.getroot()
    version_tag = root_node.attrib["version"]
    if not version_tag == "1.0":
        raise ValueError(f"Unsupported version {version_tag}. Only '1.0' is supported right now.")  
    # Export list of dict for emitters / dict for other secen settings and film settings / list for obj files
    children = [child for child in root_node]
    xet.Element
    filter_func = lambda iterable, tag: list(filter(lambda x: x.tag == tag, iterable))
    emitter_nodes   = filter_func(children, "emitter")
    bsdf_nodes      = filter_func(children, "bsdf")
    shape_nodes     = filter_func(children, "shape")
    sensor_node     = empty_or_front(filter_func(children, "sensor"))
    emitter_configs = parse_emitters(emitter_nodes)
    bsdf_configs    = parse_bsdf(bsdf_nodes)
    meshes          = parse_wavefront(directory, shape_nodes)
    configs         = parse_global_sensor(sensor_node)
    return emitter_configs, bsdf_configs, meshes, configs

if __name__ == "__main__":
    emitter_configs, bsdf_configs, meshes, configs = mitsuba_parsing("./test/", "test.xml")
    print(emitter_configs, bsdf_configs, meshes, configs)