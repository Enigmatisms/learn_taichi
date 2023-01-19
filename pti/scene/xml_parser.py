"""
    Parsing scene XML file (simple one)
    currently, only BSDF, path tracer settings, emitter settings and .obj file
    configurations are supported
    @author Qianyue He (Enigmatisms)
    @date 2023.1.18
"""

import numpy as np
from typing import List
from obj_loader import *
from numpy import ndarray as Arr
import xml.etree.ElementTree as xet

def parse_emitters(em_elem: list):
    """
        Parsing scene emitters from list of xml nodes
        only [Point], [Area], [Directional] are supported
    """
    return []

def parse_wavefront(obj_list: List[xet.Element]) -> List[Arr]:
    """
        Parsing wavefront obj file (filename) from list of xml nodes    
        TODO: the first function to implement (rasterizer)
    """
    all_objs = []
    for elem in obj_list:
        obj = np.empty(0, dtype = np.float32)                   # obj should a bunch of triangles
        trans_r, trans_t = None, None                           # transform
        for children in elem:
            if children.tag == "string":
                obj = load_obj_file(children.get("value"))
            elif children.tag == "transform":
                trans_r, trans_t = parse_transform(children)
        obj = apply_transform(obj, trans_r, trans_t)
        all_objs.append(obj)
    return all_objs

def parse_bsdf(obj_list: list):
    """
        Parsing wavefront obj file (filename) from list of xml nodes    
        note that participating medium is complex, therefore will not be added in the early stage
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

def mitsuba_parsing(path: str):
    node_tree = xet.parse(path)
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
    meshes          = parse_wavefront(shape_nodes)
    configs         = parse_global_sensor(sensor_node)
    return emitter_configs, bsdf_configs, meshes, configs

if __name__ == "__main__":
    emitter_configs, bsdf_configs, meshes, configs = mitsuba_parsing("./test/test.xml")
    print(emitter_configs, bsdf_configs, meshes, configs)