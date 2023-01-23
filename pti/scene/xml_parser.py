"""
    Parsing scene XML file (simple one)
    currently, only BSDF, path tracer settings, emitter settings and .obj file
    configurations are supported
    @author Qianyue He (Enigmatisms)
    @date 2023.1.18
"""
import os
import sys
sys.path.append("..")

import numpy as np
import xml.etree.ElementTree as xet

from typing import List
from obj_loader import *
from numpy import ndarray as Arr

from bsdf.bsdfs import BlinnPhong
from scene.obj_desc import ObjDescriptor
from scene.general_parser import get, transform_parse

# import emitters
from emitters.point import PointSource
from emitters.rect_area import RectAreaSource
from emitters.directional import DirectionalSource

__MAPPING__ = {"integer": int, "float": float, "string": str}

"""
    Actually I think in Taichi, we can leverage SSDS:
    AABB and triangles are on the same level (attached to the same node)
    level 1    AABB1 (tri1 tri2 tri3)     AABB2 (tri4 tri5 tri6)     AABB3 (tri4 tri5 tri6)
"""

def parse_emitters(em_elem: list):
    """
        Parsing scene emitters from list of xml nodes \\
        only [Point], [Area], [Directional] are supported
        - for example, point light source is the simplest of call
        - rasterizer should have two functionalities: 
            - point source direct illumination (Phong model)
            - depth map rasterization
    """
    sources = []
    for elem in em_elem:
        emitter_type = elem.get("type")
        if emitter_type == "point":
            sources.append(PointSource(elem))
        elif emitter_type == "rect_area":
            sources.append(RectAreaSource(elem))
        elif emitter_type == "directional":
            sources.append(DirectionalSource(elem))
    return sources

def parse_wavefront(directory: str, obj_list: List[xet.Element], bsdf_dict: dict) -> List[Arr]:
    """
        Parsing wavefront obj file (filename) from list of xml nodes    
    """
    all_objs = []
    for elem in obj_list:
        trans_r, trans_t = None, None                           # transform
        filepath_child      = elem.find("string")
        ref_child           = elem.find("ref")        
        meshes, normals     = load_obj_file(os.path.join(directory, filepath_child.get("value")))
        transform_child     = elem.find("transform")
        if transform_child is not None:
            trans_r, trans_t    = transform_parse(transform_child)
            meshes, normals     = apply_transform(meshes, normals, trans_r, trans_t)
        # AABB calculation should be done after transformation
        if ref_child is None:
            raise ValueError("Object should be attached with a BSDF for now since no default one implemented yet.")
        bsdf_item = bsdf_dict[ref_child.get("id")]
        all_objs.append(ObjDescriptor(meshes, normals, bsdf_item, trans_r, trans_t))
    return all_objs

def parse_bsdf(bsdf_list: List[xet.Element]):
    """
        Parsing wavefront obj file (filename) from list of xml nodes    
        note that participating medium is complex, therefore will not be added in the early stage
        FIXME: bsdf is of the lowest priority, only after rasterizer is completed can this gets implemented
        return: dict
    """
    results = dict()
    for bsdf_node in bsdf_list:
        bsdf_type = bsdf_node.get("type")
        bsdf_id = bsdf_node.get("id")
        # FIXME: to be implemented
        if bsdf_type == "blinn-phong":
            bsdf = BlinnPhong(get(bsdf_node.find("float"), "value"))
        else:
            raise NotImplementedError(f"Work for BSDF type of {bsdf_type} is on the way")
        if bsdf_id in results:
            print(f"Warning: BSDF {bsdf_id} re-defined in XML file. Overwriting the existing BSDF.")
        results[bsdf_id] = bsdf
    return results

def parse_global_sensor(sensor_elem: xet.Element):
    """
        Parsing sensor (there can only be one sensor)
        Other global configs related to film, etc. are loaded here
    """
    sensor_config = {}
    for elem in sensor_elem:
        if elem.tag in __MAPPING__:
            name = elem.get("name")
            sensor_config[name] = get(elem, "value", __MAPPING__[elem.tag])

    sensor_config["transform"]  = transform_parse(sensor_elem.find("transform"))
    film_elems                  = sensor_elem.find("film").findall("integer")
    assert(len(film_elems) >= 3)        # at least width, height and sample count (meaningless for rasterizer)
    sensor_config["film"]       = np.int32([get(int_elem, "value", int) for int_elem in film_elems])
    return sensor_config

def mitsuba_parsing(directory: str, file: str):
    xml_file = os.path.join(directory, file)
    node_tree = xet.parse(xml_file)
    root_node = node_tree.getroot()
    version_tag = root_node.attrib["version"]
    if not version_tag == "1.0":
        raise ValueError(f"Unsupported version {version_tag}. Only '1.0' is supported right now.")  
    # Export list of dict for emitters / dict for other secen settings and film settings / list for obj files
    emitter_nodes   = root_node.findall("emitter")
    bsdf_nodes      = root_node.findall("bsdf")
    shape_nodes     = root_node.findall("shape")
    sensor_node     = root_node.find("sensor")
    assert(sensor_node)
    emitter_configs = parse_emitters(emitter_nodes)
    bsdf_dict       = parse_bsdf(bsdf_nodes)
    meshes          = parse_wavefront(directory, shape_nodes, bsdf_dict)
    configs         = parse_global_sensor(sensor_node)
    return emitter_configs, bsdf_dict, meshes, configs

if __name__ == "__main__":
    emitter_configs, bsdf_configs, meshes, configs = mitsuba_parsing("./test/", "test.xml")
    print(emitter_configs, bsdf_configs, meshes, configs)