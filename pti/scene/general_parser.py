"""
    Various kinds of low level element parsers
    @date: 2023.1.20
    @author: Qianyue He
"""

import numpy as np

def rgb_parse(val_str: str):
    if val_str.startswith("#"): # html-like hexidecimal RGB
        rgb = np.zeros(3, dtype = np.float32)
        for i in range(3):
            base = 1 + (i << 1)
            rgb[i] = int(val_str[base:base + 2], 16) / 255.
        return rgb
    else:
        splitter = (',', ' ')
        for split in splitter:
            if split in val_str:
                all_parts = val_str.split(split)
                return np.float32([float(part.strip()) for part in all_parts])
        else:       # single scalar marked as RGB
            return np.float32([float(val_str)] * 3)

