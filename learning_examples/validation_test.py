import taichi as ti

"""
The following function would not run:
ti.random only supports primitive types
    @ti.kernel
    def random() -> ti.types.vector(3, ti.f32):
        return ti.random(ti.types.vector(3, ti.f32))
"""
