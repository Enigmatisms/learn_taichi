"""
    This py file serves as fast tester
"""

import taichi as ti
import numpy as np

"""
The following function would not run:
ti.random only supports primitive types
    @ti.kernel
    def random() -> ti.types.vector(3, ti.f32):
        return ti.random(ti.types.vector(3, ti.f32))
"""

ti.init()

def list_indexing_failure_example():
    glob = [ti.field(ti.f32, (20, 20)) for _ in range(8)]

    @ti.kernel
    def indexing(idx: int):
        field = glob[idx % 8]
        for i, j in field:
            field[i, j] = float(idx)

    print("Process started.")
    for i in range(64):
        indexing(i)
    print("Process completed.")

def vector_ndarray_failure_example():
    vnd = ti.Vector.ndarray(3, ti.f32, (5, 5))
    vnd.from_numpy(np.random.rand(5, 5, 3))
    vec = ti.Vector([1, 2, 3])

    @ti.kernel
    def set_vnd():
        for i in range(5):
            for j in range(5):
                vnd[i, j] = ti.Vector([i, j, i + j])

    @ti.kernel
    def set_vec():
        global vec
        vec = ti.Vector([4, 5, 6])

    print(f"Process started: {vec}")
    # set_vnd()
    set_vec()
    print(f"Process completed: {vec}")


def test_atomic():
    field = ti.field(ti.i32, (20, 20))
    
    # @ti.kernel
    # def increment():
    #     for i, j in field:
    #         ti.atomic_add(field[i, j], 1)

    print(f"Element[0, 0] = {field[0, 0]}")
    # increment()
    ti.atomic_add(field[0, 0], 1)
    print(f"Element[0, 0] = {field[0, 0]}")
    ti.VectorNdarray(3, ti.f32, (5,))
