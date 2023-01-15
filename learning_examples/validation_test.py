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


def test_unpack():
    
    f = ti.field(float, 16)
    vf = ti.Vector.field(2, float, 16)

    @ti.func
    def valid_angle(ang, diff = False):
        value = ang
        if ang > ti.math.pi:
            if diff:
                value = 2.0 - ang
            else:
                value = ang - 2.0
        elif ang < -ti.math.pi:
            value = ang + 2.0
        return value

    @ti.kernel
    def initialize():
        for i in vf:
            x, y = vf[i]
            f[i] = valid_angle(x + y, True)

    initialize()

def ssds_test():        # test for spatially sparse data structure
    gx = 5
    gy = 5
    masked_grid = ti.field(ti.i32)
    sn_handle = ti.root.bitmasked(ti.ij, (gx + 2, gy + 2))
    sn_handle.place(masked_grid, offset = (-1, -1))
    coords = ti.Vector.field(2, ti.i16, 5)

    @ti.func
    def neighbor_search(idx: ti.i16):
        g_i, g_j = coords[idx]
        sums = 0
        for i in range(g_i - 1, g_i + 2):
            for j in range(g_j - 1, g_j + 2):
                sums += masked_grid[i, j]
        return sums

    @ti.kernel
    def init_masked():
        # for i, j, k in masked_grid:
        #     masked_grid[i, j, k] = 2
        for i in ti.static(range(-1, 3)):
            for j in ti.static(range(-1, 3)):
                    masked_grid[i, j] = i
                    masked_grid[i, j] += neighbor_search(0)
    @ti.kernel
    def print_one(i: ti.i32):
        for j in masked_grid[i, :]:
            print(masked_grid[i, j])

    @ti.kernel
    def print_active():
        for i, j in sn_handle:
            print("Active block", i, j)
        for i, j in masked_grid:
            print("Yeah")
            print('field x[{}, {}] = {}'.format(i, j, masked_grid[i, j]))

    init_masked()
    # sn_handle.deactivate_all()
    print_active()
    print(masked_grid)
    print("[", end='')
    for i in range(-1, 6):
        print("[", end='')
        for j in range(-1, 6):
            print(f"{masked_grid[i, j]} ", end='')
        print("]")
    print("]")

    print_one(3)

ssds_test() 