import taichi as ti
import numpy as np

ti.init()

@ti.data_oriented
class Test:
    def __init__(self):
        self.container = ti.field(ti.f32)
        self.node = ti.root.dense(ti.i, 16)
        self.node.bitmasked(ti.j, 16).place(self.container)

        for i in range(16):
            for j in range(i):
                self.container[i, j] = i

field = ti.Matrix.field(3, 3, ti.f32, (40, 40))

matrix_1 = ti.Vector.field(3, ti.f32, 16)

for i in range(16):
    matrix_1[i] = ti.Vector([i, i, i])

@ti.kernel
def fast_test():
    for i, j in field:
        
        matrix = ti.Matrix.cols([matrix_1[(i + j) % 16], matrix_1[i % 16], matrix_1[j % 16]])
        px, py, pz = matrix_1[i % 16]
        print(px, py, pz)
        field[i, j].fill(3)
        
    
if __name__ == '__main__':
    test = Test()
    print(test.container)
    fast_test()
    print(field)

