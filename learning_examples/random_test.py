import taichi as ti

@ti.kernel
def random() -> ti.types.vector(3, ti.f32):
    return ti.random(ti.types.vector(3, ti.f32))

if __name__ == "__main__":
    ti.init()
    print(random())