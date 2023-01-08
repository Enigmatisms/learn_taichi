"""
    Marching square SDF zero set visualization with taichi
    @date 2023.1.8
"""

import numpy as np
import taichi as ti
import taichi.math as tm

COLOR_MAP = []  # create color map for up to 16 colors 

"""
    Color mixing strategy: Since the border of SDF (zero-set) is drawn as pure white (black background)
    the color inside (and the convex combination of the colors) will be clipped to (30, 30, 30) - (230, 230, 230)
    As is said above, color is mixed in a convex combination way, and tnere will be a tag field (u8)
    places with positive sdf value will be marked as 0x0000, and corresponding bit will be set to 1
    if a pixel is inside the corresponding ball. The color of a negative valued pixel (or tagged with non-zero u8)
    will be the convex weighted average of the color of each participating ball.
"""

@ti.data_oriented
class MarchingSquareSDF:
    def __init__(self, width, height, ball_prop: dict):
        self.w = width
        self.h = height
        self.ball_max = ball_prop['max_radius']
        self.ball_min = ball_prop['min_radius']
        self.ball_num = ball_prop['ball_num']                                   
        assert(self.ball_num <= 16)                                             # up to 15 balls

        self.sdf_map = ti.field(dtype = ti.f32, shape = (width, height))        # SDF calculation is done here
        self.tag_map = ti.field(dtype = ti.u16, shape = (width, height))        # tag map

        # RGB image is stored in vector.field
        self.pixels  = ti.Vector.field(3, dtype = ti.f32, shape = (width, height))     

        self.ball_pos   = ti.VectorNdarray(2, ti.f32, (self.ball_num,))     # 5 dims (px, py, vx, vy, radius)
        self.ball_vel   = ti.VectorNdarray(2, ti.f32, (self.ball_num,))     # 5 dims (px, py, vx, vy, radius)
        self.ball_radii = ti.field(dtype = ti.f32, shape = self.ball_num)     # 5 dims (px, py, vx, vy, radius)

        self.initialize_balls()

    @ti.kernel
    def initialize_balls(self):
        # initialize ball with random radius / position / speed (2D)
        pass

    @ti.kernel
    def update_balls(self):
        # update ball position (boundary checks - velocity changes)
        pass

    @ti.kernel
    def calculate_sdf_tag(self, ball_id: ti.u16):
        # Note that SDF value is calculate per pixel (top-left corner)
        tag_bit = 1 << ball_id
        radius      = self.ball_radii[ball_id]
        ball_pos    = self.ball_pos[ball_id]
        for i, j in self.sdf_map:
            # calculate distance to ball indexed by ball_id
            pos = ti.Vector([i, j])
            dist = (pos - ball_pos).norm() - radius
            """
                This should be reformulated. SDF value should be computed at pixel center
                while color and tag should be calculated at pixel top-left corner
                This way, tag can be determined by four closest neighbors (kind of like 2D linear interpolation)
                TODO and FIXME
            """            


    