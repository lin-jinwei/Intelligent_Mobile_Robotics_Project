from flight_environment import FlightEnvironment
import numpy as np
import heapq
import math
import os


start = (1, 2, 0)
goal = (18, 18, 3)
env = FlightEnvironment(50, protected_points=[start, goal])

# --------------------------------------------------------------------------------------------------- #
# Call your path planning algorithm here. 
# The planner should return a collision-free path and store it in the variable `path`. 
# `path` must be an N×3 numpy array, where:
#   - column 1 contains the x-coordinates of all path points
#   - column 2 contains the y-coordinates of all path points
#   - column 3 contains the z-coordinates of all path points
# This `path` array will be provided to the `env` object for visualization.

# --------------------------------------------------------------------------------------------------- #


def _is_free(point):
    return (not env.is_outside(point)) and (not env.is_collide(point))

def _segment_is_free(p1, p2, step=0.2):
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    dist = np.linalg.norm(p2 - p1)
    if dist == 0:
        return _is_free(p1)
    n = int(math.ceil(dist / step))
    for i in range(n + 1):
        alpha = i / n
        p = (1 - alpha) * p1 + alpha * p2
        if not _is_free(p):
            return False
    return True


def _nearest_free_index(idx, max_radius, dims):
    x_max, y_max, z_max = dims
    ix, iy, iz = idx
    for r in range(max_radius + 1):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    nx, ny, nz = ix + dx, iy + dy, iz + dz
                    if nx < 0 or ny < 0 or nz < 0:
                        continue
                    if nx > x_max or ny > y_max or nz > z_max:
                        continue
                    yield (nx, ny, nz)



# --------------------------------------------------------------------------------------------------- #
#   Call your trajectory planning algorithm here. The algorithm should
#   generate a smooth trajectory that passes through all the previously
#   planned path points.
#
#   After generating the trajectory, plot it in a new figure.
#   The figure should contain three subplots showing the time histories of
#   x, y, and z respectively, where the horizontal axis represents time (in seconds).
#
#   Additionally, you must also plot the previously planned discrete path
#   points on the same figure to clearly show how the continuous trajectory
#   follows these path points.




# --------------------------------------------------------------------------------------------------- #



# You must manage this entire project using Git. 
# When submitting your assignment, upload the project to a code-hosting platform 
# such as GitHub or GitLab. The repository must be accessible and directly cloneable. 
#
# After cloning, running `python3 main.py` in the project root directory 
# should successfully execute your program and display:
#   1) the 3D path visualization, and
#   2) the trajectory plot.
#
# You must also include the link to your GitHub/GitLab repository in your written report.
