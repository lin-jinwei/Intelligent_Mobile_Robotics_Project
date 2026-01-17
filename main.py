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


def plan_path(start_point, goal_point, resolution=0.5):
    width, length, height = env.env_width, env.env_length, env.env_height
    max_ix = int(round(width / resolution))
    max_iy = int(round(length / resolution))
    max_iz = int(round(height / resolution))

    def to_index(p):
        return (
            int(round(p[0] / resolution)),
            int(round(p[1] / resolution)),
            int(round(p[2] / resolution)),
        )

    def to_point(idx):
        return np.array(
            [idx[0] * resolution, idx[1] * resolution, idx[2] * resolution],
            dtype=float,
        )

    start_idx = to_index(start_point)
    goal_idx = to_index(goal_point)

    # Snap to nearest free grid nodes if needed
    dims = (max_ix, max_iy, max_iz)
    for candidate in _nearest_free_index(start_idx, 3, dims):
        if _is_free(to_point(candidate)):
            start_idx = candidate
            break
    for candidate in _nearest_free_index(goal_idx, 3, dims):
        if _is_free(to_point(candidate)):
            goal_idx = candidate
            break

    neighbors = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                neighbors.append((dx, dy, dz))

    open_set = []
    heapq.heappush(open_set, (0.0, start_idx))
    came_from = {}
    g_score = {start_idx: 0.0}

    def heuristic(a, b):
        pa = to_point(a)
        pb = to_point(b)
        return float(np.linalg.norm(pa - pb))

    visited = set()
    while open_set:
        _, current = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)
        if current == goal_idx:
            break
        for dx, dy, dz in neighbors:
            nx, ny, nz = current[0] + dx, current[1] + dy, current[2] + dz
            if nx < 0 or ny < 0 or nz < 0:
                continue
            if nx > max_ix or ny > max_iy or nz > max_iz:
                continue
            neighbor = (nx, ny, nz)
            if neighbor in visited:
                continue
            p_current = to_point(current)
            p_neighbor = to_point(neighbor)
            if not _is_free(p_neighbor):
                continue
            if not _segment_is_free(p_current, p_neighbor):
                continue
            tentative = g_score[current] + np.linalg.norm(p_neighbor - p_current)
            if tentative < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                f_score = tentative + heuristic(neighbor, goal_idx)
                heapq.heappush(open_set, (f_score, neighbor))

    if goal_idx not in came_from and goal_idx != start_idx:
        raise RuntimeError("Path planning failed. Try adjusting resolution or start/goal.")

    # Reconstruct path
    path_indices = [goal_idx]
    while path_indices[-1] != start_idx:
        path_indices.append(came_from[path_indices[-1]])
    path_indices.reverse()
    return np.array([to_point(idx) for idx in path_indices], dtype=float)



# --------------------------------------------------------------------------------------------------- #

# 定义起点、终点
path = plan_path(start, goal, resolution=0.5)

# 保存图片
os.makedirs("imgs", exist_ok=True)
env.plot_cylinders(
    path=None,
    start=start,
    goal=goal,
    save_path=os.path.join("imgs", "p1.png"),
    show=False,
)
env.plot_cylinders(
    path,
    start=start,
    goal=goal,
    save_path=os.path.join("imgs", "p2.png"),
    show=True,
)

# 规划轨迹的生成函数
def generate_trajectory(path_points, speed=1.0, dt=0.05):
    path_points = np.array(path_points, dtype=float)
    if len(path_points) < 2:
        raise ValueError("Path must contain at least two points.")

    # Cumulative time at each path point based on distance and speed
    diffs = np.linalg.norm(np.diff(path_points, axis=0), axis=1)
    t_points = np.concatenate(([0.0], np.cumsum(diffs / speed)))

    samples_t = []
    samples_xyz = []
    for i in range(len(path_points) - 1):
        p1 = path_points[i]
        p2 = path_points[i + 1]
        seg_time = max(t_points[i + 1] - t_points[i], dt)
        n_samples = max(int(math.ceil(seg_time / dt)), 2)
        for j in range(n_samples):
            tau = j / (n_samples - 1)
            s = 3 * tau * tau - 2 * tau * tau * tau  # smooth step (zero vel at ends)
            point = p1 + s * (p2 - p1)
            samples_xyz.append(point)
            samples_t.append(t_points[i] + tau * seg_time)

    traj = np.array(samples_xyz, dtype=float)
    t = np.array(samples_t, dtype=float)
    return t, traj, t_points


t, traj, t_points = generate_trajectory(path, speed=1.0, dt=0.05)

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
