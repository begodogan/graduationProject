import tkinter as tk
from tkinter import messagebox
import time
import tracemalloc
import matplotlib.pyplot as plt
import numpy as np
import math
import heapq

def heuristic(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def astar(array, start, goal):
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))

    while oheap:
        current = heapq.heappop(oheap)[1]

        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data[::-1]

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + (math.sqrt(2) if abs(i) + abs(j) == 2 else 1)
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:
                    if array[neighbor[0]][neighbor[1]] >= 0.5:
                        continue
                else:
                    continue
            else:
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))

    return False

def file_read(f):
    with open(f) as data:
        measures = [line.split(",") for line in data]
    angles = []
    distances = []
    for measure in measures:
        angles.append(float(measure[0]))
        distances.append(float(measure[1]))
    angles = np.array(angles)
    distances = np.array(distances)
    return angles, distances

def bresenham(start, end):
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    is_steep = abs(dy) > abs(dx)
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
    dx = x2 - x1
    dy = y2 - y1
    error = int(dx / 2.0)
    y_step = 1 if y1 < y2 else -1
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = [y, x] if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += y_step
            error += dx
    if swapped:
        points.reverse()
    points = np.array(points)
    return points

def calc_grid_map_config(ox, oy, xy_resolution):
    EXTEND_AREA = 1.0
    min_x = round(min(ox) - EXTEND_AREA / 2.0)
    min_y = round(min(oy) - EXTEND_AREA / 2.0)
    max_x = round(max(ox) + EXTEND_AREA / 2.0)
    max_y = round(max(oy) + EXTEND_AREA / 2.0)
    xw = int(round((max_x - min_x) / xy_resolution))
    yw = int(round((max_y - min_y) / xy_resolution))
    return min_x, min_y, max_x, max_y, xw, yw

def generate_ray_casting_grid_map(ox, oy, xy_resolution, breshen=True):
    min_x, min_y, max_x, max_y, x_w, y_w = calc_grid_map_config(ox, oy, xy_resolution)
    occupancy_map = np.ones((x_w, y_w)) / 2
    center_x = int(round(-min_x / xy_resolution))
    center_y = int(round(-min_y / xy_resolution))
    if breshen:
        for (x, y) in zip(ox, oy):
            ix = int(round((x - min_x) / xy_resolution))
            iy = int(round((y - min_y) / xy_resolution))
            laser_beams = bresenham((center_x, center_y), (ix, iy))
            for laser_beam in laser_beams:
                occupancy_map[laser_beam[0]][laser_beam[1]] = 0.0
            occupancy_map[ix][iy] = 1.0
            occupancy_map[ix + 1][iy] = 1.0
            occupancy_map[ix][iy + 1] = 1.0
            occupancy_map[ix + 1][iy + 1] = 1.0
    else:
        occupancy_map = init_flood_fill((center_x, center_y), (ox, oy), (x_w, y_w), (min_x, min_y), xy_resolution)
        flood_fill((center_x, center_y), occupancy_map)
        occupancy_map = np.array(occupancy_map, dtype=float)
        for (x, y) in zip(ox, oy):
            ix = int(round((x - min_x) / xy_resolution))
            iy = int(round((y - min_y) / xy_resolution))
            occupancy_map[ix][iy] = 1.0
            occupancy_map[ix + 1][iy] = 1.0
            occupancy_map[ix][iy + 1] = 1.0
            occupancy_map[ix + 1][iy + 1] = 1.0
    return occupancy_map, min_x, max_x, min_y, max_y, xy_resolution

def path_length(path):
    if not path:
        return float('inf')
    return len(path)

def main(start, goal):
    xy_resolution = 0.05
    ang, dist = file_read(f"C:\\Users\\emrey\\Downloads\\lidar1.csv")
    ox = np.sin(ang) * dist
    oy = np.cos(ang) * dist
    occupancy_map, min_x, max_x, min_y, max_y, xy_resolution = generate_ray_casting_grid_map(ox, oy, xy_resolution, True)

    if occupancy_map[start[0]][start[1]] >= 0.5 or occupancy_map[goal[0]][goal[1]] >= 0.5:
        print("Start or goal is in an occupied area!")
        return

    tracemalloc.start()
    path = astar(occupancy_map, start, goal)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    length = path_length(path)
    if path:
        print(f"A* - Path Length: {length}")
    else:
        print("A* - No valid path found!")

    # Plotting the path
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.get_cmap('viridis')
    cax = ax.imshow(occupancy_map, cmap=cmap, origin='lower')
    cbar = fig.colorbar(cax, ticks=[0, 1])
    cbar.ax.set_yticklabels(['Free',  'Occupied'])

    # Grid lines
    ax.set_xticks(np.arange(occupancy_map.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(occupancy_map.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='k', linestyle='-', linewidth=1)

    # Mark start and goal points
    ax.plot(start[1], start[0], 'go')  # Start point
    ax.plot(goal[1], goal[0], 'ro')  # Goal point

    if path:
        path = np.array(path)
        ax.plot(path[:, 1], path[:, 0], 'r--', linewidth=2)

    plt.show()

def run_pathfinding():
    try:
        start_x = int(start_x_entry.get())
        start_y = int(start_y_entry.get())
        goal_x = int(goal_x_entry.get())
        goal_y = int(goal_y_entry.get())
        start = (start_x, start_y)
        goal = (goal_x, goal_y)
        main(start, goal)
    except ValueError:
        messagebox.showerror("Invalid input", "Please enter valid integer coordinates for start and goal points.")

# Create the main window
root = tk.Tk()
root.title("Pathfinding Algorithm - A*")

# Create and place labels and entry fields
tk.Label(root, text="Start X:").grid(row=0, column=0)
start_x_entry = tk.Entry(root)
start_x_entry.grid(row=0, column=1)

tk.Label(root, text="Start Y:").grid(row=1, column=0)
start_y_entry = tk.Entry(root)
start_y_entry.grid(row=1, column=1)

tk.Label(root, text="Goal X:").grid(row=2, column=0)
goal_x_entry = tk.Entry(root)
goal_x_entry.grid(row=2, column=1)

tk.Label(root, text="Goal Y:").grid(row=3, column=0)
goal_y_entry = tk.Entry(root)
goal_y_entry.grid(row=3, column=1)

# Create and place the run button
run_button = tk.Button(root, text="Run Pathfinding", command=run_pathfinding)
run_button.grid(row=4, column=0, columnspan=2)

# Start the main event loop
root.mainloop()