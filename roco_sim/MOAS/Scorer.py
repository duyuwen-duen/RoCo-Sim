import os
import sys
import math
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.append(f"{os.path.dirname(__file__)}/../../data_utils/data_prepare")
from get_possible_position import check_in_image

np.random.seed(0)
MIN = -1e6


def read_possible_pos(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def get_map_range(possible_points):
    x_vals = [obj['3d_location']['x'] for obj in possible_points]
    y_vals = [obj['3d_location']['y'] for obj in possible_points]

    raw_x_min = int(min(x_vals))
    raw_x_max = int(max(x_vals))
    raw_y_min = int(min(y_vals))
    raw_y_max = int(max(y_vals))

    # 向上取整最接近的 20 倍数（缩小范围）
    x_min = ((raw_x_min + 19) // 20) * 20
    x_max = (raw_x_max // 20) * 20

    y_min = ((raw_y_min + 19) // 20) * 20
    y_max = (raw_y_max // 20) * 20

    return [x_min, x_max, y_min, y_max]


class Scorer:
    def __init__(self, args):
        # Grid and map config
        self.grid_size = [20, 20]
        self.candidate_points = read_possible_pos(args.insert_pos_file)
        self.map_size = get_map_range(self.candidate_points)
        # Scoring params
        self.view_weight = args.view_weight
        self.occupancy_weight = args.occupancy_weight
        self.cam_num = args.cam_num
        self.draw_map = args.draw_map
        self.H_list = args.H
        self.W_list = args.W

        # Visualization
        if self.draw_map:
            self.fig, self.ax = plt.subplots()

        # Internal state
        self.W_num = 0
        self.H_num = 0
        self.grid_points,self.grid_view_points,self.grid_view_score,self.grid_score,self.grid_occupancy = [],[],[],[],[]

    def init_map_scorer(self, args, l2cs, camera_intrinsics, l2ws):
        # Compute grid layout
        self.W_num = (self.map_size[1] - self.map_size[0]) // self.grid_size[0]
        self.H_num = (self.map_size[3] - self.map_size[2]) // self.grid_size[1]

        # Initialize containers
        self.grid_score = np.zeros((self.W_num, self.H_num))
        self.grid_occupancy = np.zeros((self.W_num, self.H_num))
        self.grid_views = np.zeros((self.W_num, self.H_num))
        self.grid_points = np.empty((self.W_num, self.H_num), dtype=object)
        self.grid_view_points = np.empty((self.W_num, self.H_num, self.cam_num), dtype=object)
        self.grid_view_score = np.zeros((self.W_num, self.H_num, self.cam_num))

        for i in range(self.W_num):
            for j in range(self.H_num):
                self.grid_points[i, j] = []
                for k in range(self.cam_num):
                    self.grid_view_points[i, j, k] = []
        # Read insertable points
        for data in self.candidate_points:
            x, y = data['3d_location']['x'], data['3d_location']['y']
            if not self._in_map_bounds(x, y):
                continue
            i, j = self._grid_index(x, y)
            self.grid_points[i][j].append(data)
            if self.draw_map:
                self.plot_grid((x, y), 'green', alpha=0.1)

        # Calculate visibility for each grid
        self.points_ratio = np.zeros((self.W_num, self.H_num))
        for i in range(self.W_num):
            for j in range(self.H_num):
                points = self.grid_points[i, j]
                self.points_ratio[i, j] = len(points)

                if len(points) == 0:
                    self.grid_views[i, j] = MIN
                    continue

                view_scores = np.zeros(self.cam_num)
                for data in points:
                    view_result = check_in_image(data, l2cs, camera_intrinsics, l2ws,
                                                 args.base_road, self.H_list, self.W_list)
                    for cam in range(self.cam_num):
                        if view_result[cam]:
                            self.grid_view_points[i, j, cam].append(data)
                    view_scores += view_result

                self.grid_views[i, j] = np.sum(view_scores) / len(points)

        self.points_ratio /= np.sum(self.points_ratio)

    def init_grid_occupancy(self, labels):
        for label in labels:
            x, y = label['3d_location']['x'], label['3d_location']['y']
            if self._in_map_bounds(x, y):
                i, j = self._grid_index(x, y)
                self.grid_occupancy[i, j] += 0.5
                if self.draw_map:
                    self.plot_grid((x, y), 'darkslategray')

    def update_grid_occupancy(self, label):
        x, y = label['3d_location']['x'], label['3d_location']['y']
        if self._in_map_bounds(x, y):
            i, j = self._grid_index(x, y)
            self.grid_occupancy[i, j] += 1
            if self.draw_map:
                self.plot_grid((x, y), 'salmon')

    def score(self, choose_num, save_path):
        k1, k2 = self.view_weight, self.occupancy_weight
        original_score = self.grid_score.copy()

        for i in range(self.W_num):
            for j in range(self.H_num):
                if len(self.grid_points[i, j]) == 0:
                    self.grid_score[i, j] = MIN
                else:
                    view_score = np.exp(k1 * self.grid_views[i, j])
                    occupancy_penalty = np.exp(k2 * self.grid_occupancy[i, j])
                    self.grid_score[i, j] += view_score / occupancy_penalty

        self.grid_order = np.argsort(-self.grid_score.flatten())

        if self.draw_map:
            self.save_grid_map(save_path, temp_save=True)

        self.grid_score = original_score

    def reset_map_scorer(self):
        self.grid_score = np.random.uniform(0, np.exp(self.view_weight), (self.W_num, self.H_num))
        self.grid_occupancy = np.zeros((self.W_num, self.H_num))
        self.grid_view_score = np.zeros((self.W_num, self.H_num, self.cam_num))

    def plot_grid(self, point=None, color='darkslategray', alpha=1):
        x_ticks = np.arange(self.map_size[0], self.map_size[1], self.grid_size[0])
        y_ticks = np.arange(self.map_size[2], self.map_size[3], self.grid_size[1])
        self.ax.set_xticks(x_ticks)
        self.ax.set_yticks(y_ticks)
        if point:
            self.ax.scatter(*point, c=color, alpha=alpha)

    def save_grid_map(self, save_path, temp_save=False):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        mask = (self.grid_score != MIN)
        if np.max(self.grid_score[mask]) != np.min(self.grid_score[mask]):
            normalized = self.grid_score / np.max(self.grid_score[mask])
        else:
            normalized = self.grid_score / np.max(self.grid_score)

        normalized[~mask] = 0

        x = np.arange(self.map_size[0], self.map_size[1] + self.grid_size[0], self.grid_size[0])
        y = np.arange(self.map_size[2], self.map_size[3] + self.grid_size[1], self.grid_size[1])

        patches = []
        for i in range(self.W_num):
            for j in range(self.H_num):
                score = math.sqrt(math.sqrt(normalized[i, j]))
                color = (score, score, 0)
                fill = self.ax.fill_between([x[i], x[i+1]], [y[j]]*2, [y[j+1]]*2, color=color, alpha=0.3)
                patches.append(fill)

        plt.savefig(save_path, bbox_inches='tight')
        print(f"[Saved] Grid heatmap: {save_path}")

        for fill in patches:
            fill.remove()

    def init_fig(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(self.map_size[0], self.map_size[1])
        self.ax.set_ylim(self.map_size[2], self.map_size[3])

        for i in range(self.W_num):
            for j in range(self.H_num):
                for data in self.grid_points[i, j]:
                    x, y = data['3d_location']['x'], data['3d_location']['y']
                    if self._in_map_bounds(x, y):
                        self.ax.scatter(x, y, c='green', alpha=0.1)
                        
    def _in_map_bounds(self, x, y):
        return (self.map_size[0] <= x <= self.map_size[1] and
                self.map_size[2] <= y <= self.map_size[3])

    def _grid_index(self, x, y):
        i = int((x - self.map_size[0]) / self.grid_size[0])
        j = int((y - self.map_size[2]) / self.grid_size[1])
        return i, j
