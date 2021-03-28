import pykitti
import matplotlib.pyplot as plt
import cv2
import numpy as np

DEBUG = True
debug_velo_scatter_fig = plt.figure() if DEBUG else None

R_MAX_M = 30


def load_data(basedir, date, dataset_number):
    data = pykitti.raw(basedir, date, dataset_number)
    return data


class OccupancyMap:
    def __init__(self, x_size_m, y_size_m, resolution_cell_m):
        self.x_size_m = x_size_m
        self.y_size_m = y_size_m
        self.resolution = resolution_cell_m
        self.grid_size = (x_size_m / resolution_cell_m) * (y_size_m / resolution_cell_m)
        self.log_odds_prob = np.zeros((int(x_size_m / resolution_cell_m), int(y_size_m / resolution_cell_m)), order='C')
        self.log_occupied = 0.8
        self.log_free = 0.2

    def update_map_from_velo(self, cur_car_w_coor_m, cur_velo_w_coor_m):
        # TODO take only velo higher than ?[m]

        cur_car_grid_coor = cur_car_w_coor_m / self.resolution
        cur_velo_grid_coor = cur_velo_w_coor_m / self.resolution

        velo_grid = self._velo_point_cloude_to_map(cur_velo_grid_coor)

        for xx, yy in zip(range(np.shape(self.log_odds_prob)[0]), range(np.shape(self.log_odds_prob)[1])):
            self._inverse_range_sensor_model([xx, yy], cur_car_grid_coor, cur_velo_grid_coor)

            a = 3

        a = 3

    def _velo_point_cloude_to_map(self, cur_velo_grid_coor):
        velo_grid = np.zeros_like(self.log_odds_prob)
        for cur_velo in cur_velo_grid_coor:
            x = int(round(cur_velo[0]))
            y = int(round(cur_velo[1]))
            velo_grid[x, y] += 1
        if DEBUG:
            plt.figure()
            plt.title('velo_grid')
            plt.imshow(velo_grid)
            plt.show(block=False)
        return velo_grid

    def _inverse_range_sensor_model(self, cur_cell_2D, cur_car_grid_coor, cur_velo_grid_coor):
        cur_car_grid_coor_2D = np.array([cur_car_grid_coor[0], cur_car_grid_coor[1]])

        # r_m = np.linalg.norm(cur_car_grid_coor_2D - cur_cell_2D)*self.resolution
        # phi_deg = np.rad2deg(np.arctan2(cur_car_grid_coor_2D[1] - cur_cell_2D[1], cur_car_grid_coor_2D[0] - cur_cell_2D[0]))

        a = 3


def plot_cam_image(cur_cam2, cur_figure):
    plt.figure(cur_figure)
    plt.subplot(2, 1, 1)
    plt.imshow(cur_cam2)
    plt.show(block=False)


def plot_car_and_velo_coordinates(fig, cur_car_coor, cur_velo_coor):
    plt.figure(fig)
    plt.axis('equal')
    plt.title('velo data')
    plt.scatter(cur_velo_coor[:, 0], cur_velo_coor[:, 1], c='blue', s=10, edgecolors='none')
    plt.scatter(cur_car_coor[0], cur_car_coor[1], s=20, color='red')
    plt.grid(True)
    plt.show(block=False)


def velo_to_ned_coord(oxts, velo, car_coord):
    T = oxts.T_w_imu
    cur_transformed_velo = []
    for cur_point in velo:
        transformed_point = T.dot(cur_point) + car_coord
        cur_transformed_velo.append(np.array(transformed_point))

    return np.array(cur_transformed_velo)


def clip_far_velo_points(cur_velo_car_coor_m):
    clipped_cur_velo_car_coor_m = []
    for cur_velo_point in cur_velo_car_coor_m:
        if np.linalg.norm(cur_velo_point[0:3]) < R_MAX_M and cur_velo_point[3] > 0:
            clipped_cur_velo_car_coor_m.append(cur_velo_point)
    return np.array(clipped_cur_velo_car_coor_m)


def create_occupancy_map(basedir, date, dataset_number, x_size_m, y_size_m, resolution_cell_m, skip):
    data = load_data(basedir, date, dataset_number)
    oxts = data.oxts[::skip]
    cam2 = list(data.cam2)[::skip]
    velo = list(data.velo)[::skip]

    map = OccupancyMap(x_size_m, y_size_m, resolution_cell_m)

    velo_vec = velo
    point_imu = np.array([0, 0, 0, 1])
    center_offset_m = int(x_size_m / 2)
    car_w_coordinates_m = [o.T_w_imu.dot(point_imu) + center_offset_m for o in oxts]

    for ii, cur_velo_car_coor_m, cur_car_w_coor_m, cur_cam2, cur_oxts in zip(range(len(car_w_coordinates_m)), velo_vec,
                                                                             car_w_coordinates_m, cam2, oxts):
        cur_figure = plt.figure()
        plt.suptitle(f'sample #{ii}')

        clipped_cur_velo_car_coor_m = clip_far_velo_points(cur_velo_car_coor_m)

        cur_velo_w_coor_m = velo_to_ned_coord(cur_oxts, cur_velo_car_coor_m, cur_car_w_coor_m)

        if DEBUG:
            plot_car_and_velo_coordinates(debug_velo_scatter_fig, cur_car_w_coor_m, cur_velo_w_coor_m)

        map.update_map_from_velo(cur_car_w_coor_m, cur_velo_w_coor_m)

        plot_cam_image(cur_cam2, cur_figure)

        a = 3

    a = 3


if __name__ == "__main__":
    basedir = '/home/nadav/studies/mapping_and_perception_autonomous_robots/first_project/organized_data'
    date = '2011_09_26'
    # dataset_number = '0093' # mine
    # dataset_number = '0015'  # road
    dataset_number = '0005'# video

    skip = 20

    x_size_m = 500
    y_size_m = 500
    resolution_cell_m = 20 * 1e-2
    create_occupancy_map(basedir, date, dataset_number, x_size_m, y_size_m, resolution_cell_m, skip)
