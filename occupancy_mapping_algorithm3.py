import pykitti
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tqdm import trange, tqdm
import time
import os
import json
import copy

DEBUG_SAVE_FIG = False
DEBUG_SHOW = False
debug_velo_scatter_fig = plt.figure() if DEBUG_SHOW else None
debug_inverse_fig = plt.figure() if DEBUG_SHOW else None
debug_velo_grid_fig = plt.figure(figsize=[15, 10], dpi=500) if DEBUG_SAVE_FIG else None
debug_log_odds_prob_fig = plt.figure(figsize=[15, 10], dpi=500) if DEBUG_SAVE_FIG else None
Z_MAX_M = 30


def load_data(basedir, date, dataset_number):
    data = pykitti.raw(basedir, date, dataset_number)
    return data


def prob_to_logit(prob):
    return np.log(prob / (1 - prob))


class OccupancyMap:
    def __init__(self, x_size_m, y_size_m, resolution_cell_m):
        self.x_size_m = x_size_m
        self.y_size_m = y_size_m
        self.resolution = resolution_cell_m
        self.grid_size = (x_size_m / resolution_cell_m) * (y_size_m / resolution_cell_m)
        self.log_odds_prob = np.zeros((int(x_size_m / resolution_cell_m), int(y_size_m / resolution_cell_m)), order='C')
        self.log_occupied = prob_to_logit(0.7)
        self.log_free = prob_to_logit(0.4)
        self.log_0 = prob_to_logit(0.5)
        self.log_saturation_max = prob_to_logit(0.95)
        self.log_saturation_min = prob_to_logit(0.05)
        self.log_occupied_th = prob_to_logit(0.8)
        self.log_free_th = prob_to_logit(0.2)
        self.z_max = Z_MAX_M
        self.num_min_pillar_hits = 1  # TODO: change to 1
        self.min_delta_degree = 5

        self.debug_accumulate_velo_grid = np.zeros_like(self.log_odds_prob)

    def get_log_odds_prob(self):
        return self.log_odds_prob

    def update_map_from_velo(self, cur_car_w_coor_m, cur_velo_w_coor_m, result_dir_timed, ii):

        cur_car_grid_coor = cur_car_w_coor_m / self.resolution
        cur_velo_grid_coor = cur_velo_w_coor_m / self.resolution

        velo_grid_map = self._velo_point_cloude_to_map(cur_velo_grid_coor, result_dir_timed, ii)
        # velo_grid_map = np.rot90(velo_grid_map, k=0)
        # velo_grid_map = np.fliplr(velo_grid_map)
        # velo_grid_map = np.flipud(velo_grid_map)

        velo_angle_from_car, velo_grid_map_only_valid = self._calculate_velo_angles_from_car(cur_car_grid_coor,
                                                                                             velo_grid_map)


        xx_range = range(int(cur_car_grid_coor[0] - self.z_max / self.resolution),
                         int(cur_car_grid_coor[0] + self.z_max / self.resolution))
        yy_range = range(int(cur_car_grid_coor[1] - self.z_max / self.resolution),
                         int(cur_car_grid_coor[1] + self.z_max / self.resolution))
        for xx in tqdm(xx_range):
            for yy in yy_range:
                if xx < 0 or xx >= self.log_odds_prob.shape[0] or yy < 0 or yy >= self.log_odds_prob.shape[1]:
                    continue
                self.log_odds_prob[xx, yy] += self._inverse_range_sensor_model([xx, yy], cur_car_grid_coor,
                                                                               velo_grid_map, velo_angle_from_car,
                                                                               velo_grid_map_only_valid)
        self._saturate_values()

        if DEBUG_SHOW:
            plt.figure()
            plt.imshow(self.log_odds_prob)
            plt.show(block=False)

    def _calculate_velo_angles_from_car(self, cur_car_grid_coor, velo_grid_map):
        velo_angle_from_car = []
        velo_grid_map_only_valid = []
        for cur_velo in np.argwhere(velo_grid_map > 0):
            theta_cur_velo = np.rad2deg(
                np.arctan2(cur_car_grid_coor[1] - cur_velo[1], cur_car_grid_coor[0] - cur_velo[0]))
            velo_angle_from_car.append(theta_cur_velo)
            velo_grid_map_only_valid.append(cur_velo)
        velo_angle_from_car = np.array(velo_angle_from_car)
        velo_grid_map_only_valid = np.array(velo_grid_map_only_valid)
        return velo_angle_from_car, velo_grid_map_only_valid

    def _saturate_values(self):
        self.log_odds_prob = np.where(self.log_odds_prob > self.log_saturation_max, self.log_saturation_max,
                                      self.log_odds_prob)
        self.log_odds_prob = np.where(self.log_odds_prob < self.log_saturation_min, self.log_saturation_min,
                                      self.log_odds_prob)

    def _velo_point_cloude_to_map(self, cur_velo_grid_coor, result_dir_timed, ii):
        velo_grid = np.zeros_like(self.debug_accumulate_velo_grid)
        for cur_velo in cur_velo_grid_coor:
            x = int(round(cur_velo[0]))
            y = int(round(cur_velo[1]))
            if x < velo_grid.shape[0] and y < velo_grid.shape[1]:
                velo_grid[x, y] += 1
                self.debug_accumulate_velo_grid[x, y] += 1

        velo_grid = np.where(velo_grid > self.num_min_pillar_hits, 1, 0)

        if DEBUG_SAVE_FIG:
            plt.figure(debug_velo_grid_fig)
            plt.subplot(1, 2, 1)
            plt.title('cur velo_grid')
            plt.imshow(velo_grid)
            plt.subplot(1, 2, 2)
            plt.imshow(self.debug_accumulate_velo_grid)
            plt.title('accumulate velo grid')
            plt.savefig(os.path.join(result_dir_timed, f'debug_accumulate_velo_grid_{ii}.png'))
            # plt.show(block=False)
        return velo_grid

    def _get_index_of_closest_angular_velo_coord(self, phi_deg, cur_velo_grid_coor, cur_car_grid_coor_2D):
        min_delta = np.inf
        index_of_min_delta = None
        for ii, cur_velo in enumerate(cur_velo_grid_coor):
            theta_cur_velo = np.rad2deg(
                np.arctan2(cur_car_grid_coor_2D[1] - cur_velo[1], cur_car_grid_coor_2D[0] - cur_velo[0]))
            if np.abs(theta_cur_velo - phi_deg) < min_delta:
                min_delta = np.abs(theta_cur_velo - phi_deg)
                index_of_min_delta = ii
        return cur_velo_grid_coor[index_of_min_delta]

    def _get_index_of_closest_angular_velo_map(self, phi_deg, velo_grid_map, cur_car_grid_coor_2D):
        min_delta = np.inf
        index_of_min_delta = None
        for cur_velo in np.argwhere(velo_grid_map > 0):
            theta_cur_velo = np.rad2deg(
                np.arctan2(cur_car_grid_coor_2D[1] - cur_velo[1], cur_car_grid_coor_2D[0] - cur_velo[0]))
            if np.abs(theta_cur_velo - phi_deg) < min_delta:
                min_delta = np.abs(theta_cur_velo - phi_deg)
                index_of_min_delta = cur_velo
        return index_of_min_delta

    def _get_index_of_closest_angular_velo_map_at_farther_distance(self, phi_deg, velo_grid_map, cur_car_grid_coor_2D,
                                                                   velo_angle_from_car, velo_grid_map_only_valid):
        delta_angle_vec = []
        coord_of_min_delta_angle = []

        delta_phi_velo_angle = abs(velo_angle_from_car - phi_deg)
        indices_of_delta_smaller_than_th = np.where(delta_phi_velo_angle < self.min_delta_degree)[0]
        velo_coord_smaller_than_th = velo_grid_map_only_valid[indices_of_delta_smaller_than_th]
        #
        # for cur_velo in np.argwhere(velo_grid_map > 0):
        #     theta_cur_velo = np.rad2deg(
        #         np.arctan2(cur_car_grid_coor_2D[1] - cur_velo[1], cur_car_grid_coor_2D[0] - cur_velo[0]))
        if DEBUG_SHOW:
            plt.figure(debug_inverse_fig)
            plt.scatter(velo_grid_map_only_valid[:, 0], velo_grid_map_only_valid[:, 1], marker='x', color='green')

            # if abs(theta_cur_velo - phi_deg) < self.min_delta_degree:
            #     delta_angle_vec.append(abs(theta_cur_velo))
            #     coord_of_min_delta_angle.append(cur_velo)

        index_of_min_delta = None
        max_r = 0
        for cur_velo in velo_coord_smaller_than_th:
            cur_r = np.linalg.norm(cur_velo - cur_car_grid_coor_2D)
            if cur_r < 30 / self.resolution and cur_r > max_r:
                max_r = cur_r
                index_of_min_delta = cur_velo

        return index_of_min_delta

    def _inverse_range_sensor_model(self, cur_cell_2D, cur_car_grid_coor, velo_grid_map, velo_angle_from_car,
                                    velo_grid_map_only_valid):
        cur_car_grid_coor_2D = np.array([cur_car_grid_coor[0], cur_car_grid_coor[1]])

        r_cell_m = np.linalg.norm(cur_car_grid_coor_2D - cur_cell_2D) * self.resolution
        phi_deg = np.rad2deg(
            np.arctan2(cur_car_grid_coor_2D[1] - cur_cell_2D[1], cur_car_grid_coor_2D[0] - cur_cell_2D[0]))

        if DEBUG_SHOW:
            plt.figure(debug_inverse_fig)
            plt.imshow(velo_grid_map)
            plt.scatter(cur_cell_2D[0], cur_cell_2D[1], marker='x', color='red', label='cell')
            plt.scatter(cur_car_grid_coor[0], cur_car_grid_coor[1], color='blue', label='car')
            plt.plot([cur_cell_2D[0], cur_car_grid_coor[0]], [cur_cell_2D[1], cur_car_grid_coor[1]], color='orange')
            plt.title(f'phi={round(phi_deg, 2)} deg')
            plt.legend()
            plt.show(block=False)

        # closest_velo = self._get_index_of_closest_angular_velo_coord(phi_deg, cur_velo_grid_coor, cur_car_grid_coor_2D)
        # closest_velo = self._get_index_of_closest_angular_velo_map(phi_deg, velo_grid_map, cur_car_grid_coor_2D)
        closest_velo = self._get_index_of_closest_angular_velo_map_at_farther_distance(phi_deg, velo_grid_map,
                                                                                       cur_car_grid_coor_2D,
                                                                                       velo_angle_from_car,
                                                                                       velo_grid_map_only_valid)

        if closest_velo is None:
            return self.log_0
        r_z_k_m = np.linalg.norm(cur_car_grid_coor_2D - closest_velo[0:2]) * self.resolution

        if r_cell_m > np.min([self.z_max, r_z_k_m]):  # ignoring second condition as the tutor explained
            return self.log_0
        elif r_z_k_m < self.z_max and np.abs(r_cell_m - r_z_k_m) < self.resolution / 2:
            return self.log_occupied
        elif r_cell_m < r_z_k_m:
            return self.log_free
        else:
            raise Exception("error")

    def get_occupancy_map(self):
        occupancy_map = np.where(self.log_odds_prob > self.log_occupied_th, 0, 0.5)
        occupancy_map = np.where(self.log_odds_prob < self.log_free_th, 1, occupancy_map)
        return occupancy_map


def plot_figure(cur_cam2, cur_occupancy_map, cur_velo_car_coor_m, cur_car_w_coor_m, ii, result_dir_timed):
    cur_figure = plt.figure(figsize=[15, 10], dpi=500)
    plt.suptitle(f'sample #{ii}')
    plt.figure(cur_figure)
    plt.subplot(2, 1, 1)
    plt.imshow(cur_cam2)

    plt.subplot(2, 2, 3)
    resolution = 0.2
    velo_grid = np.zeros([int(70 / resolution), int(70 / resolution)])
    for cur_velo in cur_velo_car_coor_m:
        x = int(round(cur_velo[0] / resolution) + velo_grid.shape[0] / 2)
        y = int(round(cur_velo[1] / resolution) + velo_grid.shape[0] / 2)
        if x < velo_grid.shape[0] and y < velo_grid.shape[1]:
            velo_grid[x, y] += 1
    velo_grid = np.where(velo_grid > 0, 0, 1)

    plt.imshow(velo_grid, cmap='gray')

    plt.subplot(2, 2, 4)
    plt.imshow(cur_occupancy_map, cmap='gray')
    plt.scatter(cur_car_w_coor_m[0] / resolution, cur_car_w_coor_m[1] / resolution, marker='x', color='red')

    plt.savefig(os.path.join(result_dir_timed, f'occ_map_{ii}.png'))
    # plt.close('all')
    # plt.show(block=False)


def plot_car_and_velo_coordinates(fig, cur_car_coor, cur_velo_coor):
    plt.figure(fig)
    plt.axis('equal')
    plt.title('velo data')
    plt.scatter(cur_velo_coor[:, 0], cur_velo_coor[:, 1], c='blue', s=10, edgecolors='none')
    plt.scatter(cur_car_coor[0], cur_car_coor[1], s=20, color='red')
    plt.grid(True)
    plt.show(block=False)

def transform_single_velo_point(velo_point, T_velo_imu, T_w_imu, car_coord):
    # transformed_velo_point = T_velo_imu.dot(velo_point)
    # transformed_velo_point = T_w_imu.dot(transformed_velo_point) + car_coord
    transformed_velo_point = T_velo_imu.dot(velo_point)
    transformed_velo_point = T_w_imu.dot(transformed_velo_point) + car_coord
    return transformed_velo_point

def transform_single_velo_point_identity_and_translation(velo_point, T_velo_imu, T_w_imu, car_coord):
    return velo_point + car_coord

def velo_to_ned_coord(oxts, velo, car_coord, T_velo_imu):
    T_w_imu = oxts.T_w_imu
    cur_transformed_velo = []
    for cur_point in velo:
        transformed_point = transform_single_velo_point_identity_and_translation(cur_point, T_velo_imu, T_w_imu, car_coord)
        cur_transformed_velo.append(np.array(transformed_point))

    center_of_velo = transform_single_velo_point_identity_and_translation(np.array([0, 0, 0, 1]), T_velo_imu, T_w_imu, car_coord)

    return np.array(cur_transformed_velo), center_of_velo


def clip_far_velo_points(cur_velo_car_coor_m):
    clipped_cur_velo_car_coor_m = []
    for cur_velo_point in cur_velo_car_coor_m:
        if np.linalg.norm(cur_velo_point[0:3]) < Z_MAX_M and cur_velo_point[3] > 0:
            clipped_cur_velo_car_coor_m.append(cur_velo_point)
    return np.array(clipped_cur_velo_car_coor_m)


def create_occupancy_map(basedir, date, dataset_number, x_size_m, y_size_m, resolution_cell_m, skip_frames, skip_velo,
                         result_dir_timed):
    data = load_data(basedir, date, dataset_number)
    oxts = data.oxts[::skip_frames]
    cam2 = list(data.cam2)[::skip_frames]
    velo = list(data.velo)[::skip_frames]
    T_velo_imu = data.calib.T_velo_imu

    map = OccupancyMap(x_size_m, y_size_m, resolution_cell_m)

    point_imu = np.array([0, 0, 0, 1])
    center_offset_m = int(x_size_m / 2)

    car_w_coordinates_m = [o.T_w_imu.dot(point_imu) + center_offset_m for o in oxts]
    # car_w_coordinates_m = []
    # for cur_o in oxts:
    #     cur_car_w_coord = cur_o.T_w_imu.dot(point_imu)
    #     cur_car_w_coord = cur_car_w_coord/cur_car_w_coord[3] + center_offset_m
    #     car_w_coordinates_m.append(cur_car_w_coord)

    for ii, cur_velo_car_coor_m, cur_car_w_coor_m, cur_cam2, cur_oxts in zip(range(len(car_w_coordinates_m)), velo,
                                                                             car_w_coordinates_m, cam2, oxts):
        cur_velo_car_coor_m = cur_velo_car_coor_m[::skip_velo]
        # TODO - add to clip only points higher than 30 cm - velo[2]>0.3
        # TODO 2 - add this:
        # missing data.calib.T_velo_imu
        # TODO 3 - add min angle from 5 to 0.5 degrees


        cur_car_w_coor_m = T_velo_imu.dot(cur_car_w_coor_m)

        clipped_cur_velo_car_coor_m = clip_far_velo_points(cur_velo_car_coor_m)

        cur_velo_w_coor_m, center_of_velo_w_coor_m = velo_to_ned_coord(cur_oxts, clipped_cur_velo_car_coor_m, cur_car_w_coor_m, T_velo_imu)


        if DEBUG_SHOW:
            plot_car_and_velo_coordinates(debug_velo_scatter_fig, cur_car_w_coor_m, cur_velo_w_coor_m)
        if True:
            plt.figure()
            plt.scatter(cur_velo_w_coor_m[:, 0], cur_velo_w_coor_m[:, 1])
            plt.scatter(center_of_velo_w_coor_m[0], center_of_velo_w_coor_m[1], color='green')
            plt.scatter(cur_car_w_coor_m[0], cur_car_w_coor_m[1], marker='x', color='red')
            plt.show(block=False)
            continue
            # TODO: here!! car is not in the right place...

        map.update_map_from_velo(cur_car_w_coor_m, cur_velo_w_coor_m, result_dir_timed, ii)
        cur_occupancy_map = map.get_occupancy_map()
        if DEBUG_SAVE_FIG:
            cur_log_odds_prob = map.get_log_odds_prob()
            plt.figure(debug_log_odds_prob_fig)
            plt.imshow(cur_log_odds_prob)
            plt.savefig(os.path.join(result_dir_timed, f'debug_log_odds_prob_{ii}.png'))

        plot_figure(cur_cam2, cur_occupancy_map, cur_velo_car_coor_m, cur_car_w_coor_m, ii,
                    result_dir_timed)

        plt.close('all')


if __name__ == "__main__":
    basedir = '/home/nadav/studies/mapping_and_perception_autonomous_robots/first_project/organized_data'
    date = '2011_09_26'
    # dataset_number = '0093' # mine
    # dataset_number = '0015'  # road
    dataset_number = '0005'  # video

    result_dir = r'/home/nadav/studies/mapping_and_perception_autonomous_robots/first_project/results'
    cur_date_time = time.strftime("%Y.%m.%d-%H.%M")

    skip_frames = 10  # TODO: in advanced frame number we need to enlarge the y_size,x_size
    skip_velo = 30

    x_size_m = 100
    y_size_m = 100
    resolution_cell_m = 20 * 1e-2

    result_dir_timed = os.path.join(result_dir, f'{cur_date_time}_skipvelo_{skip_velo}')
    print(f'saving to: {result_dir_timed}')
    os.makedirs(result_dir_timed, exist_ok=True)
    create_occupancy_map(basedir, date, dataset_number, x_size_m, y_size_m, resolution_cell_m, skip_frames, skip_velo,
                         result_dir_timed)
