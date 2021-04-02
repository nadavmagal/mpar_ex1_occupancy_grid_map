import pykitti
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tqdm import trange, tqdm
import time
import os
import json
import copy
from frames_to_video import create_video

DEBUG_SAVE_FIG = False
DEBUG_SHOW = False
debug_velo_scatter_fig = plt.figure() if DEBUG_SHOW else None
debug_inverse_fig = plt.figure() if DEBUG_SHOW else None
debug_velo_grid_fig = plt.figure(figsize=[15, 10], dpi=500) if DEBUG_SAVE_FIG else None
debug_log_odds_prob_fig = plt.figure(figsize=[15, 10], dpi=500) if DEBUG_SAVE_FIG else None
Z_MAX_M = 60


def load_data(basedir, date, dataset_number):
    data = pykitti.raw(basedir, date, dataset_number)
    return data


def prob_to_logit(prob):
    return np.log(prob / (1 - prob))


class OccupancyMap:
    def __init__(self, x_size_m, y_size_m, resolution_cell_m, occ_params):
        self.x_size_m = x_size_m
        self.y_size_m = y_size_m
        self.resolution = resolution_cell_m
        self.grid_size = (x_size_m / resolution_cell_m) * (y_size_m / resolution_cell_m)
        self.log_odds_prob = np.zeros((int(x_size_m / resolution_cell_m), int(y_size_m / resolution_cell_m)), order='C')
        self.log_occupied = prob_to_logit(occ_params['p_hit'])
        self.log_free = prob_to_logit(occ_params['p_miss'])
        self.log_0 = prob_to_logit(0.5)
        self.log_saturation_max = prob_to_logit(0.95)
        self.log_saturation_min = prob_to_logit(0.05)
        self.log_occupied_th = prob_to_logit(occ_params['occ_th'])
        self.log_free_th = prob_to_logit(occ_params['free_th'])
        self.z_max = Z_MAX_M
        self.num_min_pillar_hits = 1
        self.min_delta_degree = 1

        self.debug_accumulate_velo_grid = np.zeros_like(self.log_odds_prob)

    def get_log_odds_prob(self):
        return self.log_odds_prob

    def update_map_from_velo(self, cur_car_w_coor_m, cur_velo_car_coor_m, result_dir_timed, ii, cur_oxts):

        cur_car_w_coor_m_with_resolution = cur_car_w_coor_m / self.resolution
        cur_velo_car_coor_m_with_resolution = cur_velo_car_coor_m / self.resolution
        cur_velo_translated_only_w_coord_m_with_resolution = cur_velo_car_coor_m / self.resolution
        cur_velo_translated_only_w_coord_m_with_resolution[:, 0] += cur_car_w_coor_m_with_resolution[0]
        cur_velo_translated_only_w_coord_m_with_resolution[:, 1] += cur_car_w_coor_m_with_resolution[1]

        if False:
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.scatter(cur_velo_car_coor_m_with_resolution[:, 0], cur_velo_car_coor_m_with_resolution[:, 1])
            plt.scatter(cur_car_w_coor_m_with_resolution[0], cur_car_w_coor_m_with_resolution[1])

            plt.subplot(1, 2, 2)
            plt.scatter(cur_velo_translated_only_w_coord_m_with_resolution[:, 0],
                        cur_velo_translated_only_w_coord_m_with_resolution[:, 1])
            plt.scatter(cur_car_w_coor_m_with_resolution[0], cur_car_w_coor_m_with_resolution[1])
            plt.show(block=False)

        velo_grid_map = self._velo_point_cloude_to_map(cur_velo_translated_only_w_coord_m_with_resolution,
                                                       result_dir_timed, ii)
        velo_grid_map = velo_grid_map.T
        if False:
            plt.figure()
            plt.imshow(velo_grid_map)
            plt.scatter(cur_car_w_coor_m_with_resolution[0], cur_car_w_coor_m_with_resolution[1])
            plt.show(block=False)

        velo_angle_from_car, velo_grid_map_only_valid, velo_distance_from_car_only_valid = self._calculate_velo_angles_from_car(
            cur_car_w_coor_m_with_resolution,
            velo_grid_map, cur_oxts)  # TODO - LEFAKPEK THIS FUNCTION

        if False:
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(velo_grid_map)
            plt.scatter(cur_car_w_coor_m_with_resolution[0], cur_car_w_coor_m_with_resolution[1])
            plt.subplot(1, 2, 2)
            plt.imshow(velo_grid_map)
            plt.scatter(velo_grid_map_only_valid[:, 0], velo_grid_map_only_valid[:, 1], s=1, color='red')
            plt.scatter(cur_car_w_coor_m_with_resolution[0], cur_car_w_coor_m_with_resolution[1])
            plt.show(block=False)
        xx_range = range(int(cur_car_w_coor_m_with_resolution[0] - self.z_max / self.resolution),
                         int(cur_car_w_coor_m_with_resolution[0] + self.z_max / self.resolution))
        yy_range = range(int(cur_car_w_coor_m_with_resolution[1] - self.z_max / self.resolution),
                         int(cur_car_w_coor_m_with_resolution[1] + self.z_max / self.resolution))
        for xx in tqdm(xx_range):
            for yy in yy_range:
                if xx < 0 or xx >= self.log_odds_prob.shape[0] or yy < 0 or yy >= self.log_odds_prob.shape[1]:
                    continue
                self.log_odds_prob[xx, yy] += self._inverse_range_sensor_model([xx, yy],
                                                                               cur_car_w_coor_m_with_resolution,
                                                                               velo_grid_map, velo_angle_from_car,
                                                                               velo_grid_map_only_valid,
                                                                               velo_distance_from_car_only_valid,
                                                                               cur_oxts)
        self._saturate_values()
        a = 3
        if False:
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(np.flip(np.rot90(self.log_odds_prob, k=3)))
            plt.title('log_odds_prob')
            plt.subplot(1, 2, 2)
            plt.imshow(velo_grid_map)
            plt.title('velo_grid_map')
            plt.show(block=False)

    def _calculate_velo_angles_from_car(self, cur_car_w_coor_m_with_resolution, velo_grid_map, cur_oxts):
        velo_angle_from_car = []
        velo_grid_map_only_valid = []
        velo_distance_from_car_only_valid = []
        for cur_velo in np.argwhere(velo_grid_map > 0):
            cur_velo = np.flip(cur_velo)
            theta_cur_velo = np.rad2deg(
                np.arctan2(cur_car_w_coor_m_with_resolution[1] - cur_velo[1],
                           cur_car_w_coor_m_with_resolution[0] - cur_velo[0]))
            velo_angle_from_car.append(theta_cur_velo)
            velo_grid_map_only_valid.append(cur_velo)
            velo_distance_from_car_only_valid.append(np.linalg.norm(cur_velo - cur_car_w_coor_m_with_resolution[0:2]))
        velo_angle_from_car = np.array(velo_angle_from_car)
        velo_grid_map_only_valid = np.array(velo_grid_map_only_valid)
        velo_distance_from_car_only_valid = np.array(velo_distance_from_car_only_valid)
        return velo_angle_from_car, velo_grid_map_only_valid, velo_distance_from_car_only_valid

    def _saturate_values(self):
        self.log_odds_prob = np.where(self.log_odds_prob > self.log_saturation_max, self.log_saturation_max,
                                      self.log_odds_prob)
        self.log_odds_prob = np.where(self.log_odds_prob < self.log_saturation_min, self.log_saturation_min,
                                      self.log_odds_prob)

    def _velo_point_cloude_to_map(self, cur_velo_translated_only_w_coord_m_with_resolution, result_dir_timed, ii):
        velo_grid = np.zeros_like(self.debug_accumulate_velo_grid)
        for cur_velo in cur_velo_translated_only_w_coord_m_with_resolution:
            x = int(round(cur_velo[0]))
            y = int(round(cur_velo[1]))
            if x < velo_grid.shape[0] and y < velo_grid.shape[1]:
                velo_grid[x, y] += 1
                # self.debug_accumulate_velo_grid[x, y] += 1

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
                                                                   velo_angle_from_car, velo_grid_map_only_valid,
                                                                   velo_distance_from_car_only_valid, cur_oxts):

        yaw_deg = np.rad2deg(cur_oxts.packet.yaw)
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
                                    velo_grid_map_only_valid, velo_distance_from_car_only_valid, cur_oxts):
        cur_car_grid_coor_2D = np.array([cur_car_grid_coor[0], cur_car_grid_coor[1]])

        r_cell_m = np.linalg.norm(cur_car_grid_coor_2D - cur_cell_2D) * self.resolution
        phi_deg = np.rad2deg(
            np.arctan2(cur_car_grid_coor_2D[1] - cur_cell_2D[1], cur_car_grid_coor_2D[0] - cur_cell_2D[0]))

        if False:
            plt.figure(debug_inverse_fig)
            plt.imshow(velo_grid_map)
            plt.scatter(cur_cell_2D[0], cur_cell_2D[1], marker='x', color='red', label='cell')
            plt.scatter(cur_car_grid_coor[0], cur_car_grid_coor[1], color='blue', label='car')
            plt.plot([cur_cell_2D[0], cur_car_grid_coor[0]], [cur_cell_2D[1], cur_car_grid_coor[1]], color='orange')
            plt.title(f'phi={round(phi_deg, 2)} deg')
            plt.legend()
            plt.show(block=False)
        closest_velo = self._get_index_of_closest_angular_velo_map_at_farther_distance(phi_deg, velo_grid_map,
                                                                                       cur_car_grid_coor_2D,
                                                                                       velo_angle_from_car,
                                                                                       velo_grid_map_only_valid,
                                                                                       velo_distance_from_car_only_valid,
                                                                                       cur_oxts)

        a = 3
        if False:
            plt.figure(debug_inverse_fig)
            plt.imshow(velo_grid_map)
            plt.scatter(cur_cell_2D[0], cur_cell_2D[1], marker='x', color='red', label='cell')
            plt.scatter(cur_car_grid_coor[0], cur_car_grid_coor[1], color='blue', label='car')
            plt.scatter(closest_velo[0], closest_velo[1], color='magenta', marker='x', label='closest_velo')
            plt.plot([cur_cell_2D[0], cur_car_grid_coor[0]], [cur_cell_2D[1], cur_car_grid_coor[1]], color='orange')
            plt.title(f'phi={round(phi_deg, 2)} deg')
            plt.legend()
            plt.show(block=False)

        if closest_velo is None and r_cell_m < self.z_max:
            return self.log_free

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


def plot_figure(cur_cam2, cur_occupancy_map, cur_velo_car_coor_m, cur_car_w_coor_m, accumulate_car_coor_m, ii,
                result_dir_timed, results_fig, x_size_m, y_size_m):
    plt.figure(results_fig)
    plt.clf()
    plt.suptitle(f'sample #{ii}')
    plt.subplot(2, 1, 1)
    plt.imshow(cur_cam2)
    plt.axis('off')
    plt.title('Scene Image')

    plt.subplot(2, 2, 3)
    # plt.figure()
    resolution = 0.2
    velo_grid = np.zeros([int(70 / resolution), int(70 / resolution)])
    for cur_velo in cur_velo_car_coor_m:
        x = int(round(cur_velo[0] / resolution) + velo_grid.shape[0] / 2)
        y = int(round(cur_velo[1] / resolution) + velo_grid.shape[0] / 2)
        if x < velo_grid.shape[0] and y < velo_grid.shape[1]:
            velo_grid[x, y] += 1
    velo_grid = np.where(velo_grid > 0, 0, 1)
    x_axis_size = velo_grid.shape[0]
    y_axis_size = velo_grid.shape[1]
    plt.imshow(velo_grid, cmap='gray')
    plt.title('Instantaneous Point Cloud')
    plt.xticks(range(0, x_axis_size+1, int(x_axis_size / 2)),
               np.round(np.linspace(-x_size_m / 2, x_size_m / 2, len(range(0, x_axis_size+1, int(x_axis_size / 2)))), 2))
    plt.yticks(range(0, y_axis_size+1, int(y_axis_size / 2)),
               np.round(np.linspace(-y_size_m / 2, y_size_m / 2, len(range(0, y_axis_size+1, int(x_axis_size / 2)))), 2))
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    # plt.show(block=False)

    plt.subplot(2, 2, 4)
    plt.imshow(cur_occupancy_map, cmap='gray')  # TODO what to do here?? # TODO stopped here -> reflection in 93. wht
    accumulate_car_coor_m = np.array(accumulate_car_coor_m)
    plt.scatter(accumulate_car_coor_m[:, 1] / resolution, accumulate_car_coor_m[:, 0] / resolution, s=0.5, color='blue')
    plt.scatter(cur_car_w_coor_m[1] / resolution, cur_car_w_coor_m[0] / resolution, marker='x', color='red')
    plt.title('Occupancy Map')
    x_axis_size = cur_occupancy_map.shape[0]
    y_axis_size = cur_occupancy_map.shape[1]
    plt.xticks(range(0, x_axis_size+1, int(x_axis_size / 2)),
               np.round(np.linspace(-x_size_m / 2, x_size_m / 2, len(range(0, x_axis_size+1, int(x_axis_size / 2)))), 2))
    plt.yticks(range(0, y_axis_size+1, int(y_axis_size / 2)),
               np.round(np.linspace(-y_size_m / 2, y_size_m / 2, len(range(0, y_axis_size+1, int(x_axis_size / 2)))), 2))
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')

    plt.savefig(os.path.join(result_dir_timed, f'occ_map_{ii}.png'))
    plt.close('all')
    # plt.show(block=False)


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
        transformed_point = transform_single_velo_point(cur_point, T_velo_imu, T_w_imu, car_coord)
        cur_transformed_velo.append(np.array(transformed_point))

    center_of_velo = transform_single_velo_point(np.array([0, 0, 0, 1]), T_velo_imu, T_w_imu, car_coord)

    return np.array(cur_transformed_velo), center_of_velo


def clip_far_velo_points(cur_velo_car_coor_m):
    clipped_cur_velo_car_coor_m = []
    for cur_velo_point in cur_velo_car_coor_m:
        if np.linalg.norm(cur_velo_point[0:3]) < Z_MAX_M:  # and cur_velo_point[3] > 0:
            clipped_cur_velo_car_coor_m.append(cur_velo_point)
    return np.array(clipped_cur_velo_car_coor_m)


def rotate_velo_by_yaw(cur_clipped_velo_car_coor_m, yaw):
    rotation_mat = np.array([[np.cos(yaw), -np.sin(yaw)],
                             [np.sin(yaw), np.cos(yaw)]])
    rotated_velos = []
    for ii, cur_velo in enumerate(cur_clipped_velo_car_coor_m):
        cur_x_y = np.array([cur_velo[0], cur_velo[1]])
        cur_rot_velo = np.matmul(rotation_mat, cur_x_y)
        rotated_velos.append(np.array([cur_rot_velo[0], cur_rot_velo[1], cur_velo[2], cur_velo[3]]))

    return np.array(rotated_velos)


def create_occupancy_map(basedir, date, dataset_number, x_size_m, y_size_m, resolution_cell_m, skip_frames, skip_velo,
                         result_dir_timed, start_frame, occ_params):
    data = load_data(basedir, date, dataset_number)
    oxts = data.oxts[::skip_frames]
    cam2 = list(data.cam2)[::skip_frames]
    velo = list(data.velo)[::skip_frames]
    T_velo_imu = data.calib.T_velo_imu

    my_map = OccupancyMap(x_size_m, y_size_m, resolution_cell_m, occ_params)

    point_imu = np.array([0, 0, 0, 1])
    center_offset_m = int(x_size_m / 2)

    car_w_coordinates_m = [o.T_w_imu.dot(point_imu) + center_offset_m for o in oxts]
    accumulate_car_coor_m = []
    results_fig = plt.figure(figsize=[15, 10], dpi=300)
    for ii, cur_velo_car_coor_m, cur_car_w_coor_m, cur_cam2, cur_oxts in zip(range(len(car_w_coordinates_m)), velo,
                                                                             car_w_coordinates_m, cam2, oxts):
        if ii < start_frame:
            continue

        transformations = {
            'T_velo_imu': T_velo_imu,
            'T_w_imu': cur_oxts.T_w_imu
        }
        cur_velo_car_coor_m = cur_velo_car_coor_m[::skip_velo]
        cur_clipped_velo_car_coor_m = clip_far_velo_points(cur_velo_car_coor_m)

        cur_clipped_velo_car_coor_m = rotate_velo_by_yaw(cur_clipped_velo_car_coor_m, cur_oxts.packet.yaw)

        if False:
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.scatter(cur_clipped_velo_car_coor_m[:, 0], cur_clipped_velo_car_coor_m[:, 1])
            plt.scatter(cur_car_w_coor_m[0], cur_car_w_coor_m[1])

            plt.subplot(1, 2, 2)
            plt.scatter(cur_clipped_velo_car_coor_m[:, 0] + cur_car_w_coor_m[0],
                        cur_clipped_velo_car_coor_m[:, 1] + cur_car_w_coor_m[1])
            plt.scatter(cur_car_w_coor_m[0], cur_car_w_coor_m[1])

        my_map.update_map_from_velo(cur_car_w_coor_m, cur_clipped_velo_car_coor_m, result_dir_timed, ii, cur_oxts)

        cur_occupancy_map = my_map.get_occupancy_map()
        if DEBUG_SAVE_FIG:
            cur_log_odds_prob = my_map.get_log_odds_prob()
            plt.figure(debug_log_odds_prob_fig)
            plt.imshow(cur_log_odds_prob)
            plt.savefig(os.path.join(result_dir_timed, f'debug_log_odds_prob_{ii}.png'))

        accumulate_car_coor_m.append(cur_car_w_coor_m)
        plot_figure(cur_cam2, cur_occupancy_map, cur_velo_car_coor_m, cur_car_w_coor_m, accumulate_car_coor_m, ii,
                    result_dir_timed, results_fig, x_size_m, y_size_m)

    a = 3


if __name__ == "__main__":
    basedir = '/home/nadav/studies/mapping_and_perception_autonomous_robots/first_project/organized_data'
    date = '2011_09_26'
    # dataset_number = '0093'  # old
    # dataset_number = '0095'  # mine
    # dataset_number = '0015'  # road
    dataset_number = '0005'  # video

    result_dir = r'/home/nadav/studies/mapping_and_perception_autonomous_robots/first_project/results'
    cur_date_time = time.strftime("%Y.%m.%d-%H.%M")

    start_frame = 0
    skip_frames = 1
    skip_velo = 20

    x_size_m = 500
    y_size_m = 500
    resolution_cell_m = 20 * 1e-2

    occ_params = {
        'p_hit': 0.7, # default 0.7
        'p_miss': 0.4, # default 0.4
        'occ_th': 0.8, # default 0.8
        'free_th': 0.2} # default 0.2 # TODO - change 0.5!!!

    result_dir_timed = os.path.join(result_dir, f'{cur_date_time}_{dataset_number}_skipvelo_{skip_velo}_{occ_params}')
    print(f'saving to: {result_dir_timed}')
    os.makedirs(result_dir_timed, exist_ok=True)
    create_occupancy_map(basedir, date, dataset_number, x_size_m, y_size_m, resolution_cell_m, skip_frames, skip_velo,
                         result_dir_timed, start_frame, occ_params)
    create_video(result_dir_timed)
