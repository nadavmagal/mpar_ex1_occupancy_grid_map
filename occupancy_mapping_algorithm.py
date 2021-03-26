import pykitti
import matplotlib.pyplot as plt
import cv2
import numpy as np


def load_data(basedir, date, dataset_number):

    data = pykitti.raw(basedir, date, dataset_number)
    return data


def init_map(x_num_pix, y_num_pix, resolution):
    map = dict()
    map['x_size'] = x_num_pix
    map['y_size'] = y_num_pix
    map['resolution'] = resolution
    map['grid_size'] = x_num_pix * y_num_pix
    map['log_odds_prob'] = np.zeros((int(x_num_pix / resolution), int(y_num_pix / resolution)), order='C')
    map['log_occupied'] = 0.8
    map['log_free'] = 0.2
    map['curr_veh_pt'] = None
    return map

def plot_car_and_velo_coordinates(cur_car_coor, cur_velo_coor):
    fig = plt.figure()
    plt.axis('equal')
    plt.scatter(cur_velo_coor[:, 0], cur_velo_coor[:, 1], c='blue', s=10, edgecolors='none')
    plt.scatter(cur_car_coor[0], cur_car_coor[1], s=20, color='red')
    plt.grid()
    plt.show(block=False)


def create_occupancy_map(basedir, date, dataset_number, x_num_pix, y_num_pix, resolution):
    data = load_data(basedir, date, dataset_number)
    map = init_map(x_num_pix, y_num_pix, resolution)

    velo_vec = list(data.velo)
    point_imu = np.array([0, 0, 0, 1])
    center_offset = int(x_num_pix/2)
    car_coordinates = [o.T_w_imu.dot(point_imu) + center_offset for o in data.oxts]

    for ii, cur_velo, cur_car_coor in zip(range(len(car_coordinates)), velo_vec, car_coordinates):
        cur_velo_coor = cur_car_coor + cur_velo
        plot_car_and_velo_coordinates(cur_car_coor, cur_velo_coor)

        if ii == 1:
            break
        a=3

    a = 3


if __name__ == "__main__":
    basedir = '/home/nadav/studies/mapping_and_perception_autonomous_robots/first_project/organized_data'
    date = '2011_09_26'
    dataset_number = '0093'

    x_num_pix = 500
    y_num_pix = 500
    resolution = 0.1
    create_occupancy_map(basedir, date, dataset_number, x_num_pix, y_num_pix, resolution)
