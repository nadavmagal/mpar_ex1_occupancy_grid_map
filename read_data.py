import pykitti
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# TODO - good links - https://github.com/windowsub0406/KITTI_Tutorial

def read_kitti_data(basedir, date, drive):
    return


def main():
    basedir = r'/home/nadav/studies/mapping_and_perception_autonomous_robots/first_project/organized_data'
    date = '2011_09_26'
    drive = '0095'

    # The 'frames' argument is optional - default: None, which loads the whole dataset.
    # Calibration, timestamps, and IMU data are read automatically.
    # Camera and velodyne data are available via properties that create generators
    # when accessed, or through getter methods that provide random access.
    # The 'frames' argument is optional - default: None, which loads the whole dataset.
    # Calibration, timestamps, and IMU data are read automatically.
    # Camera and velodyne data are available via properties that create generators
    # when accessed, or through getter methods that provide random access.
    # data = pykitti.raw(basedir, date, drive, frames=range(0, 433, 1))
    # data = pykitti.raw(basedir, date, drive, frames=range(0, 433, 10))
    data = pykitti.raw(basedir, date, drive)

    # dataset.calib:         Calibration data are accessible as a named tuple
    # dataset.timestamps:    Timestamps are parsed into a list of datetime objects
    # dataset.oxts:          List of OXTS packets and 6-dof poses as named tuples
    # dataset.camN:          Returns a generator that loads individual images from camera N
    # dataset.get_camN(idx): Returns the image from camera N at idx
    # dataset.gray:          Returns a generator that loads monochrome stereo pairs (cam0, cam1)
    # dataset.get_gray(idx): Returns the monochrome stereo pair at idx
    # dataset.rgb:           Returns a generator that loads RGB stereo pairs (cam2, cam3)
    # dataset.get_rgb(idx):  Returns the RGB stereo pair at idx
    # dataset.velo:          Returns a generator that loads velodyne scans as [x,y,z,reflectance]
    # dataset.get_velo(idx): Returns the velodyne scan at idx

    point_velo = np.array([0, 0, 0, 1])
    point_cam0 = data.calib.T_cam0_velo.dot(point_velo)

    point_imu = np.array([0, 0, 0, 1])
    point_w = np.array([o.T_w_imu.dot(point_imu) for o in data.oxts])

    # plt.figure()
    # for ii, cam0_image in enumerate(data.cam0):
    #     img_np = np.array(cam0_image)
    #     img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    #     a=3
    #     plt.subplot(5,5,ii+1)
    #     plt.imshow(img_cv2)

    cam2_image, cam3_image = data.get_rgb(3)
    plt.figure()
    plt.subplot(1, 2, 2)
    plt.imshow(cam2_image)
    plt.subplot(1, 2, 1)
    plt.imshow(cam3_image)
    plt.show(block=False)
    a = 3

    fig = plt.figure()
    ax = Axes3D(fig)
    cmap = 'winter'
    # cmap = 'spring'
    colors = np.arange(len(data.oxts)-1, -1, -1)
    ax.scatter(point_w[:, 0], point_w[:, 1], point_w[:, 2], c=colors, cmap=cmap)
    plt.show(block=False)

    plt.figure()
    plt.scatter(point_w[:, 0], point_w[:, 1], c=colors, cmap=cmap)
    plt.show(block=False)
    a=3

    for ii, cur_velo in enumerate(data.velo):
        if ii >5:
            break




if __name__ == "__main__":
    main()
