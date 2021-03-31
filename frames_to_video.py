import cv2
import os
from tqdm import tqdm

def create_video(input_dir):
    print('creating video')
    if os.path.isfile(os.path.join(input_dir, 'output_video.avi')):
        os.remove(os.path.join(input_dir, 'output_video.avi'))

    frames_fp = [os.path.join(input_dir, cur_name) for cur_name in os.listdir(input_dir)]

    frames_fp.sort(key=lambda x:int(os.path.basename(x).split('_')[2].split('.')[0]))

    height, width, channels = cv2.imread(frames_fp[0]).shape
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    out = cv2.VideoWriter(os.path.join(input_dir, 'output_video.avi'), fourcc, 10, (width, height), True)

    for filename in tqdm(frames_fp):
        img = cv2.imread(filename)
        out.write(img)
    print('Done!')
    out.release()

if __name__ == "__main__":
    input_dir = r'/home/nadav/studies/mapping_and_perception_autonomous_robots/first_project/results/2021.03.31-09.27_skipvelo_20'
    create_video(input_dir)
