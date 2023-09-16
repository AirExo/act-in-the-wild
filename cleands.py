import os 
import numpy as np
from tqdm import tqdm

path = "/aidata/whole-body/task1-itw/"


def work(scene):
    scene_dir = os.path.join(path, scene)
    for file in sorted(os.listdir(scene_dir)):
        if os.path.splitext(file)[1] != '.npy':
            continue
        a = np.load(os.path.join(scene_dir, file), allow_pickle=True).item()
        robot_left = a['robot_left']
        robot_right = a['robot_right']
        robot_left = np.array(robot_left).astype(np.float64)
        robot_right = np.array(robot_right).astype(np.float64)
        a['robot_left'] = robot_left
        a['robot_right'] = robot_right
        np.save(os.path.join(scene_dir, file), a)


for scene_id in tqdm(range(611, 641)):
    work("scene{}".format(scene_id))
