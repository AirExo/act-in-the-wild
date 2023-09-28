"""
Whole-body Dataset.

Device Settings: Dual Flexiv Rizon arms; Robotiq 2F-85 gripper.
"""
import os
import json
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as T

from tqdm import tqdm
from torch.utils.data import Dataset
from easydict import EasyDict as edict
from transforms3d.euler import quat2euler


def sample_timestamps(timestamps: list, start_time: int, stop_time: int, time_per_sample: float = 0.0):
    """
    Sample timestamps according to [start_time, stop_time] and time per sample.
    """
    res = []
    last_ts = 0
    for ts in sorted(timestamps):
        if ts < start_time:
            continue
        if ts > stop_time:
            break
        if ts - last_ts >= time_per_sample * 1000:
            res.append(ts)
            last_ts = ts
    return res


def find_valid_timestamp_sequence(timestamps, idx, horizon, direction = 1, time_per_sample: float = 0.0):
    """
    Find valid timestamp sequence.
    """
    res = []
    last_ts = timestamps[idx]
    idx += direction
    while idx >= 0 and idx < len(timestamps) and horizon > 0:
        if (timestamps[idx] - last_ts) * direction >= time_per_sample * 1000:
            res.append(timestamps[idx])
            last_ts = timestamps[idx]
            horizon = horizon - 1
        idx += direction
    return res


def convert_tcp(tcp):
    """
    Convert tcp from xyz+quat to xyz+rpy.
    """
    return np.concatenate((tcp[:3], quat2euler(tcp[3:])))


def convert_gripper(width) -> float:
    """
    Convert gripper width into actual width.
    """
    return (255 - width) / 255.0 * 0.85 


def cls_gripper(width, last_width, threshold = 0.03) -> int:
    if np.abs(width - last_width) < threshold:
        return 0
    else:
        return np.sign(width - last_width)


class HorizonRecorder(object):
    def __init__(self, func):
        super(HorizonRecorder, self).__init__()
        self.func = func
        self.rec = []
    
    def clear(self):
        self.rec = []
    
    def add(self, x, *args):
        self.rec.append(self.func(x, *args))
    
    def pad(self, padding_mode = "same"):
        if padding_mode == "same":
            self.rec.append(self.rec[-1])
        elif padding_mode == "zero":
            self.rec.append(torch.zeros_like(self.rec[-1]))
        else:
            raise AttributeError('Invalid padding mode.')
    
    def __getitem__(self, idx):
        return self.rec[idx]

    def __len__(self):
        return len(self.rec)

    def to_tensor(self):
        if len(self.rec) == 0:
            return None
        else:
            return torch.stack(self.rec)


class WholeArmDataset(Dataset):
    def __init__(
        self, 
        path,
        task_name,
        split = 'train',
        freq = 20,
        preload = False,
        history_horizon = 0,
        action_horizon = 1,
        obs_visual_rep = False,
        obs_image_size = (224, 224),
        obs_with_depth = False,
        action_robot = "joint",
        action_delta = False,
        action_gripper_cls = True,
        action_gripper_cls_threshold = 0.03,
        norm_stats = {},
        scene_filter = (lambda sid: True),
        train_val_filter = (lambda sid: sid % 10 != 0),
        **kwargs
    ):
        """
        Args:
          - path: str, the path to the whole arm dataset;
          - task_name: str, the task name;
          - split: (optionoal) str, default: 'train', the dataset split;
          - freq: (optional) int [positive], default: 20, the frequency of data (if frequency is too high, then change to default frequency);
          - preload (optional) bool, default: False, whether to preload all data in the memory;
          - history_horizon: (optional) int, default: 0, the history horizon of the policy (current state excluded);
          - action_horizon: (optional) int, default: 1, the action horizon of the policy (current action included);
          - obs_visual_rep: (optional) bool, default: False, whether to use the pre-trained visual representations for image observations; enabling this option requires the dataset to be in visual representation version, please see function preprocess_visual_representations(...) first.
          - obs_image_size: (optional) tuple, default: (224, 224), the observation image size;
          - obs_with_depth: (optional) bool, default: False, whether to include the depth image in observations;
          - action_robot: (optional) str, default: "joint", the type of robot action ("tcp" means end-effector action, and "joint" means joint action);
          - action_delta: (optional) bool, default: False, whether to use delta value to represent action;
          - action_gripper_cls: (optional) bool, default: True, whether to use gripper value classes (0: stay; 1: open; -1: close);
          - action_gripper_cls_threshold (optional) float, default: 0.03 (m), the threshold value of binary gripper value;
          - norm_stats: (optional) dict, default: {}, the normalization statistics;
          - scene_filter: (optional) lambda expression Int -> Bool, default: (lambda sid: True), whether to select the scene in the dataset;
          - train_val_filter: (optional) lambda expression Int -> Bool, default: (lambda sid: sid % 10 != 0), the filter for train dataset and validation dataset, True for train and False for validation.
        """
        super(WholeArmDataset, self).__init__()
        if not os.path.exists(path):
            raise AttributeError("Dataset not found.")
        if split not in ['train', 'val']:
            raise ArithmeticError("split should be in ['train', 'val'].")
        if freq <= 0:
            raise AttributeError("Frequency should be a positive integer.")
        if action_robot not in ["tcp", "joint"]:
            raise AttributeError("action_robot should be in ['tcp', 'joint'].")
        freq = np.clip(freq, 0, 20)
        self.cfgs = edict({
            "path": path,
            "task_name": task_name,
            "split": split,
            "freq": freq,
            "tps": 1.0 / freq,
            "preload": preload,
            "history_horizon": history_horizon,
            "action_horizon": action_horizon,
            "obs_visual_rep": obs_visual_rep,
            "obs_image_size": obs_image_size,
            "obs_with_depth": obs_with_depth,
            "action_robot": action_robot,
            "action_delta": action_delta,
            "action_gripper_cls": action_gripper_cls,
            "action_gripper_cls_threshold": action_gripper_cls_threshold,
            "norm_stats": norm_stats,
            **kwargs
        })
        self.scene_filter = scene_filter
        self.train_val_filter = train_val_filter
        self.img_process = T.Compose([
            T.ToTensor(),
            T.Resize(obs_image_size),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.depth_process = T.Compose([
            T.ToTensor(),
            T.Resize(obs_image_size, interpolation = T.InterpolationMode.NEAREST),
        ])
        self.__load_data(self.cfgs)
        if preload:
            print('Loading all the data into the memory ...')
            self.memory = []
            for i in tqdm(range(len(self.data))):
                self.memory.append(self.__load_item(i))
    
    def __load_data(self, cfgs):
        """
        Load the data from cfgs.
        """
        self.data = []
        self.timestamps = {}
        is_train = (self.cfgs.split == 'train')
        for scene_folder in sorted(os.listdir(self.cfgs.path)):
            if scene_folder[:5] != "scene":
                continue
            scene_id = int(scene_folder[5:])
            if not self.scene_filter(scene_id):
                continue
            if self.train_val_filter(scene_id) != is_train:
                continue
            scene_path = os.path.join(self.cfgs.path, scene_folder)
            with open(os.path.join(scene_path, "meta.json"), "r") as f:
                meta = json.load(f)
            timestamps = sample_timestamps(meta["timestamps"], meta["start_time"], meta["stop_time"])
            self.timestamps[scene_path] = timestamps
            final_timestamp = timestamps[-1]
            for i in range(len(timestamps)):
                if final_timestamp - timestamps[i] < cfgs.tps * 1000:
                    break
                self.data.append({
                    "path": scene_path,
                    "timestamp": timestamps[i],
                    "history_timestamps": find_valid_timestamp_sequence(timestamps, i, cfgs.history_horizon, -1, cfgs.tps),
                    "action_timestamps": find_valid_timestamp_sequence(timestamps, i, cfgs.action_horizon, 1, cfgs.tps)
                })
    
    def __len__(self):
        return len(self.data)

    def __load_item(self, idx):
        """
        Load the idx-th item into the memory and return it.
        """
        if idx < 0 or idx >= len(self.data):
            raise AttributeError("Index out of bound.")
        sample = self.data[idx]
        res = {}
        cur = np.load(os.path.join(sample["path"], "{}.npy".format(sample["timestamp"])), allow_pickle = True).item()
        with_gripper_l = ("gripper_left" in cur.keys())
        with_gripper_r = ("gripper_right" in cur.keys())
        # 1. "obs/?"  including history horizon
        # 1.1  initialize horizon recorder for each field
        if self.cfgs.obs_visual_rep:
            obs_rep = HorizonRecorder(lambda x: torch.from_numpy(x["image_rep"]).float())
        else:
            if self.cfgs.task_name == "grasp_from_the_curtained_shelf":
                obs_img = HorizonRecorder(lambda x: torch.stack((self.img_process(x["image"]), self.img_process(x["image_up"]))))
            else:
                obs_img = HorizonRecorder(lambda x: self.img_process(x["image"]))
        if self.cfgs.obs_with_depth:
            if self.cfgs_task_name == "grasp_from_the_curtained_shelf":
                obs_depth = HorizonRecorder(lambda x: torch.stack((self.depth_process(x["depth"]), self.depth_process(x["depth_up"]))))
            else:
                obs_depth = HorizonRecorder(lambda x: self.depth_process(x["depth"]))
        obs_joint_l = HorizonRecorder(lambda x: torch.from_numpy(x["robot_left"][0:7]).float())
        obs_jointvel_l = HorizonRecorder(lambda x: torch.from_numpy(x["robot_left"][7:14]).float())
        obs_tcp_l = HorizonRecorder(lambda x: torch.from_numpy(convert_tcp(x["robot_left"][14:21])).float())
        obs_tcpvel_l = HorizonRecorder(lambda x: torch.from_numpy(x["robot_left"][21:27]).float())
        obs_fttcp_l = HorizonRecorder(lambda x: torch.from_numpy(x["robot_left"][27:33]).float())
        obs_ftbase_l = HorizonRecorder(lambda x: torch.from_numpy(x["robot_left"][33:39]).float())
        obs_joint_r = HorizonRecorder(lambda x: torch.from_numpy(x["robot_right"][0:7]).float())
        obs_jointvel_r = HorizonRecorder(lambda x: torch.from_numpy(x["robot_right"][7:14]).float())
        obs_tcp_r = HorizonRecorder(lambda x: torch.from_numpy(convert_tcp(x["robot_right"][14:21])).float())
        obs_tcpvel_r = HorizonRecorder(lambda x: torch.from_numpy(x["robot_right"][21:27]).float())
        obs_fttcp_r = HorizonRecorder(lambda x: torch.from_numpy(x["robot_right"][27:33]).float())
        obs_ftbase_r = HorizonRecorder(lambda x: torch.from_numpy(x["robot_right"][33:39]).float())
        if with_gripper_l:
            obs_gripper_l = HorizonRecorder(lambda x: torch.FloatTensor([convert_gripper(x["gripper_left"][0])]).float())
        if with_gripper_r:
            obs_gripper_r = HorizonRecorder(lambda x: torch.FloatTensor([convert_gripper(x["gripper_right"][0])]).float())
        # 1.2  add current value into horizon recorder
        if self.cfgs.obs_visual_rep:
            obs_rep.add(cur)
        else:
            obs_img.add(cur)
        if self.cfgs.obs_with_depth:
            obs_depth.add(cur)
        obs_joint_l.add(cur)
        obs_jointvel_l.add(cur)
        obs_tcp_l.add(cur)
        obs_tcpvel_l.add(cur)
        obs_fttcp_l.add(cur)
        obs_ftbase_l.add(cur)
        obs_joint_r.add(cur)
        obs_jointvel_r.add(cur)
        obs_tcp_r.add(cur)
        obs_tcpvel_r.add(cur)
        obs_fttcp_r.add(cur)
        obs_ftbase_r.add(cur)
        if with_gripper_l:
            obs_gripper_l.add(cur)
        if with_gripper_r:
            obs_gripper_r.add(cur)
        # 1.3  add history values into horizon recorder
        for ts in sample["history_timestamps"]:
            his = np.load(os.path.join(sample["path"], "{}.npy".format(ts)), allow_pickle = True).item()
            if self.cfgs.obs_visual_rep:
                obs_rep.add(his)
            else:
                obs_img.add(his)
            if self.cfgs.obs_with_depth:
                obs_depth.add(his)
            obs_joint_l.add(his)
            obs_jointvel_l.add(his)
            obs_tcp_l.add(his)
            obs_tcpvel_l.add(his)
            obs_fttcp_l.add(his)
            obs_ftbase_l.add(his)
            obs_joint_r.add(his)
            obs_jointvel_r.add(his)
            obs_tcp_r.add(his)
            obs_tcpvel_r.add(his)
            obs_fttcp_r.add(his)
            obs_ftbase_r.add(his)
            if with_gripper_l:
                obs_gripper_l.add(his)
            if with_gripper_r:
                obs_gripper_r.add(his)
        # 1.4  padding into the same horizon length
        for _ in range(self.cfgs.history_horizon - len(sample["history_timestamps"])):
            if self.cfgs.obs_visual_rep:
                obs_rep.pad()
            else:
                obs_img.pad()
            if self.cfgs.obs_with_depth:
                obs_depth.pad()
            obs_joint_l.pad()
            obs_jointvel_l.pad()
            obs_tcp_l.pad()
            obs_tcpvel_l.pad()
            obs_fttcp_l.pad()
            obs_ftbase_l.pad()
            obs_joint_r.pad()
            obs_jointvel_r.pad()
            obs_tcp_r.pad()
            obs_tcpvel_r.pad()
            obs_fttcp_r.pad()
            obs_ftbase_r.pad()
            if with_gripper_l:
                obs_gripper_l.pad()
            if with_gripper_r:
                obs_gripper_r.pad()
        # 1.5  get the final result
        res["obs/is_pad"] = torch.zeros((self.cfgs.history_horizon + 1), dtype = torch.bool)
        res["obs/is_pad"][len(sample["history_timestamps"]) + 1:] = 1
        if self.cfgs.obs_visual_rep:
            res["obs/image_rep"] = obs_rep.to_tensor()
        else:
            res["obs/image"] = obs_img.to_tensor()
        if self.cfgs.obs_with_depth:
            res["obs/depth"] = obs_depth.to_tensor()
        res["obs/left_joint"] = obs_joint_l.to_tensor()
        res["obs/left_joint_vel"] = obs_jointvel_l.to_tensor()
        res["obs/left_tcp"] = obs_tcp_l.to_tensor()
        res["obs/left_tcpvel"] = obs_tcpvel_l.to_tensor()
        res["obs/left_fttcp"] = obs_fttcp_l.to_tensor()
        res["obs/left_ftbase"] = obs_ftbase_l.to_tensor()
        res["obs/right_joint"] = obs_joint_r.to_tensor()
        res["obs/right_joint_vel"] = obs_jointvel_r.to_tensor()
        res["obs/right_tcp"] = obs_tcp_r.to_tensor()
        res["obs/right_tcpvel"] = obs_tcpvel_r.to_tensor()
        res["obs/right_fttcp"] = obs_fttcp_r.to_tensor()
        res["obs/right_ftbase"] = obs_ftbase_r.to_tensor()
        if with_gripper_l:
            res["obs/left_gripper"] = obs_gripper_l.to_tensor()
        if with_gripper_r:
            res["obs/right_gripper"] = obs_gripper_r.to_tensor()
        # 2. "action/?"  including action horizon
        # 2.1  initialize horizon recorder for each field
        if self.cfgs.action_robot == "joint":
            if self.cfgs.action_delta:
                action_robot_l = HorizonRecorder(lambda x, y: torch.from_numpy(x["robot_left"][0:7] - y["robot_left"][0:7]).float())
                action_robot_r = HorizonRecorder(lambda x, y: torch.from_numpy(x["robot_right"][0:7] - y["robot_right"][0:7]).float())
            else:
                action_robot_l = HorizonRecorder(lambda x: torch.from_numpy(x["robot_left"][0:7]).float())
                action_robot_r = HorizonRecorder(lambda x: torch.from_numpy(x["robot_right"][0:7]).float())
        else:
            if self.cfgs.action_delta:
                action_robot_l = HorizonRecorder(lambda x, y: torch.from_numpy(convert_tcp(x["robot_left"][14:21]) - convert_tcp(y["robot_left"][14:21])).float())
                action_robot_r = HorizonRecorder(lambda x, y: torch.from_numpy(convert_tcp(x["robot_right"][14:21]) - convert_tcp(y["robot_right"][14:21])).float())
            else:
                action_robot_l = HorizonRecorder(lambda x: torch.from_numpy(convert_tcp(x["robot_left"][14:21])).float())
                action_robot_r = HorizonRecorder(lambda x: torch.from_numpy(convert_tcp(x["robot_right"][14:21])).float())
        if with_gripper_l:
            if self.cfgs.action_gripper_cls:
                action_gripper_l = HorizonRecorder(
                    lambda x, y: torch.LongTensor([cls_gripper(convert_gripper(x["gripper_left"][0]), convert_gripper(y["gripper_left"][0]), self.cfgs.action_gripper_cls_threshold)])
                )
            else:
                action_gripper_l = HorizonRecorder(lambda x: torch.FloatTensor([convert_gripper(x["gripper_left"][0])]))
        if with_gripper_r:
            if self.cfgs.action_gripper_cls:
                action_gripper_r = HorizonRecorder(
                    lambda x, y: torch.LongTensor([cls_gripper(convert_gripper(x["gripper_right"][0]), convert_gripper(y["gripper_right"][0]), self.cfgs.action_gripper_cls_threshold)])
                )
            else:
                action_gripper_r = HorizonRecorder(lambda x: torch.FloatTensor([convert_gripper(x["gripper_right"][0])]))
        action_terminate = HorizonRecorder(lambda x: torch.LongTensor([x]))
        # 2.2  add future action into horizon recorder
        last = cur
        for ts in sample["action_timestamps"]:
            future = np.load(os.path.join(sample["path"], "{}.npy".format(ts)), allow_pickle = True).item()
            if self.cfgs.action_delta:
                action_robot_l.add(future, last)
                action_robot_r.add(future, last)
            else:
                action_robot_l.add(future)
                action_robot_r.add(future)
            if with_gripper_l:
                if self.cfgs.action_gripper_cls:
                    action_gripper_l.add(future, last)
                else:
                    action_gripper_l.add(future)
            if with_gripper_r:
                if self.cfgs.action_gripper_cls:
                    action_gripper_r.add(future, last)
                else:
                    action_gripper_r.add(future)
            action_terminate.add(ts == self.timestamps[sample["path"]][-1])
            last = future
        # 3.3  padding into the same horizon length
        for _ in range(self.cfgs.action_horizon - len(sample["action_timestamps"])):
            action_robot_l.pad(padding_mode = ("zero" if self.cfgs.action_delta else "same"))
            action_robot_r.pad(padding_mode = ("zero" if self.cfgs.action_delta else "same"))
            if with_gripper_l:
                action_gripper_l.pad(padding_mode = ("zero" if self.cfgs.action_gripper_cls else "same"))
            if with_gripper_r:
                action_gripper_r.pad(padding_mode = ("zero" if self.cfgs.action_gripper_cls else "same"))
            action_terminate.pad()
        # 3.4  get the final result
        res["action/is_pad"] = torch.zeros((self.cfgs.action_horizon), dtype = torch.bool)
        res["action/is_pad"][len(sample["action_timestamps"]):] = 1
        res["action/left_robot"] = action_robot_l.to_tensor()
        res["action/right_robot"] = action_robot_r.to_tensor()
        if with_gripper_l:
            res["action/left_gripper"] = action_gripper_l.to_tensor()
        if with_gripper_r:
            res["action/right_gripper"] = action_gripper_r.to_tensor()
        res["action/is_terminate"] = action_terminate.to_tensor()
        # 4. obtain the state and action according to task name
        if self.cfgs.task_name == "gather_balls":
            res["obs/robot_state"] = torch.cat([res["obs/left_joint"], res["obs/right_joint"]], dim = -1).float()
            res["obs/robot_state_reduced"] = torch.cat([res["obs/left_joint"][:, :4], res["obs/right_joint"][:, :4]], dim = -1).float()
            res["action/robot"] = torch.cat([res["action/left_robot"], res["action/right_robot"]], dim = -1).float()
            res["action/robot_reduced"] = torch.cat([res["action/left_robot"][:, :4], res["action/right_robot"][:, :4]], dim = -1).float()
        elif self.cfgs.task_name == "grasp_from_the_curtained_shelf":
            res["obs/robot_state"] = torch.cat([res["obs/left_joint"], res["obs/left_gripper"], res["obs/right_joint"]], dim = -1).float()
            res["obs/robot_state_reduced"] = torch.cat([res["obs/left_joint"][:, :], res["obs/left_gripper"][:, :], res["obs/right_joint"][:, :4]], dim = -1).float()
            res["action/robot"] = torch.cat([res["action/left_robot"], res["action/left_gripper"], res["action/right_robot"]], dim = -1).float()
            res["action/robot_reduced"] = torch.cat([res["action/left_robot"][:, :], res["action/left_gripper"][:, :], res["action/right_robot"][:, :4]], dim = -1).float()
        for key in self.cfgs.norm_stats.keys():
            if key in res.keys():
                res[key] = (res[key] - self.cfgs.norm_stats[key]["mean"]) / self.cfgs.norm_stats[key]["std"]
                res[key] = res[key].float()
        return res
    
    def __getitem__(self, idx):
        """
        Get the idx-th item.
        """
        if idx < 0 or idx >= len(self.data):
            raise AttributeError("Index out of bound.")
        if self.cfgs.preload:
            return self.memory[idx]
        else:
            return self.__load_item(idx)

    def fetch_database(self, fields = []):
        """
        Fetch the database according to certain key fields. The process could be very slow if preload is False.
        """
        res = []
        if self.cfgs.preload:
            for field in fields:
                res.append(torch.stack([memory_record.get(field) for memory_record in self.memory]))
        else:
            temp = [list(map(self.__load_item(idx).get, fields)) for idx in range(len(self.data))]
            for i, field in enumerate(fields):
                res.append(torch.stack([temp_record[i] for temp_record in temp]))
            del temp
        return res


# In-the-wild version
class WholeArmITWDataset(Dataset):
    def __init__(
        self, 
        path,
        task_name,
        split = 'train',
        freq = 20,
        preload = False,
        history_horizon = 0,
        action_horizon = 1,
        obs_visual_rep = False,
        obs_image_size = (224, 224),
        obs_with_depth = False,
        action_robot = "joint",
        action_delta = False,
        action_gripper_cls = True,
        action_gripper_cls_threshold = 0.03,
        norm_stats = {},
        scene_filter = (lambda sid: True),
        train_val_filter = (lambda sid: sid % 10 != 0),
        **kwargs
    ):
        """
        Args:
          - path: str, the path to the whole arm dataset;
          - task_name: str, the task name;
          - split: (optionoal) str, default: 'train', the dataset split;
          - freq: (optional) int [positive], default: 20, the frequency of data (if frequency is too high, then change to default frequency);
          - preload (optional) bool, default: False, whether to preload all data in the memory;
          - history_horizon: (optional) int, default: 0, the history horizon of the policy (current state excluded);
          - action_horizon: (optional) int, default: 1, the action horizon of the policy (current action included);
          - obs_visual_rep: (optional) bool, default: False, whether to use the pre-trained visual representations for image observations; enabling this option requires the dataset to be in visual representation version, please see function preprocess_visual_representations(...) first.
          - obs_image_size: (optional) tuple, default: (224, 224), the observation image size;
          - obs_with_depth: (optional) bool, default: False, whether to include the depth image in observations;
          - action_robot: (optional) str, default: "joint", the type of robot action, this can only be joint for in-the-wild dataset;
          - action_delta: (optional) bool, default: False, whether to use delta value to represent action;
          - action_gripper_cls: (optional) bool, default: True, whether to use gripper value classes (0: stay; 1: open; -1: close);
          - action_gripper_cls_threshold (optional) float, default: 0.03 (m), the threshold value of binary gripper value;
          - norm_stats: (optional) dict, default: {}, the normalization statistics;
          - scene_filter: (optional) lambda expression Int -> Bool, default: (lambda sid: True), whether to select the scene in the dataset;
          - train_val_filter: (optional) lambda expression Int -> Bool, default: (lambda sid: sid % 10 != 0), the filter for train dataset and validation dataset, True for train and False for validation.
        """
        super(WholeArmITWDataset, self).__init__()
        if not os.path.exists(path):
            raise AttributeError("Dataset not found.")
        if split not in ['train', 'val']:
            raise ArithmeticError("split should be in ['train', 'val'].")
        if freq <= 0:
            raise AttributeError("Frequency should be a positive integer.")
        if action_robot != "joint":
            raise AttributeError("action_robot can only be joint for in-the-wild dataset.")
        freq = np.clip(freq, 0, 20)
        self.cfgs = edict({
            "path": path,
            "task_name": task_name,
            "split": split,
            "freq": freq,
            "tps": 1.0 / freq,
            "preload": preload,
            "history_horizon": history_horizon,
            "action_horizon": action_horizon,
            "obs_visual_rep": obs_visual_rep,
            "obs_image_size": obs_image_size,
            "obs_with_depth": obs_with_depth,
            "action_robot": action_robot,
            "action_delta": action_delta,
            "action_gripper_cls": action_gripper_cls,
            "action_gripper_cls_threshold": action_gripper_cls_threshold,
            "norm_stats": norm_stats,
            **kwargs
        })
        self.scene_filter = scene_filter
        self.train_val_filter = train_val_filter
        self.img_process = T.Compose([
            T.ToTensor(),
            T.Resize(obs_image_size),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.depth_process = T.Compose([
            T.ToTensor(),
            T.Resize(obs_image_size, interpolation = T.InterpolationMode.NEAREST),
        ])
        self.vflip = T.RandomVerticalFlip(p = 1.0)
        self.hflip = T.RandomHorizontalFlip(p = 1.0)
        self.__load_data(self.cfgs)
        if preload:
            print('Loading all the data into the memory ...')
            self.memory = []
            for i in tqdm(range(len(self.data))):
                self.memory.append(self.__load_item(i))
    
    def __load_data(self, cfgs):
        """
        Load the data from cfgs.
        """
        self.data = []
        self.timestamps = {}
        is_train = (self.cfgs.split == 'train')
        for scene_folder in sorted(os.listdir(self.cfgs.path)):
            if scene_folder[:5] != "scene":
                continue
            scene_id = int(scene_folder[5:])
            if not self.scene_filter(scene_id):
                continue
            if self.train_val_filter(scene_id) != is_train:
                continue
            scene_path = os.path.join(self.cfgs.path, scene_folder)
            with open(os.path.join(scene_path, "meta.json"), "r") as f:
                meta = json.load(f)
            timestamps = sample_timestamps(meta["timestamps"], meta["start_time"], meta["stop_time"])
            self.timestamps[scene_path] = timestamps
            final_timestamp = timestamps[-1]
            for i in range(len(timestamps)):
                if final_timestamp - timestamps[i] < cfgs.tps * 1000:
                    break
                self.data.append({
                    "path": scene_path,
                    "timestamp": timestamps[i],
                    "history_timestamps": find_valid_timestamp_sequence(timestamps, i, cfgs.history_horizon, -1, cfgs.tps),
                    "action_timestamps": find_valid_timestamp_sequence(timestamps, i, cfgs.action_horizon, 1, cfgs.tps)
                })
    
    def __len__(self):
        return len(self.data)

    def __load_item(self, idx):
        """
        Load the idx-th item into the memory and return it.
        """
        if idx < 0 or idx >= len(self.data):
            raise AttributeError("Index out of bound.")
        sample = self.data[idx]
        res = {}
        cur = np.load(os.path.join(sample["path"], "{}.npy".format(sample["timestamp"])), allow_pickle = True).item()
        # 1. "obs/?"  including history horizon
        # 1.1  initialize horizon recorder for each field
        if self.cfgs.obs_visual_rep:
            obs_rep = HorizonRecorder(lambda x: torch.from_numpy(x["image_rep"]).float())
        else:
            if self.cfgs.task_name == "gather_balls":
                obs_img = HorizonRecorder(lambda x: self.hflip(self.vflip(self.img_process(x["image"]))))
            elif self.cfgs.task_name == "grasp_from_the_curtained_shelf":
                obs_img = HorizonRecorder(lambda x: torch.stack((self.img_process(x["image"]), self.img_process(x["image_up"]))))
            else:
                obs_img = HorizonRecorder(lambda x: self.img_process(x["image"]))
        if self.cfgs.obs_with_depth:
            if self.cfgs.task_name == "gather_balls":
                obs_depth = HorizonRecorder(lambda x: self.hflip(self.vflip(self.depth_process(x["depth"]))))
            elif self.cfgs_task_name == "grasp_from_the_curtained_shelf":
                obs_depth = HorizonRecorder(lambda x: torch.stack((self.depth_process(x["depth"]), self.depth_process(x["depth_up"]))))
            else:
                obs_depth = HorizonRecorder(lambda x: self.depth_process(x["depth"]))
        obs_joint_l = HorizonRecorder(lambda x: torch.from_numpy(x["robot_left"][0:7]).float())
        obs_joint_r = HorizonRecorder(lambda x: torch.from_numpy(x["robot_right"][0:7]).float())
        obs_gripper_l = HorizonRecorder(lambda x: torch.FloatTensor([convert_gripper(x["robot_left"][7])]).float())
        obs_gripper_r = HorizonRecorder(lambda x: torch.FloatTensor([convert_gripper(x["robot_right"][7])]).float())
        # 1.2  add current value into horizon recorder
        if self.cfgs.obs_visual_rep:
            obs_rep.add(cur)
        else:
            obs_img.add(cur)
        if self.cfgs.obs_with_depth:
            obs_depth.add(cur)
        obs_joint_l.add(cur)
        obs_joint_r.add(cur)
        obs_gripper_l.add(cur)
        obs_gripper_r.add(cur)
        # 1.3  add history values into horizon recorder
        for ts in sample["history_timestamps"]:
            his = np.load(os.path.join(sample["path"], "{}.npy".format(ts)), allow_pickle = True).item()
            if self.cfgs.obs_visual_rep:
                obs_rep.add(his)
            else:
                obs_img.add(his)
            if self.cfgs.obs_with_depth:
                obs_depth.add(his)
            obs_joint_l.add(his)
            obs_joint_r.add(his)
            obs_gripper_l.add(his)
            obs_gripper_r.add(his)
        # 1.4  padding into the same horizon length
        for _ in range(self.cfgs.history_horizon - len(sample["history_timestamps"])):
            if self.cfgs.obs_visual_rep:
                obs_rep.pad()
            else:
                obs_img.pad()
            if self.cfgs.obs_with_depth:
                obs_depth.pad()
            obs_joint_l.pad()
            obs_joint_r.pad()
            obs_gripper_l.pad()
            obs_gripper_r.pad()
        # 1.5  get the final result
        res["obs/is_pad"] = torch.zeros((self.cfgs.history_horizon + 1), dtype = torch.bool)
        res["obs/is_pad"][len(sample["history_timestamps"]) + 1:] = 1
        if self.cfgs.obs_visual_rep:
            res["obs/image_rep"] = obs_rep.to_tensor()
        else:
            res["obs/image"] = obs_img.to_tensor()
        if self.cfgs.obs_with_depth:
            res["obs/depth"] = obs_depth.to_tensor()
        res["obs/left_joint"] = obs_joint_l.to_tensor()
        res["obs/right_joint"] = obs_joint_r.to_tensor() 
        res["obs/left_gripper"] = obs_gripper_l.to_tensor()
        res["obs/right_gripper"] = obs_gripper_r.to_tensor()
        # 2. "action/?"  including action horizon
        # 2.1  initialize horizon recorder for each field
        if self.cfgs.action_delta:
            action_robot_l = HorizonRecorder(lambda x, y: torch.from_numpy(x["robot_left"][0:7] - y["robot_left"][0:7]).float())
            action_robot_r = HorizonRecorder(lambda x, y: torch.from_numpy(x["robot_right"][0:7] - y["robot_right"][0:7]).float())
        else:
            action_robot_l = HorizonRecorder(lambda x: torch.from_numpy(x["robot_left"][0:7]).float())
            action_robot_r = HorizonRecorder(lambda x: torch.from_numpy(x["robot_right"][0:7]).float())
        if self.cfgs.action_gripper_cls:
            action_gripper_l = HorizonRecorder(
                lambda x, y: torch.LongTensor([cls_gripper(convert_gripper(x["robot_left"][7]), convert_gripper(y["robot_left"][7]), self.cfgs.action_gripper_cls_threshold)])
            )
        else:
            action_gripper_l = HorizonRecorder(lambda x: torch.FloatTensor([convert_gripper(x["robot_left"][7])]))
        if self.cfgs.action_gripper_cls:
            action_gripper_r = HorizonRecorder(
                lambda x, y: torch.LongTensor([cls_gripper(convert_gripper(x["robot_right"][7]), convert_gripper(y["robot_right"][7]), self.cfgs.action_gripper_cls_threshold)])
            )
        else:
            action_gripper_r = HorizonRecorder(lambda x: torch.FloatTensor([convert_gripper(x["robot_right"][7])]))
        action_terminate = HorizonRecorder(lambda x: torch.LongTensor([x]))
        # 2.2  add future action into horizon recorder
        last = cur
        for ts in sample["action_timestamps"]:
            future = np.load(os.path.join(sample["path"], "{}.npy".format(ts)), allow_pickle = True).item()
            if self.cfgs.action_delta:
                action_robot_l.add(future, last)
                action_robot_r.add(future, last)
            else:
                action_robot_l.add(future)
                action_robot_r.add(future)
            if self.cfgs.action_gripper_cls:
                action_gripper_l.add(future, last)
            else:
                action_gripper_l.add(future)
            if self.cfgs.action_gripper_cls:
                action_gripper_r.add(future, last)
            else:
                action_gripper_r.add(future)
            action_terminate.add(ts == self.timestamps[sample["path"]][-1])
            last = future
        # 3.3  padding into the same horizon length
        for _ in range(self.cfgs.action_horizon - len(sample["action_timestamps"])):
            action_robot_l.pad(padding_mode = ("zero" if self.cfgs.action_delta else "same"))
            action_robot_r.pad(padding_mode = ("zero" if self.cfgs.action_delta else "same"))
            action_gripper_l.pad(padding_mode = ("zero" if self.cfgs.action_gripper_cls else "same"))
            action_gripper_r.pad(padding_mode = ("zero" if self.cfgs.action_gripper_cls else "same"))
            action_terminate.pad()
        # 3.4  get the final result
        res["action/is_pad"] = torch.zeros((self.cfgs.action_horizon), dtype = torch.bool)
        res["action/is_pad"][len(sample["action_timestamps"]):] = 1
        res["action/left_robot"] = action_robot_l.to_tensor()
        res["action/right_robot"] = action_robot_r.to_tensor()
        res["action/left_gripper"] = action_gripper_l.to_tensor()
        res["action/right_gripper"] = action_gripper_r.to_tensor()
        res["action/is_terminate"] = action_terminate.to_tensor()
        # 4. obtain the state and action according to task name
        if self.cfgs.task_name == "gather_balls":
            res["obs/robot_state"] = torch.cat([res["obs/left_joint"], res["obs/right_joint"]], dim = -1).float()
            res["obs/robot_state_reduced"] = torch.cat([res["obs/left_joint"][:, :4], res["obs/right_joint"][:, :4]], dim = -1).float()
            res["action/robot"] = torch.cat([res["action/left_robot"], res["action/right_robot"]], dim = -1).float()
            res["action/robot_reduced"] = torch.cat([res["action/left_robot"][:, :4], res["action/right_robot"][:, :4]], dim = -1).float()
        elif self.cfgs.task_name == "grasp_from_the_curtained_shelf":
            res["obs/robot_state"] = torch.cat([res["obs/left_joint"], res["obs/left_gripper"], res["obs/right_joint"]], dim = -1).float()
            res["obs/robot_state_reduced"] = torch.cat([res["obs/left_joint"][:, :], res["obs/left_gripper"][:, :], res["obs/right_joint"][:, :4]], dim = -1).float()
            res["action/robot"] = torch.cat([res["action/left_robot"], res["action/left_gripper"], res["action/right_robot"]], dim = -1).float()
            res["action/robot_reduced"] = torch.cat([res["action/left_robot"][:, :], res["action/left_gripper"][:, :], res["action/right_robot"][:, :4]], dim = -1).float()
        for key in self.cfgs.norm_stats.keys():
            if key in res.keys():
                res[key] = (res[key] - self.cfgs.norm_stats[key]["mean"]) / self.cfgs.norm_stats[key]["std"]
                res[key] = res[key].float()
        return res
    
    def __getitem__(self, idx):
        """
        Get the idx-th item.
        """
        if idx < 0 or idx >= len(self.data):
            raise AttributeError("Index out of bound.")
        if self.cfgs.preload:
            return self.memory[idx]
        else:
            return self.__load_item(idx)

    def fetch_database(self, fields = []):
        """
        Fetch the database according to certain key fields. The process could be very slow if preload is False.
        """
        res = []
        if self.cfgs.preload:
            for field in fields:
                res.append(torch.stack([memory_record.get(field) for memory_record in self.memory]))
        else:
            temp = [list(map(self.__load_item(idx).get, fields)) for idx in range(len(self.data))]
            for i, field in enumerate(fields):
                res.append(torch.stack([temp_record[i] for temp_record in temp]))
            del temp
        return res


if __name__ == '__main__': 
    dataset = WholeArmDataset('/path/to/data/task2/', task_name = 'grasp_from_the_curtained_shelf', split = 'train', freq = 1.0, preload = False, history_horizon = 0, action_horizon = 1, obs_visual_rep = False)
    print(len(dataset))
    print(dataset[0])
    """
    dataset = WholeArmDataset('/path/to/data/task1/', task_name = 'gather_balls', split = 'train', freq = 1.0, preload = False, history_horizon = 0, action_horizon = 1, obs_visual_rep = False)
    print(len(dataset))
    print(dataset[0])
    
    dataset = WholeArmITWDataset('/path/to/data/task1-in-the-wild/', task_name = 'gather_balls', split = 'train', freq = 1.0, preload = False, history_horizon = 0, action_horizon = 1, obs_visual_rep = False)
    print(len(dataset))
    print(dataset[0])
    """