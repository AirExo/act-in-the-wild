import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from tqdm import tqdm
from wholearm import WholeArmDataset, WholeArmITWDataset


class WholeArmDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, task_name, split, freq, camera_names, chunk_size, norm_stats):
        super(WholeArmDatasetWrapper).__init__()
        self.norm_stats = norm_stats
        self.split = split
        if chunk_size is None:
            action_horizon = 1
        else:
            action_horizon = chunk_size
        self.dataset = WholeArmDataset(
            dataset_dir, 
            task_name, 
            split = split, 
            freq = freq, 
            preload = False, 
            history_horizon = 0, 
            action_horizon = action_horizon, 
            obs_visual_rep = False, 
            obs_image_size = (480, 640), 
            norm_stats = norm_stats,
            scene_filter = (lambda sid: sid % 5 == 2 or sid % 10 == 0),
            train_val_filter = (lambda sid: sid % 10 != 0)
        )
        print('Dataset loaded, # {} sample: {}'.format(split, len(self.dataset)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        qpos_data = data['obs/robot_state_reduced'][0]
        image_data = data['obs/image']
        action_data = data['action/robot_reduced']
        is_pad = data['action/is_pad']
        return image_data, qpos_data, action_data, is_pad


class WholeArmITWDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, task_name, split, freq, camera_names, chunk_size, norm_stats):
        super(WholeArmITWDatasetWrapper).__init__()
        self.norm_stats = norm_stats
        self.split = split
        if chunk_size is None:
            action_horizon = 1
        else:
            action_horizon = chunk_size
        self.dataset = WholeArmITWDataset(
            dataset_dir, 
            task_name, 
            split = split, 
            freq = freq, 
            preload = False, 
            history_horizon = 0, 
            action_horizon = action_horizon, 
            obs_visual_rep = False, 
            obs_image_size = (480, 640), 
            norm_stats = norm_stats, 
            scene_filter = (lambda sid: sid <= 100),
            train_val_filter = (lambda sid: sid % 10 != 0),
            action_gripper_cls_threshold = 0.1
        )
        print('Dataset loaded, # sample: {}'.format(len(self.dataset)))
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        qpos_data = data['obs/robot_state_reduced'][0]
        image_data = data['obs/image']
        action_data = data['action/robot_reduced']
        is_pad = data['action/is_pad']
        return image_data, qpos_data, action_data, is_pad


def load_data(dataset_dir, task_name, camera_names, batch_size_train, batch_size_val, chunk_size, norm_stats_file, freq = 1.0, itw = False):
    print(f'\nData from: {dataset_dir} with frequency = {freq}\n')

    if os.path.exists(norm_stats_file):
        norm_stats = np.load(norm_stats_file, allow_pickle = True).item()
        print('Normalization statistics loaded.')
    else:
        print('No normaliation statistics found, calculating statistics ...')
        if task_name == "gather_balls":
            keys = ["obs/robot_state_reduced", "action/robot_reduced"]
            data = get_stats(dataset_dir, lambda x: np.concatenate((x['robot_left'][0:4], x['robot_right'][0:4])))
            res = {key: data for key in keys}
            norm_stats = res
            np.save(norm_stats_file, res)
        elif task_name == "grasp_from_the_curtained_shelf":
            data = get_stats(dataset_dir, lambda x: np.concatenate((x['robot_left'], x['robot_right'][0:4])))
            mean = data['mean']
            std = data['std']
            res = {
                "obs/robot_state_reduced": {
                    'mean': np.concatenate((mean[0:7], np.array([0.0]), mean[7:11])),
                    'std': np.concatenate((std[0:7], np.array([0.425]), std[7:11]))
                },
                "action/robot_reduced": {
                    'mean': np.concatenate((mean[0:7], np.array([0.0]), mean[7:11])),
                    'std': np.concatenate((std[0:7], np.array([1.0]), std[7:11]))
                }
            }
            norm_stats = res
            np.save(norm_stats_file, res)
        else:
            raise AttributeError('Invalid task.')
        print('Normalization statistics calculated and saved.')
    if itw: 
        train_dataset = WholeArmITWDatasetWrapper(dataset_dir, task_name, 'train', freq, camera_names, chunk_size, norm_stats)
        val_dataset = WholeArmITWDatasetWrapper(dataset_dir, task_name, 'val', freq, camera_names, chunk_size, norm_stats)
    else:
        train_dataset = WholeArmDatasetWrapper(dataset_dir, task_name, 'train', freq, camera_names, chunk_size, norm_stats)
        val_dataset = WholeArmDatasetWrapper(dataset_dir, task_name, 'val', freq, camera_names, chunk_size, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size_train, shuffle = True, pin_memory = True, num_workers = 100)
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size_val, shuffle = True, pin_memory = True, num_workers = 100)

    return train_dataloader, val_dataloader


def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def get_stats(
    path,
    func,
    **kwargs
):
    """
    Get the statistics of dataset.

    Args:
      - path: str, the path to the whole arm dataset;
      - func: lambda expression, the specific area of interest.
    """
    rec = []
    for scene_folder in tqdm(sorted(os.listdir(path))):
        if scene_folder[:5] != "scene":
            continue
        scene_path = os.path.join(path, scene_folder)
        for record in sorted(os.listdir(scene_path)):
            if os.path.splitext(record)[-1] != '.npy':
                continue
            t = np.load(os.path.join(scene_path, record), allow_pickle=True).item()
            rec.append(torch.from_numpy(func(t)))
    rec = torch.stack(rec)
    mean = rec.mean(dim = 0)
    std = rec.std(dim = 0)
    std = torch.clip(std, 1e-2, 10)
    return {
        'mean': mean,
        'std': std
    }