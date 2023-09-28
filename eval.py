import os
import json
import time
import torch
import argparse
import numpy as np
import torchvision.transforms as T

from PIL import Image
from easydict import EasyDict as edict
from easyrobot.robot.api import get_robot
from easyrobot.camera.api import get_rgbd_camera

from utils import get_stats, set_seed
from policy import ACTPolicy, CNNMLPPolicy


def main(args):
    set_seed(1)

    # command line parameters
    ckpt = args['ckpt']
    robot_cfgs = args['robot_cfgs']
    policy_class = args['policy_class']
    task_name = args['task_name']
    control_freq = args['control_freq']

    # get task parameters
    from constants import TASK_CONFIGS
    task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']
    norm_stats = task_config['norm_stats']
    chunk_size = args['chunk_size']

    # fixed parameters
    state_dim = 12
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'num_queries': chunk_size,
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         'state_dim': state_dim
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names, 'state_dim': state_dim}
    else:
        raise NotImplementedError

    config = {
        'ckpt': ckpt,
        'robot_cfgs': robot_cfgs,
        'control_freq': control_freq,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'policy_class': policy_class,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'dataset_dir': dataset_dir,
        'norm_stats': norm_stats
    }

    eval_bc(config)
    

def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy

def convert_gripper(width) -> float:
    """
    Convert gripper width into actual width.
    """
    return (255 - width) / 255.0 * 0.85 



def eval_bc(config):
    set_seed(config["seed"])
    # load robot configurations
    if not os.path.exists(config["robot_cfgs"]):
        raise AttributeError('Please provide the configuration file {}.'.format(config["robot_cfgs"]))
    with open(config["robot_cfgs"], 'r') as f:
        cfgs = edict(json.load(f))
    
    task_name = config["task_name"]
    dataset_dir = config["dataset_dir"]
    norm_stats_file = config["norm_stats"]

    # load norm statistics file
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
        else:
            raise AttributeError('Invalid task.')
        print('Normalization statistics calculated and saved.')
    
    # if task_name == "gather_balls":
    norm_state = norm_stats['obs/robot_state_reduced']
    norm_action = norm_stats['action/robot_reduced']

    # initialize camera(s)
    cameras = []
    for cam in cfgs.cameras:
        cam = get_rgbd_camera(**cam)
        cameras.append(cam)
    
    # initialize robots and grippers
    robot_left = get_robot(**cfgs.robot_left)
    robot_right = get_robot(**cfgs.robot_right)
    robot_left.send_joint_pos(cfgs.initialization.robot_left, wait = True, **cfgs.initialization.params_left)
    robot_right.send_joint_pos(cfgs.initialization.robot_right, wait = True, **cfgs.initialization.params_right)
    
    if "gripper_left" in cfgs.initialization.keys():
        if cfgs.initialization.gripper_left == 1:
            robot_left.open_gripper()
        else:
            robot_left.close_gripper()
    if "gripper_right" in cfgs.initialization.keys():
        if cfgs.initialization.gripper_right == 1:
            robot_right.open_gripper()
        else:
            robot_right.close_gripper()       

    time.sleep(5)
    
    # preparation for load policy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = config['ckpt']
    state_dim = config['state_dim']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    max_timesteps = config['episode_len']
    temporal_agg = config['temporal_agg']
    norm_stats_file = config['norm_stats']

    # load policy
    policy = make_policy(policy_class, policy_config)
    policy.load_state_dict(torch.load(ckpt, map_location = device))
    policy.to(device)
    policy.eval()
    print(f'Policy loaded: {ckpt}')

    # load preprocessing and postprocessing functions
    pre_process = lambda s_qpos: (s_qpos - norm_state['mean']) / norm_state['std']
    post_process = lambda a: a * norm_action['std'] + norm_action['mean']

    # load max timesteps
    max_timesteps = int(max_timesteps * 20) # may increase for real-world tasks

    # temporal aggregation
    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']
        all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).to(device)
    
    # image transformation
    tf = T.Compose([
        T.ToTensor(),
        T.Resize((480, 640)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # step time
    step_time = 1.0 / config["control_freq"]

    with torch.inference_mode():
        for t in range(max_timesteps):
            start_time = time.time()
            # fetch images
            imgs = []
            for cam in cameras:
                imgs.append(tf(Image.fromarray(cam.get_rgb_image().astype(np.uint8))))
            img = torch.stack(imgs).float().unsqueeze(0)
            img = img.to(device)
            
            # fetch robot states
            if task_name == 'gather_balls':
                qpos = np.concatenate((robot_left.get_joint_pos()[:4],robot_right.get_joint_pos()[:4]))  
            elif task_name == 'grasp_from_the_curtained_shelf':    
                arr = []
                arr.append(convert_gripper(robot_left.get_gripper_info()[0]))
                qpos = np.concatenate((robot_left.get_joint_pos()[:7], arr, robot_right.get_joint_pos()[:4]))
            else:
                raise AttributeError('Invalid task.')
            qpos = pre_process(torch.from_numpy(qpos))
            qpos = qpos.float().unsqueeze(0)
            qpos = qpos.to(device)

            # query policy
            if config['policy_class'] == "ACT":
                if t % query_frequency == 0:
                    all_actions = policy(qpos, img)
                if temporal_agg:
                    all_time_actions[[t], t: t + num_queries] = all_actions
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).unsqueeze(dim=1).to(device)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    raw_action = all_actions[:, t % query_frequency]
            elif config['policy_class'] == "CNNMLP":
                raw_action = policy(qpos, img)
            else:
                raise NotImplementedError

            # post-process actions
            raw_action = raw_action.squeeze(0).cpu()
            action = post_process(raw_action).numpy()

            robot_action_left = np.zeros(cfgs.dof.robot_left)
            robot_action_right = np.zeros(cfgs.dof.robot_right)

            # calculate real actions for robot
            if task_name == 'gather_balls':
                robot_action_left[:4] = action[:4]
                robot_action_right[:4] = action[4:]
                for key, value in cfgs.action.fixed_left.items():
                    robot_action_left[int(key)] = value
                for key, value in cfgs.action.fixed_right.items():
                    robot_action_right[int(key)] = value
                gripper_action_left = None
                gripper_action_right = None
            elif task_name == 'grasp_from_the_curtained_shelf':
                robot_action_left[:7] = action[:7]
                robot_action_right[:4] = action[8:]
                for key, value in cfgs.action.fixed_left.items():
                    robot_action_left[int(key)] = value
                for key, value in cfgs.action.fixed_right.items():
                    robot_action_right[int(key)] = value
                gripper_action_left = action[7]
                gripper_action_right = None
            else:
                raise AttributeError('Invalid task.')
            
            # action
            robot_left.send_joint_pos(robot_action_left, wait = False, **cfgs.action.params_left)
            robot_right.send_joint_pos(robot_action_right, wait = False, **cfgs.action.params_right)
            if gripper_action_left is not None:
                if gripper_action_left < cfgs.action.gripper_close_threshold_left:
                    robot_left.close_gripper()
                if gripper_action_left > cfgs.action.gripper_open_threshold_left:
                    robot_left.open_gripper()

            if gripper_action_right is not None:
                if gripper_action_right < cfgs.action.gripper_close_threshold_right:
                    robot_right.close_gripper()
                if gripper_action_right > cfgs.action.gripper_open_threshold_right:
                    robot_right.open_gripper()

            duration = time.time() - start_time
            if duration < step_time:
                time.sleep(step_time - duration)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', action='store', type=str, help='checkpoint file', required=True)
    parser.add_argument('--robot_cfgs', action='store', type=str, help='real-robot evaluation config file', required=True)
    parser.add_argument('--control_freq', action='store', type=float, help='control frequency', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')

    main(vars(parser.parse_args()))
