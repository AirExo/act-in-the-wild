import os

### Task parameters
DATA_DIR = '/path/to/data/'    
BASE_DIR = '.'
TASK_CONFIGS = {
    'gather_balls': {
        'dataset_dir': os.path.join(DATA_DIR, 'task1'),
        "dataset_dir_itw": os.path.join(DATA_DIR, 'task1-in-the-wild'),
        'episode_len': 400,
        'camera_names': ['top'],
        'state_dim': 8,
        'stats_dir': os.path.join(BASE_DIR, 'stats', 'gather_balls'),
        'norm_stats': os.path.join(BASE_DIR, 'stats', 'gather_balls', 'norm_stats.npy')
    },
    "grasp_from_the_curtained_shelf": {
        'dataset_dir': os.path.join(DATA_DIR, 'task2'),
        'dataset_dir_itw': os.path.join(DATA_DIR, 'task2-in-the-wild'),
        'episode_len': 600,
        'camera_names': ['base', 'up'],
        'state_dim': 12,
        'stats_dir': os.path.join(BASE_DIR, 'stats', 'grasp_from_the_curtained_shelf'),
        'norm_stats': os.path.join(BASE_DIR, 'stats', 'grasp_from_the_curtained_shelf', 'norm_stats.npy')
    }
}
