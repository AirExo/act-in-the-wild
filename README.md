# ACT with in-the-Wild Learning

[[Paper]](https://arxiv.org/pdf/2309.14975.pdf) [[Project Page]](https://airexo.github.io/) [[Data Collector]](https://github.com/AirExo/collector) [[Sample Demonstration Data]](https://drive.google.com/drive/folders/1f_bmrFPep90aUSBj28TdXRiNvHo7PpxR?usp=drive_link)

This repository contains code for training the policy with in-the-wild demonstrations and teleoperated demonstrations. These demonstrations are collected by ***AirExo***, see data collector for details. This repository is modified upon [the original ACT repository](https://github.com/tonyzhaozh/act) to adapt our demonstration data and increase validation sample density per epoch, as pointed out in [here](https://github.com/tonyzhaozh/act/issues/3).

## Requirements

Python Dependencies:

- numpy
- torch
- torchvision
- opencv-python
- pillow
- easydict
- argparse
- tqdm
- transforms3d
- ipython

For real-world evaluations, you will need the same hardware as described in data collector and [easyrobot](https://github.com/galaxies99/easyrobot) library.

## Run

### Dataset Configurations

Please configurate the dataset information in `constants.py` and the lambda expressions of `scene_filter` and `train_val_filter` in `utils.py` for data selection (*e.g.*, if you only want to use 10 samples for training, *etc.*).

### Training from Scratch

To train ACT from scratch:

```bash
python train.py \
        --ckpt_dir [ckpt_dir] \
        --policy_class ACT \
        --task_name [task name] \
        --batch_size [batch size] \
        --seed [seed] \
        --num_epochs [num epoch] \
        --save_epoch [save epoch] \
        --lr [learning rate] \
        --freq [data frequency] \
        --kl_weight [kl weight] \
        --chunk_size [chunk size] \
        --hidden_dim [hidden dim] \
        --dim_feedforward [dim feedforward]
```

### In-the-Wild Learning Framework

For in-the-wild learning stage 1: pre-training with in-the-wild demonstrations:

```bash
python train.py \
        --ckpt_dir [ckpt_dir] \
        --policy_class ACT \
        --task_name [task name] \
        --batch_size [batch size] \
        --seed [seed] \
        --num_epochs [num epoch] \
        --save_epoch [save epoch] \
        --lr [learning rate] \
        --freq [data frequency] \
        --in_the_wild
        --kl_weight [kl weight] \
        --chunk_size [chunk size] \
        --hidden_dim [hidden dim] \
        --dim_feedforward [dim feedforward]
```

For in-the-wild learning stage 2: fine-tuning with teleoperated demonstrations:

```bash
python train.py \
        --ckpt_dir [ckpt_dir] \
        --policy_class ACT \
        --task_name [task name] \
        --batch_size [batch size] \
        --seed [seed] \
        --num_epochs [num epoch] \
        --save_epoch [save epoch] \
        --lr [learning rate] \
        --freq [data frequency] \
        --resume_ckpt [pre-trained checkpoint] \
        --kl_weight [kl weight] \
        --chunk_size [chunk size] \
        --hidden_dim [hidden dim] \
        --dim_feedforward [dim feedforward]
```

### Real-World Evaluation

To evaluate the policy in real world:

```bash
python eval.py \
        --ckpt [checkpoint] \
        --robot_cfgs [evaluation configurations] \
        --control_freq [control frequency] \
        --policy_class ACT \
        --task_name [task name] \
        --seed [seed] \
        --kl_weight [kl weight] \
        --chunk_size [chunk size] \
        --hidden_dim [hidden dim] \
        --dim_feedforward [dim feedforward] \
        --temporal_agg
```

For evaluation configurations, see `configs` folder for details.

## Reference

Original ACT repository: [https://github.com/tonyzhaozh/act](https://github.com/tonyzhaozh/act).

## Citation

If you find ***AirExo*** useful in your research, please consider citing the following paper:

```bibtex
@article{
    fang2023low,
    title = {Low-Cost Exoskeletons for Learning Whole-Arm Manipulation in the Wild},
    author = {Fang, Hongjie and Fang, Hao-Shu and Wang, Yiming and Ren, Jieji and Chen, Jingjing and Zhang, Ruo and Wang, Weiming and Lu, Cewu},
    journal = {arXiv preprint arXiv:2309.14975},
    year = {2023}
}
```
