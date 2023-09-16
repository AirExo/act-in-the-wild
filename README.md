#
## Example usages
To train ACT with muti-gpu:
```
python -m torch.distributed.launch --nproc_per_node=2 train.py \
--ckpt_dir [ckpt_dir] --task_name [category of the task] \
--policy_class ACT --batch_size 8 --num_epochs 1000 --chunk_size 20 --freq 5 --save_epoch 10 \
--lr 1e-5 --hidden_dim 1024 --dim_feedforward 6400 --kl_weight 10 \
--seed 0
```

To eval the policy, please run the following command and if you want to enable temporal ensembling, add `--temporal ensembling`.
```
python eval.py \
--task_name [category of the task] --ckpt [ckpt_dir] --robot_cfgs [configs] \
--control_freq 10 --policy_class ACT --seed 3407 --chunk_size 20 --hidden_dim 1024 --dim_feedforward 6400 \
--temporal_agg
```