#!/bin/bash
#SBATCH --job-name=unet
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --ntasks=8
#SBATCH --gres=gpu:4
#SBATCH --time=144:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --exclude=SPG-1-[1-4] 
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL

source ~/.bashrc
conda activate paddle_x3.9
module load cuda

export CUDA_VISIBLE_DEVICES=0,1,2,3

'''
# config the following environment variables to enable parallel graph mode 
# and nccl2 distributed training mode on demand

export FLAGS_sync_nccl_allreduce=1 # use nccl to do allreduce
export FLAGS_enable_parallel_graph=1 # enable parallel graph mode
'''

python -m paddle.distributed.launch --log_dir=output_unet unet.py
