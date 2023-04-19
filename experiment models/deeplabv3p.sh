'''
Operating System: CentOS 8 (x86-64)
HPC software stack: OpenHPC
Job Scheduler: SLURM
Parallel computing framework: PaddlePaddle

Change the following variables to fit your own environment
'''

#!/bin/bash
#SBATCH --job-name=deeplabv3p
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
conda activate paddle_x3.9  # specify your conda environment name
module load cuda            # load cuda module on demand

export CUDA_VISIBLE_DEVICES=0,1,2,3 # use 4 GPUs

'''
# config the following environment variables to enable parallel graph mode 
# and nccl2 distributed training mode on demand

export FLAGS_sync_nccl_allreduce=1 # use nccl to do allreduce
export FLAGS_enable_parallel_graph=1 # enable parallel graph mode
'''

# distribute training on 4 GPUs
# if you only have 1 GPU, remove the '-m paddle.distributed.launch' command
python -m paddle.distributed.launch --log_dir=output_deeplabv3p deeplabv3p.py
