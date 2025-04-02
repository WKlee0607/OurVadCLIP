#!/usr/bin/bash

#SBATCH -J Train_XD_ver2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -w aurora-g8
#SBATCH -t 3-0
#SBATCH -o logs/slurm-%A.out

pwd
which python
export CUDA_VISIBLE_DEVICES="0"
#torchrun --nnodes=1 --nproc_per_node=1 ./src/xd_train.py
python ./src/xd_train.py
exit 0

