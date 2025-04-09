#!/usr/bin/bash

#SBATCH -J Visual_XD
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
python ./src/xd_visual_test.py 
exit 0

