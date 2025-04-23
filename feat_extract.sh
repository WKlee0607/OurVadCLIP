#!/usr/bin/bash

#SBATCH -J Feat_extract_UCF
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -w aurora-g8
#SBATCH -t 3-0
#SBATCH -o logs/slurm-%A.out

pwd
which python

export CUDA_VISIBLE_DEVICES="0,1,2,3"
python ./extract_feat/main_ucf_vgg.py

exit 0

