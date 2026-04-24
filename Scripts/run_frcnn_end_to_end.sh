#!/bin/bash
# =============================================================================
# Variation 1 -- End-to-end fine-tuning (backbone + head, differential LR)
#
# All model parts train jointly. Backbone receives lr_backbone (slow) to
# preserve COCO-pretrained representations; FPN, RPN, and custom DNN head
# train at lr_head (fast). Custom DNN box predictor active.
#
# Use-case: primary experiment; expected to achieve the best overall mAP
#           given sufficient compute.
# =============================================================================
#SBATCH --job-name=frcnn_e2e
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --gres=gpu:h200:1  # swap to this if H200 nodes are available
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/home/<username>/visdrone/logs/frcnn_e2e_%j.out
#SBATCH --error=/home/<username>/visdrone/logs/frcnn_e2e_%j.err

source /shared/centos7/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate visdrone
module load cuda/12.1

python ~/visdrone/train_resnet50_frcnn_dnn.py \
    --head          custom      \
    --training_mode end_to_end  \
    --epochs        30          \
    --batch_size    4           \
    --img_max_size  1024        \
    --lr_backbone   5e-5        \
    --lr_head       5e-4        \
    --num_workers   4
