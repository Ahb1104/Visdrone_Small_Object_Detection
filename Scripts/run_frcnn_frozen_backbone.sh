#!/bin/bash
# =============================================================================
# Variation 2 -- Frozen backbone, train FPN + RPN + custom DNN head
#
# COCO-pretrained ResNet-50 backbone is fully frozen. FPN, RPN, and custom
# DNN box predictor head are fine-tuned on VisDrone at lr_head. Isolates
# the contribution of the custom head architecture against Variation 1.
#
# Use-case: ablation -- quantifies how much of Variation 1's gain comes
#           from backbone adaptation vs. head architecture.
# =============================================================================
#SBATCH --job-name=frcnn_frozen_bb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --gres=gpu:h200:1  # swap to this if H200 nodes are available
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/home/<username>/visdrone/logs/frcnn_frozen_bb_%j.out
#SBATCH --error=/home/<username>/visdrone/logs/frcnn_frozen_bb_%j.err

source /shared/centos7/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate visdrone
module load cuda/12.1

python ~/visdrone/train_resnet50_frcnn_dnn.py \
    --head          custom           \
    --training_mode frozen_backbone  \
    --epochs        30               \
    --batch_size    4                \
    --img_max_size  1024             \
    --lr_head       5e-4             \
    --num_workers   4
