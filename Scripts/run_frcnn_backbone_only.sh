#!/bin/bash
# =============================================================================
# Variation 3 -- Backbone only fine-tuning, standard head frozen
#
# FPN, RPN, and the standard FastRCNNPredictor head are fully frozen at their
# COCO-pretrained weights. Only the ResNet-50 backbone is fine-tuned on
# VisDrone at lr_backbone. No custom DNN head -- standard head used
# throughout so that only backbone adaptation is being measured.
#
# Use-case: ablation -- isolates how much backbone domain adaptation
#           contributes independent of any head architecture changes.
# =============================================================================
#SBATCH --job-name=frcnn_bb_only
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --gres=gpu:h200:1  # swap to this if H200 nodes are available
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/home/<username>/visdrone/logs/frcnn_bb_only_%j.out
#SBATCH --error=/home/<username>/visdrone/logs/frcnn_bb_only_%j.err

source /shared/centos7/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate visdrone
module load cuda/12.1

python ~/visdrone/train_resnet50_frcnn_dnn.py \
    --head          standard       \
    --training_mode backbone_only  \
    --epochs        30             \
    --batch_size    4              \
    --img_max_size  1024           \
    --lr_backbone   5e-5           \
    --num_workers   4
