#!/bin/bash
#SBATCH --job-name=tsgnn-train
#SBATCH --output=logs/tsgnn-train_%j.out
#SBATCH --error=logs/tsgnn-train_%j.err
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH --container-image ghcr.io\#arslanardavic/equivarianceeverywhere

set -euo pipefail

# Work from the directory where you ran `sbatch train.sbatch`
mkdir -p logs
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Run your training
python -u main.py \
  --is_train \
  --train_test_setup trainset1 \
  --gnn_type MEAN_GNN \
  --hidden_dim 16 \
  --num_layers 2 \
  --lp_ratio 0.4 \
  --max_epochs 2000 \
  --lr 0.01
