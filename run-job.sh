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

# Use node-local scratch if available, otherwise /tmp
SCRATCH="${SLURM_TMPDIR:-/tmp}"

# Put all JIT/caches on exec-capable scratch
export TRITON_CACHE_DIR="$SCRATCH/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="$SCRATCH/torchinductor"
export XDG_CACHE_HOME="$SCRATCH/xdg_cache"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME"
chmod 700 "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME"

# Packages missed in the container can be installed here
python -m pip install --no-cache-dir ogb 

# Run your training
python -u main.py \
  --project ALLab-Boun/EquivarianceEverywhere-Reproduction\
  --train_test_setup trainset1 \
  --gnn_type MEAN_GNN \
  --num_layers 2 \
  --lp_ratio 0.4 \
  --max_epochs 2000 \
  --lr 0.01
