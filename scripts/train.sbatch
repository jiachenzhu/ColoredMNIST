#!/bin/bash
#SBATCH --job-name=CM
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1
#SBATCH --array=1
#SBATCH --time=10:00:00
#SBATCH --constraint="gpu_12gb"
#SBATCH --exclude="weaver[1-5], vine[3-14], lion[1-26]"
#SBATCH --export=ALL
#SBATCH --open-mode=append
#SBATCH --output=logs/train_%4a_%j.txt
#SBATCH --error=error_logs/train_%4a_%j.txt

eval "$(conda shell.bash hook)"
conda activate

python main.py configs/base_train.yml configs/experiment_$SLURM_ARRAY_TASK_ID.yml