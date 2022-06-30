#!/bin/bash
#SBATCH --job-name=CM
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1
#SBATCH --array=325-648
#SBATCH --time=5:00:00
#SBATCH --export=ALL
#SBATCH --open-mode=append
#SBATCH --output=logs/train_%4a_%j.txt
#SBATCH --error=error_logs/train_%4a_%j.txt

srun singularity exec --nv --overlay $SCRATCH/overlay/conda.ext3:ro /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "
source /ext3/env.sh
conda activate
python main.py configs/base_train.yml configs/experiment_$SLURM_ARRAY_TASK_ID.yml
"