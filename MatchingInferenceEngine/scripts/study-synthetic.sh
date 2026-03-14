#!/bin/bash

#SBATCH --job-name=eduranker_main_synthetic           
#SBATCH --array=0-99                  
#SBATCH --nodes=1                    
#SBATCH --cpus-per-task=8             
#SBATCH --mem=8GB                     
#SBATCH --time=24:00:00             
#SBATCH --account=torch_pr_594_general
#SBATCH --output=/scratch/rm6609/EduRanker/MatchingInferenceEngine/experiment-results/mass-sim-logs/job_%A_%a.log
#SBATCH --mail-user=rm6609@nyu.edu

SEED=$((40 + SLURM_ARRAY_TASK_ID))
TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')

# Stagger job starts to avoid overlay locking conflicts
DELAY=$((SLURM_ARRAY_TASK_ID * 10))
sleep $DELAY

echo "========================================"
echo "Job Start: $TIMESTAMP | Seed: $SEED"
echo "========================================"

singularity exec --fakeroot --overlay /scratch/rm6609/research/overlay-15GB-500K.ext3:ro /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif /bin/bash -c "source /ext3/env.sh && conda activate research && time python3 /scratch/rm6609/EduRanker/MatchingInferenceEngine/src/synthetic_experiments_driver.py --seed $SEED"

echo "Job End: $(date '+%Y-%m-%d_%H-%M-%S')"
