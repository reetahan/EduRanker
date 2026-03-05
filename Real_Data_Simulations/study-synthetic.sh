#!/bin/bash

#SBATCH --job-name=eduranker_main_synthetic           
#SBATCH --array=0-79                 
#SBATCH --nodes=1                    
#SBATCH --cpus-per-task=1             
#SBATCH --mem=8GB                     
#SBATCH --time=24:00:00             
#SBATCH --account=torch_pr_594_general
#SBATCH --output=/scratch/rm6609/EduRanker/Real_Data_Simulations/mass-sim-logs/job_%A_%a.log

SEED=$((40 + SLURM_ARRAY_TASK_ID))
TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')

# Stagger job starts to avoid overlay locking conflicts
DELAY=$((SLURM_ARRAY_TASK_ID * 10))
sleep $DELAY

echo "========================================" 2>&1 | tee -a /scratch/rm6609/EduRanker/Real_Data_Simulations/job_%A_%a.log
echo "Job Start: $TIMESTAMP | Seed: $SEED" 2>&1 | tee -a /scratch/rm6609/EduRanker/Real_Data_Simulations/job_%A_%a.log
echo "========================================" 2>&1 | tee -a /scratch/rm6609/EduRanker/Real_Data_Simulations/job_%A_%a.log

singularity exec --fakeroot --overlay /scratch/rm6609/research/overlay-15GB-500K.ext3:ro /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif /bin/bash -c "source /ext3/env.sh && conda activate research && time python3 /scratch/rm6609/EduRanker/Real_Data_Simulations/em_sim_data_GLOBAL.py --synthetic --seed $SEED" 2>&1 | tee -a /scratch/rm6609/EduRanker/Real_Data_Simulations/job_%A_%a.log

echo "Job End: $(date '+%Y-%m-%d_%H-%M-%S')" 2>&1 | tee -a /scratch/rm6609/EduRanker/Real_Data_Simulations/job_%A_%a.log
