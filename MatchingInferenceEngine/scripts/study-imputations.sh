#!/bin/bash

#SBATCH --job-name=eduranker_imputations                             
#SBATCH --nodes=1                    
#SBATCH --cpus-per-task=8             
#SBATCH --mem=8GB                     
#SBATCH --time=40:10:00             
#SBATCH --account=torch_pr_594_general
#SBATCH --array=0-99
#SBATCH --output=/scratch/rm6609/EduRanker/MatchingInferenceEngine/experiment-results/mass-sim-logs/job_%A_%a.log
#SBATCH --mail-user=rm6609@nyu.edu

IMPUTATION_DIR="/scratch/rm6609/EduRanker/MatchingInferenceEngine/sample-data/data/master_data_04_residential_district_random_imputations"
SEED=40
TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')

# Get the list of files and sort them
FILES=($(ls -1 "$IMPUTATION_DIR"/imputed_seed_*.csv | sort))

# Get the file for this array task
DF_FILE="${FILES[$SLURM_ARRAY_TASK_ID]}"

echo "========================================"
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "Job Start: $TIMESTAMP | Seed: $SEED"
echo "Data File: $DF_FILE"
echo "========================================"

singularity exec --fakeroot --overlay /scratch/rm6609/research/overlay-15GB-500K.ext3:ro /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif /bin/bash -c "source /ext3/env.sh && conda activate research && time python3 /scratch/rm6609/EduRanker/MatchingInferenceEngine/src/real_experiment_driver.py --seed $SEED --df-filepath $DF_FILE"

echo "Job End: $(date '+%Y-%m-%d_%H-%M-%S')"
