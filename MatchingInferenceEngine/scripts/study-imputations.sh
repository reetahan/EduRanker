#!/bin/bash

#SBATCH --job-name=eduranker_imputations                             
#SBATCH --nodes=1                    
#SBATCH --cpus-per-task=64             
#SBATCH --mem=8GB                     
#SBATCH --time=40:10:00             
#SBATCH --account=torch_pr_594_general
#SBATCH --array=0-99
#SBATCH --output=/scratch/rm6609/EduRanker/MatchingInferenceEngine/experiment-results/mass-sim-logs/job_%A_%a.log
#SBATCH --mail-user=rm6609@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

IMPUTATION_DIR="/scratch/rm6609/EduRanker/MatchingInferenceEngine/sample-data/data/master_data_04_residential_district_random_imputations"
SEED=40
K=6
M=10
MAX_ITER=10
MAX_ITER_OPT=10
N_JOBS=64
TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
OVERLAY="/scratch/rm6609/research/overlay-persistent-manual.ext3"

# Get the list of files and sort them
FILES=($(ls -1 "$IMPUTATION_DIR"/imputed_seed_*.csv | sort))

# Get the file for this array task
DF_FILE="${FILES[$SLURM_ARRAY_TASK_ID]}"

echo "========================================"
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "Job Start: $TIMESTAMP | Seed: $SEED"
echo "K=$K | M=$M | max_iter=$MAX_ITER | max_iter_opt=$MAX_ITER_OPT | n_jobs=$N_JOBS"
echo "Data File: $DF_FILE"
echo "========================================"

singularity exec --fakeroot --overlay "$OVERLAY:ro" \
/share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif \
/bin/bash -c "
	source /ext3/env.sh
	cd /scratch/rm6609/EduRanker/MatchingInferenceEngine
	python3 src/real_experiment_driver.py --seed $SEED --K $K --M $M --max_iter $MAX_ITER --max_iter_opt $MAX_ITER_OPT --n_jobs $N_JOBS --df-filepath \"$DF_FILE\"
"

echo "Job End: $(date '+%Y-%m-%d_%H-%M-%S')"
