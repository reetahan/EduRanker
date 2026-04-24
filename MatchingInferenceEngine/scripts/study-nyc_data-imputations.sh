#!/bin/bash

#SBATCH --job-name=eduranker_imputations                             
#SBATCH --nodes=1                    
#SBATCH --cpus-per-task=64             
#SBATCH --mem=8GB                     
#SBATCH --time=40:10:00             
#SBATCH --account=torch_pr_594_tandon_priority
#SBATCH --array=0-1699
#SBATCH --output=/scratch/rm6609/EduRanker/MatchingInferenceEngine/experiment-results/mass-sim-logs/job_%A_%a.log
#SBATCH --mail-user=rm6609@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

IMPUTATION_DIR="/scratch/rm6609/EduRanker/MatchingInferenceEngine/sample-data/data/master_data_04_residential_district_random_imputations"
SEED=40
# Baseline values for non-varied parameters
K=4
M=4
MAX_ITER=4
MAX_ITER_OPT=4
N_JOBS=64
TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
OVERLAY="/scratch/rm6609/research/overlay-persistent-manual.ext3"
IMAGE="/share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif"
WORKDIR="/scratch/rm6609/EduRanker/MatchingInferenceEngine"
CONDA_ENV="research"
#PARAMS=("K" "M" "MAX_ITER" "MAX_ITER_OPT")
PARAMS=("K")
VALUES=(4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
N_IMPUTATIONS=100
N_PARAMS=${#PARAMS[@]}
N_VALUES=${#VALUES[@]}
TOTAL_TASKS=$((N_PARAMS * N_VALUES * N_IMPUTATIONS))

# Get the list of files and sort them
mapfile -t FILES < <(find "$IMPUTATION_DIR" -maxdepth 1 -type f -name 'imputed_seed_*.csv' | sort)

if [ ${#FILES[@]} -lt $N_IMPUTATIONS ]; then
	echo "ERROR: Expected at least $N_IMPUTATIONS imputations, found ${#FILES[@]} in $IMPUTATION_DIR"
	exit 1
fi

if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL_TASKS" ]; then
	echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID exceeds TOTAL_TASKS-1=$((TOTAL_TASKS - 1))"
	exit 1
fi

# Map each array task to one parameter/value pair and one imputation index
IMPUTATION_INDEX=$((SLURM_ARRAY_TASK_ID % N_IMPUTATIONS))
VALUE_INDEX=$(((SLURM_ARRAY_TASK_ID / N_IMPUTATIONS) % N_VALUES))
PARAM_INDEX=$((SLURM_ARRAY_TASK_ID / (N_IMPUTATIONS * N_VALUES)))
VARIED_PARAM="${PARAMS[$PARAM_INDEX]}"
VARIED_VALUE="${VALUES[$VALUE_INDEX]}"

case "$VARIED_PARAM" in
	K)
		K=$VARIED_VALUE
		;;
	M)
		M=$VARIED_VALUE
		;;
	MAX_ITER)
		MAX_ITER=$VARIED_VALUE
		;;
	MAX_ITER_OPT)
		MAX_ITER_OPT=$VARIED_VALUE
		;;
	*)
		echo "ERROR: Unknown VARIED_PARAM=$VARIED_PARAM"
		exit 1
		;;
esac

# Get the file for this task's imputation index
DF_FILE="${FILES[$IMPUTATION_INDEX]}"

echo "========================================"
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "Job Start: $TIMESTAMP | Seed: $SEED"
echo "Varied Parameter: $VARIED_PARAM"
echo "Varied Value: $VARIED_VALUE"
echo "Imputation Index: $IMPUTATION_INDEX"
echo "K=$K | M=$M | max_iter=$MAX_ITER | max_iter_opt=$MAX_ITER_OPT | n_jobs=$N_JOBS"
echo "Data File: $DF_FILE"
echo "========================================"

if [ ! -f "$OVERLAY" ]; then
	echo "ERROR: Overlay file not found: $OVERLAY"
	exit 1
fi

singularity exec --fakeroot --overlay "$OVERLAY:ro" \
"$IMAGE" \
/bin/bash -c "
	set -euo pipefail
	if [ -f /ext3/miniconda3/bin/activate ]; then
		source /ext3/miniconda3/bin/activate $CONDA_ENV
	elif [ -f /ext3/env.sh ]; then
		source /ext3/env.sh
		if command -v conda >/dev/null 2>&1; then
			conda activate $CONDA_ENV || true
		fi
	elif [ -f /ext3/miniconda3/etc/profile.d/conda.sh ]; then
		source /ext3/miniconda3/etc/profile.d/conda.sh
		conda activate $CONDA_ENV || true
	elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
		source /opt/conda/etc/profile.d/conda.sh
		conda activate $CONDA_ENV || true
	else
		echo 'ERROR: No conda bootstrap script found in container (/ext3/miniconda3/bin/activate, /ext3/env.sh, /ext3/miniconda3/etc/profile.d/conda.sh, /opt/conda/etc/profile.d/conda.sh).'
		echo 'DEBUG: Listing /ext3 (if present):'
		ls -la /ext3 2>/dev/null || true
		exit 2
	fi

	cd $WORKDIR
	echo 'Python executable:'
	command -v python3
	python3 -c 'import pandas' >/dev/null 2>&1 || { echo 'ERROR: pandas unavailable after conda activation. Fix overlay/env and retry.'; exit 3; }
	python3 src/real_experiment_driver.py --seed $SEED --K $K --M $M --max_iter $MAX_ITER --max_iter_opt $MAX_ITER_OPT --n_jobs $N_JOBS --df-filepath \"$DF_FILE\"
"

echo "Job End: $(date '+%Y-%m-%d_%H-%M-%S')"
