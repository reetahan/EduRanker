#!/bin/bash

#SBATCH --job-name=eduranker_main_real_chilean                             
#SBATCH --nodes=1                    
#SBATCH --cpus-per-task=8         
#SBATCH --mem=32GB                     
#SBATCH --time=40:10:00             
#SBATCH --account=torch_pr_594_tandon_priority
#SBATCH --output=/scratch/rm6609/EduRanker/MatchingInferenceEngine/experiment-results/mass-sim-logs/job_%A_%a.log
#SBATCH --mail-user=rm6609@nyu.edu

#SBATCH --mail-type=BEGIN,END,FAIL

SEED=40
K=6
M=15
MAX_ITER=20
MAX_ITER_OPT=12
N_JOBS=8
PROFILE_TIMING=1

TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')

PROFILE_ARG=""
if [[ "$PROFILE_TIMING" -eq 1 ]]; then
    PROFILE_ARG="--profile_timing"
fi


echo "========================================"
echo "Job Start: $TIMESTAMP | Seed: $SEED"
echo "Profile timing: $PROFILE_TIMING"
echo "========================================"

OVERLAY="/scratch/rm6609/research/overlay-persistent-manual.ext3"

singularity exec --fakeroot --overlay "$OVERLAY:ro" \
/share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif \
/bin/bash -c "
    source /ext3/miniconda3/bin/activate research
    cd /scratch/rm6609/EduRanker/MatchingInferenceEngine
    python3 src/chilean_experiment_driver_main.py --seed $SEED --K $K --M $M --max_iter $MAX_ITER --max_iter_opt $MAX_ITER_OPT --n_jobs $N_JOBS $PROFILE_ARG
"

echo "Job End: $(date '+%Y-%m-%d_%H-%M-%S')"
