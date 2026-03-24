#!/bin/bash

#SBATCH --job-name=eduranker_main_real                             
#SBATCH --nodes=1                    
#SBATCH --cpus-per-task=64            
#SBATCH --mem=8GB                     
#SBATCH --time=40:10:00             
#SBATCH --account=torch_pr_594_tandon_priority
#SBATCH --output=/scratch/rm6609/EduRanker/MatchingInferenceEngine/experiment-results/mass-sim-logs/job_%A_%a.log
#SBATCH --mail-user=rm6609@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

SEED=40
K=4
M=4
MAX_ITER=4
MAX_ITER_OPT=4
N_JOBS=64
TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')


echo "========================================"
echo "Job Start: $TIMESTAMP | Seed: $SEED"
echo "========================================"

OVERLAY="/scratch/rm6609/research/overlay-persistent-manual.ext3"

singularity exec --fakeroot --overlay "$OVERLAY:ro" \
/share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif \
/bin/bash -c "
    source /ext3/env.sh
    cd /scratch/rm6609/EduRanker/MatchingInferenceEngine
    python3 src/real_experiment_driver.py --seed $SEED --K $K --M $M --max_iter $MAX_ITER --max_iter_opt $MAX_ITER_OPT --n_jobs $N_JOBS
"

echo "Job End: $(date '+%Y-%m-%d_%H-%M-%S')"
