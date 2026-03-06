#!/bin/bash

#SBATCH --job-name=eduranker_final
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8GB
#SBATCH --time=24:00:00
#SBATCH --account=torch_pr_594_general
#SBATCH --output=/scratch/rm6609/EduRanker/MatchingInferenceEngine/experiment-results/mass-sim-logs/job_final_analysis.log
#SBATCH --mail-user=rm6609@nyu.edu
#SBATCH --mail-type=END

TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')

# This job runs AFTER the array job completes (add dependency: --dependency=afterok:JOBID)
echo "========================================"
echo "Final Analysis Start: $TIMESTAMP"
echo "Running final analysis on accumulated results..."
echo "========================================"

singularity exec --fakeroot --overlay /scratch/rm6609/research/overlay-15GB-500K.ext3:ro /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif /bin/bash -c "source /ext3/env.sh && conda activate research && time python3 /scratch/rm6609/EduRanker/MatchingInferenceEngine/src/synthetic_experiments_driver.py --final-analysis"

echo "Final Analysis End: $(date '+%Y-%m-%d_%H-%M-%S')"
