#!/bin/bash

#SBATCH --job-name=eduranker_main           
#SBATCH --nodes=1                    
#SBATCH --cpus-per-task=4             
#SBATCH --mem=8GB                     
#SBATCH --time=24:10:00             
#SBATCH --account=torch_pr_594_general
#SBATCH --output=/scratch/rm6609/EduRanker/Real_Data_Simulations/job_%j.log  # The output will be saved here. %j will be replaced by the slurm job ID
#SBATCH --mail-user=rm6609@nyu.edu   # Email address
#SBATCH --mail-type=END               # Send an email when all the instances of this job are completed

TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')

echo "========================================" 2>&1 | tee -a /scratch/rm6609/EduRanker/Real_Data_Simulations/job_%j.log
echo "Real Data Analysis Start: $TIMESTAMP" 2>&1 | tee -a /scratch/rm6609/EduRanker/Real_Data_Simulations/job_%j.log
echo "========================================" 2>&1 | tee -a /scratch/rm6609/EduRanker/Real_Data_Simulations/job_%j.log

singularity exec --fakeroot --overlay /scratch/rm6609/research/overlay-15GB-500K.ext3:ro /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif /bin/bash -c "source /ext3/env.sh && conda activate research && time python3 /scratch/rm6609/EduRanker/Real_Data_Simulations/em_sim_data_GLOBAL.py" 2>&1 | tee -a /scratch/rm6609/EduRanker/Real_Data_Simulations/job_%j.log

echo "Real Data Analysis End: $(date '+%Y-%m-%d_%H-%M-%S')" 2>&1 | tee -a /scratch/rm6609/EduRanker/Real_Data_Simulations/job_%j.log
