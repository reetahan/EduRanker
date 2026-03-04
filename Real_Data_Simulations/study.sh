#!/bin/bash

#SBATCH --job-name=eduranker_main           # The name of the job
#SBATCH --nodes=8                     # Request 1 compute node per job instance
#SBATCH --cpus-per-task=4             # Request 1 CPU per job instance
#SBATCH --mem=8GB                     # Request 2GB of RAM per job instance
#SBATCH --time=24:10:00               # Request 10 mins per job instance
#SBATCH --account=torch_pr_594_general
#SBATCH --output=/scratch/rm6609/EduRanker/Real_Data_Simulations/job_%j.log  # The output will be saved here. %j will be replaced by the slurm job ID
#SBATCH --mail-user=rm6609@nyu.edu   # Email address
#SBATCH --mail-type=END               # Send an email when all the instances of this job are completed

singularity exec --fakeroot --overlay /scratch/rm6609/research/overlay-15GB-500K.ext3:rw /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif /bin/bash -c "source /ext3/env.sh && conda activate research && time python3 /scratch/rm6609/EduRanker/Real_Data_Simulations/em_sim_data_GLOBAL.py"
