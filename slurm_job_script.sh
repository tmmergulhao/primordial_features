#!/bin/bash -l
               
##############################
#   MCMC SAMPLING JOB SCRIPT  #
##############################

# Define your job name here so that you can recognize it in the queue.
#SBATCH --job-name=mcmc_sampling

# Define the output file name (%j at end will append the job id at the end of the name). This will store the standard output 'stdout' for the job. 
#SBATCH -o /home/dc-merg1/primordial_features/log/mcmc_output_%j.txt

# Define file name for storing any errors (stderr). 
#SBATCH -e /home/dc-merg1/primordial_features/log/mcmc_error_%j.txt

# Define the partition on which you want to run.
#SBATCH -p slurm

# Define the Account/Project name from which the computation would be charged. 
#SBATCH -A dp322

# Define how many nodes you need. Here, we ask for 1 node.
#SBATCH --nodes=1

# Define the number of tasks (CPUs) for the MCMC sampling.
#SBATCH --ntasks=32  # Adjust as necessary for your MCMC workload

# Define how long the job will run in real time. Adjust as needed.
#SBATCH --time=03:00:00

# Memory requirements; adjust according to your needs.
#SBATCH --mem-per-cpu=2000MB  # Adjust as necessary

# Execute your MCMC sampling script using multiprocessing
export OMP_NUM_THREADS=1
conda activate emcee
cd /home/dc-merg1/primordial_features
python3 main.py --env envs/DESI_MOCK_lin.env --mock 1 --omega_min 2900 --omega_max 4000
exit 0
