#!/bin/bash -l

#SBATCH -J model_HMI_set
#SBATCH -t 4:00:00
#SBATCH -p cops
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64

ml unload conda/02
source ~/.local-co/bin/setup-environment.sh
ml prun

ulimit -s unlimited  # Unlimited stack
ulimit -u 16000      # Increase max number of tasks
ulimit -n 65536      # Increase max number of open files

prun python3 generate_spectra_mpi.py '/cfs/home/alju5794/steppz/sps_models/model_HMI/training_data/'
