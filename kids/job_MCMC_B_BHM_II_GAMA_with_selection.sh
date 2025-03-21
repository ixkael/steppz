#!/bin/bash -l

#SBATCH -J BIIGS
#SBATCH -t 48:00:00
#SBATCH -A cops
#SBATCH -p ampere
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1

ml unload conda/02
ml cuda
source ~/.local-co/bin/setup-environment.sh

ulimit -s unlimited  # Unlimited stack
ulimit -u 16000      # Increase max number of tasks
ulimit -n 65536      # Increase max number of open files

python3 MCMC_B_BHM_II_GAMA_with_selection.py
