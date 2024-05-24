#!/bin/bash

#SBATCH --job-name=calculate_distances
#SBATCH --output=calculate_distances.out
#SBATCH --error=calculate_distances.err
#SBATCH --ntasks=10
#SBATCH --partition=caslake
#SBATCH --account=macs30123
#SBATCH --time=01:00:00
#SBATCH --mem=32G

module load python
source /home/kaiwen1/30123-Project-Kaiwen/myenv/bin/activate

# Run the Python script using mpiexec
mpiexec -n 10 python congo_mine_feature_adding.py