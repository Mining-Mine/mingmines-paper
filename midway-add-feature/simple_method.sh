#!/bin/bash

#SBATCH --job-name=simple_method
#SBATCH --output=simple_method.out
#SBATCH --error=simple_method.err
#SBATCH --ntasks=10
#SBATCH --partition=caslake
#SBATCH --account=macs30123
#SBATCH --time=01:00:00
#SBATCH --mem=32G

module load python
source /home/kaiwen1/30123-Project-Kaiwen/myenv/bin/activate

# Run the Python script using mpiexec
mpiexec -n 10 python Simple_method.py