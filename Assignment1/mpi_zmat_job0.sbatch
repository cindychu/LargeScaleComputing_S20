#!/bin/bash

#SBATCH --job-name=mpi_zmat
#SBATCH --ntasks=1
#SBATCH --partition=broadwl
#SBATCH --constraint=fdr

# Load the default mpi4py/Anaconda module.
module load mpi4py/3.0.1a0_py3

# Run the python program with mpirun, using & to run jobs at the same time
mpirun -n 1 python ./mpi_zmat.py > ./mpi_nprocs_plot.out 

# Wait until all simultaneous mpiruns are done
#wait