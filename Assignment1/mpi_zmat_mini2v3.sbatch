#!/bin/bash

#SBATCH --job-name=mpi_mini2
#SBATCH --output=mpi_mini2v3.out
#SBATCH --ntasks=3
#SBATCH --partition=broadwl
#SBATCH --constraint=fdr

# Load the default mpi4py/Anaconda module.
module load mpi4py/3.0.1a0_py3

# Run the python program with mpirun, using & to run jobs at the same time
#mpirun -n 1 python ./mpi_zmat.py > ./mpi_nprocs01.out &
#mpirun -n 2 python ./mpi_zmat.py > ./mpi_nprocs02.out &
#mpirun -n 3 python ./mpi_zmat.py > ./mpi_nprocs03.out &
mpirun python ./mpi_zmat_mini2_v3.py

# Wait until all simultaneous mpiruns are done
#wait