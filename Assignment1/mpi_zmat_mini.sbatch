#!/bin/bash

#SBATCH --job-name=mpi_mini
#SBATCH --output=mpi_mini.out
#SBATCH --ntasks=20
#SBATCH --partition=broadwl
#SBATCH --constraint=fdr

# Load the default mpi4py/Anaconda module.
module load mpi4py/3.0.1a0_py3

# Run the python program with mpirun, using & to run jobs at the same time
#mpirun -n 1 python ./mpi_zmat.py > ./mpi_nprocs01.out &
#mpirun -n 2 python ./mpi_zmat.py > ./mpi_nprocs02.out &
#mpirun -n 3 python ./mpi_zmat.py > ./mpi_nprocs03.out &
mpirun python ./mpi_zmat_mini.py

# Wait until all simultaneous mpiruns are done
#wait