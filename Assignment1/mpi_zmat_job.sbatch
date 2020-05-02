#!/bin/bash

#SBATCH --job-name=mpi_zmat
#SBATCH --ntasks=210
#SBATCH --partition=broadwl
#SBATCH --constraint=fdr

# Load the default mpi4py/Anaconda module.
module load mpi4py/3.0.1a0_py3

# Run the python program with mpirun, using & to run jobs at the same time
mpirun -n 1 python ./mpi_zmat.py > ./mpi_nprocs01.out &
mpirun -n 2 python ./mpi_zmat.py > ./mpi_nprocs02.out &
mpirun -n 3 python ./mpi_zmat.py > ./mpi_nprocs03.out &
mpirun -n 4 python ./mpi_zmat.py > ./mpi_nprocs04.out &
mpirun -n 5 python ./mpi_zmat.py > ./mpi_nprocs05.out &
mpirun -n 6 python ./mpi_zmat.py > ./mpi_nprocs06.out &
mpirun -n 7 python ./mpi_zmat.py > ./mpi_nprocs07.out &
mpirun -n 8 python ./mpi_zmat.py > ./mpi_nprocs08.out &
mpirun -n 9 python ./mpi_zmat.py > ./mpi_nprocs09.out &
mpirun -n 10 python ./mpi_zmat.py > ./mpi_nprocs10.out &
mpirun -n 11 python ./mpi_zmat.py > ./mpi_nprocs11.out &
mpirun -n 12 python ./mpi_zmat.py > ./mpi_nprocs12.out &
mpirun -n 13 python ./mpi_zmat.py > ./mpi_nprocs13.out &
mpirun -n 14 python ./mpi_zmat.py > ./mpi_nprocs14.out &
mpirun -n 15 python ./mpi_zmat.py > ./mpi_nprocs15.out &
mpirun -n 16 python ./mpi_zmat.py > ./mpi_nprocs16.out &
mpirun -n 17 python ./mpi_zmat.py > ./mpi_nprocs17.out &
mpirun -n 18 python ./mpi_zmat.py > ./mpi_nprocs18.out &
mpirun -n 19 python ./mpi_zmat.py > ./mpi_nprocs19.out &
mpirun -n 20 python ./mpi_zmat.py > ./mpi_nprocs20.out &

# Wait until all simultaneous mpiruns are done
wait