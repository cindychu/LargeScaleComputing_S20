from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np
import time
import scipy.stats as sts 

def sim_index_parallel(n_runs):
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	size = comm.Get_size()

    #set basic parameters
    rho=0.5
    mu=3.0
    sigma=1.0
    z_0=mu
    T=int(4160)

    #set random seed
    #np.random.seed(25)

    # Start time:
    t0 = time.time()

    N=int(n_runs/size)

    if rank==0:
    	data=np.linspace(-0.95,0.95,10)
    	print('Before Scatter: process %d has %s' % (rank, data))
    else:
    	data=np.empty(N,dtype='float')
    	print('Before Scatter: process %d has %s' % (rank, data))

    data=comm.Scatter(data,root=0)
    print('Before Scatter: process %d has %s' % (rank, data))

    time_elapsed = time.time() - t0
    print("Simulated in: %f seconds on %d MPI processes"
                % (time_elapsed, size))

def main():
    sim_index_parallel(n_runs = 10)

if __name__ == '__main__':
    main()