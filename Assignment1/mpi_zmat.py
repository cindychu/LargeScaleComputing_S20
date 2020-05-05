from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np
import time
import scipy.stats as sts 

def sim_index_parallel(n_runs):
    # Get rank of process and overall size of communicator:
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
    np.random.seed(25)

    # Start time:
    t0 = time.time()

    # Evenly distribute number of simulation runs across processes
    N = int(n_runs/size)
    eps_mat=sts.norm.rvs(loc=0,scale=sigma,size=(T,N))
    z_mat=np.zeros((T,N))
    z_mat[0,:]=z_0
    # Simulate N random walks and specify as a NumPy Array

    for s_ind in range(N):
    	z_tm1=z_0
    	for t_ind in range(T):
    		e_t=eps_mat[t_ind,s_ind]
    		z_t=rho*z_tm1+(1-rho)*mu+e_t
    		z_mat[t_ind,s_ind]=z_t
    		z_tm1=z_t
    z_mat_array=np.array(z_mat)

    # Gather all simulation arrays to buffer of expected size/dtype on rank 0
    z_mat_all = None
    if rank == 0:
        z_mat_all = np.empty([T, N*size], dtype='float')
    comm.Gather(sendbuf = z_mat_array, recvbuf = z_mat_all, root=0)

    # Print/plot simulation results on rank 0
    if rank == 0:
        # Calculate time elapsed after computing mean and std
        #average_finish = np.mean(r_walks_all[:,-1])
        #std_finish = np.std(r_walks_all[:,-1])
        time_elapsed = time.time() - t0

        # Print time elapsed + simulation results
        print("Simulated %d Health Index in: %f seconds on %d MPI processes"
                % (n_runs, time_elapsed, size))
        #print("Average final position: %f, Standard Deviation: %f"
                #% (average_finish, std_finish))

        # Plot Simulations and save to file
        plt.plot(z_mat_all)
        plt.savefig("health_nprocs%d_nruns%d.png" % (size, n_runs))

def main():
    sim_index_parallel(n_runs = 1000)

if __name__ == '__main__':
    main()