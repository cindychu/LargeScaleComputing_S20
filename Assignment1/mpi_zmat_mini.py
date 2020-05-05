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

    # Start time:
    t0 = time.time()

    #set basic parameters
    #rho=0.5
    mu=3.0
    sigma=1.0
    z_0=mu
    T=int(4160)
    S =1000  #1000 
    z_mat=np.zeros((T,S))
    z_mat[0,:]=z_0

    if rank==0:
        #set random seed
        np.random.seed(25)
        eps_mat=sts.norm.rvs(loc=0,scale=sigma,size=(T,S))
    else:
        eps_mat=np.empty([T,S],dtype='float')
    comm.Bcast(eps_mat,root=0)

    #distribute rho
    N=int(n_runs/size)
    if rank==0:
        rho_set0=np.linspace(-0.95,0.95,n_runs)
        #print('Before Scatter: process %d has %s' % (rank, rho_set0))
    else:
        rho_set0=None
        #print('Before Scatter: process %d has %s' % (rank, rho_set0))

    rho_set=np.empty(N,dtype='float')
    comm.Scatter(rho_set0,rho_set,root=0)
    #print('After Scatter: process %d has %s' % (rank, rho_set))

    # Simulate S random walks by N rhos and specify as a NumPy Array
    rho_avgt=[]
    for rho in rho_set:
        avg_t=[]
        for s_ind in range(S):
            z_tm1=z_0
            i=0
            for t_ind in range(T):
                i=i+1
                e_t=eps_mat[t_ind,s_ind]
                z_t=rho*z_tm1+(1-rho)*mu+e_t
                z_mat[t_ind,s_ind]=z_t
                if z_t<=0:
                    avg_t.append(i)
                    #print(i,z_t)
                    break
                if i==T:
                    avg_t.append(i)
                z_tm1=z_t
        avgt=sum(avg_t)/len(avg_t)
        rho_avgt.append((rho,avgt))
    rho_avgt_array=np.array(rho_avgt)

    # Gather all simulation arrays to buffer of expected size/dtype on rank 0
    rho_avgt_all = None
    if rank == 0:
        rho_avgt_all = np.empty([N*size, 2], dtype='float')
    comm.Gather(sendbuf = rho_avgt_array, recvbuf = rho_avgt_all, root=0)

    # Print/plot simulation results on rank 0
    if rank == 0:
        # Calculate time elapsed after computing mean and std
        #average_finish = np.mean(r_walks_all[:,-1])
        #std_finish = np.std(r_walks_all[:,-1])
        time_elapsed = time.time() - t0
        x=[ a for (a,b) in rho_avgt_all]
        y=[ b for (a,b) in rho_avgt_all]
        max_periods=max(y)
        max_rho=x[y.index(max_periods)]
        

        # Print time elapsed + simulation results
        print("Simulated %d in: %f seconds on %d MPI processes" % (n_runs, time_elapsed, size))
        #print("Average final position: %f, Standard Deviation: %f"
                #% (average_finish, std_finish))
        print("Max Period: %f; Max Rho: %f." % (max_periods,max_rho))

        # Plot Simulations and save to file
        #x=[ a for (a,b) in rho_avgt_all]
        #y=[ b for (a,b) in rho_avgt_all]
        plt.plot(x,y)
        plt.title('Averaged periods of first negative index across Rho')
        plt.xlabel('Rho')
        plt.ylabel('Avged Period')
        #plt.plot(r_walks_all.transpose())
        plt.savefig("rho_avgt%d_nruns%d.png" % (size, n_runs))

def main():
    sim_index_parallel(n_runs = 200)  #200

if __name__ == '__main__':
    main()