from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np
import time
import scipy.stats as sts
from scipy.optimize import minimize

def mini_parallel(x,stop,size):
	comm.Bcast(stop,root=0)
	if stop[0]==0:
		comm.Bcast(x,root=0)
		rho=x[0]
		mu=3.0
		sigma=1.0
		z_0=mu
		T=int(4160)

		n_runs=10 #1000
		N = int(n_runs/size)

		# Evenly distribute number of simulation runs across processes
		np.random.seed(25)
		eps_mat=sts.norm.rvs(loc=0,scale=sigma,size=(T,N))
		z_mat=np.zeros((T,N))
		z_mat[0,:]=z_0

		# Simulate N random walks and specify as a NumPy Array
		all_t=[]
		for s_ind in range(N):
			z_tm1=z_0
			i=0
			for t_ind in range(T):
				i=i+1
				e_t=eps_mat[t_ind,s_ind]
				z_t=rho*z_tm1+(1-rho)*mu+e_t
				z_mat[t_ind,s_ind]=z_t
				if z_t<=0:
					all_t.append(i)
					break
				z_tm1=z_t
		all_t_array=np.array(all_t)

		# Gather all simulation arrays to buffer of expected size/dtype on rank 0
		#t_all = None
		if rank == 0:
			t_all = np.empty([N*size, 1], dtype='float')
		comm.Gather(sendbuf = all_t_array, recvbuf = t_all, root=0)

		if rank==0:
			avgt=np.mean(t_all)
			return -avgt

def sim_rho_parallel(n_runs):
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	size = comm.Get_size()

	t0 = time.time()
	#N=int(n_runs/size)

	stop=np.ones(1)
	x=np.zeros(1)
	
	if rank==0:
		stop[0]=0
		x[0]=0.1
		xmin=-0.95
		xmax=0.95
		rhomin=minimize(mini_parallel,x,args=(stop,size),method='COBYLA',bounds=((xmin,xmax),),options={'rhobeg':0.01})
		stop=[1]
		mini_parallel(x,stop)
	else:
		while stop[0]==0:
			mini_parallel(x,stop,size)

	if rank==0:
		#print(rhomini.fun)
		#print(rhomini.success)
		#print(rhomini.x)
		max_rho=rhomini.x
		max_periods=rhomini.fun
		
		# Print time elapsed + simulation results
		time_elapsed = time.time() - t0
		print("Simulated %d in: %f seconds on %d MPI processes" % (n_runs, time_elapsed, size))
		print("Max Period: %f; Max Rho: %f." % (max_periods,max_rho))			


def main():
	sim_rho_parallel(n_runs = 1000)

if __name__ == '__main__':
	main()
