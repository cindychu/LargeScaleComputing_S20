from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np
import time
import scipy.stats as sts
from scipy.optimize import minimize

def mini_parallel(x,stop):
  print('before scatter stop')
  stop[0]=comm.bcast(stop[0], root=0)
  print('after scatter stop')
  avgt=0
  if stop[0]==0:
    print("In minimize: proc%d, %f." % (rank,x[0]))
    print(np.shape(x))
    x[0]=comm.bcast(x[0],root=0)
    print('after scatter x')
    rho=x[0]
    #print(rho)
    #mu=3.0
    #sigma=1.0
    #T=int(4160)
    #z_0=mu

    #n_runs=1000
    #N=int(n_runs/size)
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
          #print(i)
          break
        z_tm1=z_t
    all_t_array=np.array(all_t)
    print(rho,np.shape(all_t_array))
    #print(np.mean(all_t_array))
  
    # Gather all simulation arrays to buffer of expected size/dtype on rank 0
    #if rank==0:
      #t_all = np.empty([N*size, 1], dtype='float')
    #comm.Gather(sendbuf = all_t_array, recvbuf = t_all, root=0)
    print('before gather results')
    all_t_array=comm.gather(all_t_array,root=0)
    print('after gather results')
    
    if rank==0:
      #avgt=sum(t_all)/len(t_all)
      print(rho,np.shape(all_t_array))
      avgt=np.mean(all_t_array)
      print(rho,avgt)
  
  return -avgt 


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

t0 = time.time()

n_runs=1000
N=int(n_runs/size)
T=int(4160)
mu=3.0
sigma=1.0
z_0=mu

stop=np.ones(1)
x=np.zeros(1)

if rank==0:
  np.random.seed(25)
  eps_mat0=sts.norm.rvs(loc=0,scale=sigma,size=(T,N*size))
else:
  eps_mat0=None

eps_mat=np.empty([T,N],dtype='float')
comm.Scatter(eps_mat0,eps_mat,root=0)
print('Scatter EPS successfully')

if rank==0:
  print("In minimize: proc%d, stop%f, ." % (rank,stop[0]))
  stop[0]=0
  x[0]=0.1
  xmin=-0.95
  xmax=0.95
  print('Before minimize')
  rhomin=minimize(mini_parallel,x,args=(stop),method='COBYLA',bounds=((xmin,xmax),),options={'rhobeg':0.01,'tol':0.00001})
  stop[0]=2 #stop
  mini_parallel(x,stop)
else:
  print("In minimize: proc%d, stop%f, ." % (rank,stop[0]))
  while stop[0]!=2:
    print("Before minimize: proc%d, %f." % (rank,x[0]))
    mini_parallel(x,stop)
  
if rank==0:
  print(rhomin.fun)
  print(rhomin.x)


          


