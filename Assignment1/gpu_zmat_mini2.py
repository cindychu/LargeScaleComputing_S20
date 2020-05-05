#gpu_zmat_mini2.py

import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import pyopencl.clrandom as clrand
import pyopencl.tools as cltools
from pyopencl.scan import GenericScanKernel
import matplotlib.pyplot as plt
import time
import scipy.stats as sts
from scipy.optimize import minimize

#function to calculate value to be minimized
def mini_parallel(x):
  rho=x
  prefix_sum(ran,seg_boundary_flags,dev_result,rho,mu)
  health_index_all = (dev_result.get().reshape(n_runs, n_steps))
    
  # Find the positive or negative values
  t_all=[]
  for s in health_index_all:
    if 1 in s:
      s=list(s)
      t=s.index(1)
    else:
      t=n_steps
    t_all.append(t)
  #print(len(t_all))
  avg_t=sum(t_all)/len(t_all)
  print(rho,avg_t)
  return -avg_t

#set GPU environment
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mem_pool = cltools.MemoryPool(cltools.ImmediateAllocator(queue))

t0 = time.time()

#set parameters 
mu=3.0
z_0=mu
sigma=1.0
n_steps=int(4160)
n_runs=1000

#generate random numbers 
rand_gen = clrand.PhiloxGenerator(ctx)
ran = rand_gen.normal(queue, (n_runs*n_steps), np.float32, mu=0, sigma=1.0)

# Establish boundaries for each simulated walk (i.e. start and end)
seg_boundaries = [1] + [0]*(n_steps-1)
seg_boundaries = np.array(seg_boundaries, dtype=np.uint8)
seg_boundary_flags = np.tile(seg_boundaries, int(n_runs))
seg_boundary_flags = cl_array.to_device(queue, seg_boundary_flags)

# GPU: Define Segmented Scan Kernel, scanning simulations
prefix_sum = GenericScanKernel(ctx, np.float32,
            arguments="__global float *ary, __global char *segflags, "
                "__global float *out, float rho, float mu",
            input_expr="segflags[i] ? (ary[i]+mu):(ary[i]+(1-rho)*mu)",
            scan_expr="across_seg_boundary ? (b):(rho*a+b)", neutral="0",
            is_segment_start_expr="segflags[i]",
            output_statement="out[i] =(item>0) ? (0):(1)",
            options=[])

#set memory for results
dev_result = cl_array.arange(queue, len(ran), dtype=np.float32, allocator=mem_pool)

#set parameters and call minimize function
x0=np.zeros(1)
x0[0]=0.1  #initial set-up of rho
xmin=-0.95
xmax=0.95
rhomini=minimize(mini_parallel,x0,method='COBYLA',bounds=((xmin,xmax),),options={'rhobeg':0.01})

#report the results after minimization
final_time = time.time()
time_elapsed = final_time - t0
print("After Final Calculation: %f seconds" % (time_elapsed))
print("Maximized avged period: %f" % (-rhomini.fun))
print("Rho for Maximized avg period: %f" % (rhomini.x))