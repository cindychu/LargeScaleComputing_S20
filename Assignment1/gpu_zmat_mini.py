#gup_zmat_mini.py
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import pyopencl.clrandom as clrand
import pyopencl.tools as cltools
from pyopencl.scan import GenericScanKernel
import matplotlib.pyplot as plt
import time

def sim_health_index(n_runs):
  # Set up OpenCL context and command queue
  ctx = cl.create_some_context()
  queue = cl.CommandQueue(ctx)
  mem_pool = cltools.MemoryPool(cltools.ImmediateAllocator(queue))

  t0 = time.time()

  rho=0.5
  mu=3.0
  sigma=1.0
  z_0=mu

  # Generate an array of Normal Random Numbers on GPU of length n_sims*n_steps
  n_steps = int(4160) #4160
  rand_gen = clrand.PhiloxGenerator(ctx)
  ran = rand_gen.normal(queue, (n_runs*n_steps), np.float32, mu=0, sigma=1.0)

  # Establish boundaries for each simulated walk (i.e. start and end)
  # Necessary so that we perform scan only within rand walks and not between
  seg_boundaries = [1] + [0]*(n_steps-1)
  seg_boundaries = np.array(seg_boundaries, dtype=np.uint8)
  seg_boundary_flags = np.tile(seg_boundaries, int(n_runs))
  seg_boundary_flags = cl_array.to_device(queue, seg_boundary_flags)

  # GPU: Define Segmented Scan Kernel, scanning simulations: rho*f(n-1)+(1-rho)*mu+ran
  # also output whether the value is smaller than 0 or not
  prefix_sum = GenericScanKernel(ctx, np.float32,
              arguments="__global float *ary, __global char *segflags, "
                  "__global float *out, float rho, float mu",
              input_expr="segflags[i] ? (ary[i]+mu):(ary[i]+(1-rho)*mu)",
              scan_expr="across_seg_boundary ? (b):(rho*a+b)", neutral="0",
              is_segment_start_expr="segflags[i]",
              output_statement="out[i] =(item>0) ? (0):(1)",
              options=[])
  
  dev_result = cl_array.arange(queue, len(ran), dtype=np.float32, allocator=mem_pool)

  # print time of GPU simulation
  #sim_time = time.time()
  #time_elapsed = sim_time - t0
  #print("Simulated %d Health Index in: %f seconds"% (n_runs, time_elapsed))

  # Iterate For 200 rho values
  rho_set=np.linspace(-0.95,0.95,200)
  rho_avgt_t=[]
  for rho in rho_set:
    #Enqueue and Run Scan Kernel
    #print(rho)
    prefix_sum(ran,seg_boundary_flags,dev_result,rho,mu)
    # Get results back on CPU to plot and do final calcs, just as in Lab 1
    health_index_all = (dev_result.get().reshape(n_runs, n_steps))
    
    # Find and averaged the index of first negative values across simulations
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
    rho_avgt_t.append(avg_t)

  final_time = time.time()
  time_elapsed = final_time - t0
  print("Simulated %d Health Index for 200 rho values in: %f seconds"% (n_runs, time_elapsed))

  plt.plot(rho_set,rho_avgt_t)
  plt.title('Averaged periods of first negative index across Rho')
  plt.xlabel('Rho')
  plt.ylabel('Avged Period of first negative index')
  plt.savefig("GPU_rho_avgt_nruns%d.png" % (n_runs))

  max_period=max(rho_avgt_t)
  max_rho=rho_set[rho_avgt_t.index(max_period)]
  print("Max Period: %f; Max Rho: %f." % (max_period,max_rho))
  return

def main():
    sim_health_index(n_runs = 1000)

if __name__ == '__main__':
    main()