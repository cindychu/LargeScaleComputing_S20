# gpu_zmat.py

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

  t0 = time.time()

  rho=0.5
  mu=3.0
  sigma=1.0
  z_0=mu

  # Generate an array of Normal Random Numbers on GPU of length n_sims*n_steps
  n_steps = int(4160)  #4160
  rand_gen = clrand.PhiloxGenerator(ctx)
  ran = rand_gen.normal(queue, (n_runs*n_steps), np.float32, mu=0, sigma=1.0)

  # Establish boundaries for each simulated walk (i.e. start and end)
  # Necessary so that we perform scan only within rand walks and not between
  seg_boundaries = [1] + [0]*(n_steps-1)
  seg_boundaries = np.array(seg_boundaries, dtype=np.uint8)
  seg_boundary_flags = np.tile(seg_boundaries, int(n_runs))
  seg_boundary_flags = cl_array.to_device(queue, seg_boundary_flags)

  # GPU: Define Segmented Scan Kernel, scanning simulations: f(n-1) + f(n)
  prefix_sum = GenericScanKernel(ctx, np.float32,
              arguments="__global float *ary, __global char *segflags, "
                  "__global float *out, float rho, float mu",
              input_expr="segflags[i] ? (ary[i]+mu):(ary[i]+(1-rho)*mu)",
              scan_expr="across_seg_boundary ? (b):(rho*a+b)", neutral="0",
              is_segment_start_expr="segflags[i]",
              output_statement="out[i] = item",
              options=[])
  
  dev_result = cl_array.empty_like(ran)

  # Enqueue and Run Scan Kernel
  prefix_sum(ran,seg_boundary_flags,dev_result,rho,mu)

  # Get results back on CPU to plot and do final calcs, just as in Lab 1
  health_index_all = (dev_result.get().reshape(n_runs, n_steps).transpose())

  final_time = time.time()
  time_elapsed = final_time - t0

  print("Simulated %d Health Index in: %f seconds"% (n_runs, time_elapsed))
  #print(health_index_all)
  #print(ran.reshape(n_runs, n_steps).transpose())
  #plt.plot(health_index_all)
  return

def main():
    sim_health_index(n_runs = 1000)

if __name__ == '__main__':
    main()