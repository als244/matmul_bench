# Matmul Benchmarking
### Using cublasLt

## Directions

1. Download Source
2. Run `make`
3. Start Program from Command Line:
```./benchCublasLtMatmul <M> <K> <N> <dtype_fp_bits> <log2_workspace_bytes> <n_matmuls> <n_warmup> <n_compute_iters> <n_sms> <use_same_b_matrix> <use_c_matrix> <use_fp16_accum>```
	- Instead, can optionally Start Program with the Nsight System's Profiler: ```./do_profile_metrics.sh << All Arguments >>```

### Notes:

##### Compile-Time Constants 
- There are some hardcoded constants at the top of source file: `bench_cublaslt_matmul.c`. These are currently assigned to H100 constants, you can change an re-make for other devices
- There are other constants within `cublas_helper.h` that specify whether to profile each individual matmul so you can see results within Nsys Timeline and also for the number of algorithms that cublasHeuristic search does. 

##### Profiling Script
- The profiling script is parameterized for H100 machine with `--gpu-metrics-set=gh100`. For A100 you can use `ga100`, for other Ampere (e.g. GeForce RTX 4090/3090) use `ga10x`. 
- If you are profiling long or many matmuls, you can reduce the sampling frequency to avoid dropping data with ```--gpu-metrics-frequency```. It is currently set to 10000 (e.g. 10kHz = 100 microseconds), it has maximum frequency of 200000.  