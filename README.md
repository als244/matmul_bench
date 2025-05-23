# Matmul Benchmarking with cublasLt

## Directions
1. Download Source
2. Run `make`
3. Start Program from Command Line:
	- ```./benchCublasLtMatmul <M> <K> <N> <dtype_fp_bits> <n_sms> <use_fp16_accum> <log2_workspace_bytes> <n_matmuls> <n_warmup> <n_compute_iters> <use_same_b_matrix> <use_c_matrix>```
	- Instead, can start program with shell script invoking `nsys` profiler: 
		- Ensure that there is a directory `profiling` in current directory for `.nsys-rep` output to be saved.
			- Output files can then be loaded in Nsight System's GUI to see timeline view.
		- Ensure to toggle compile-time constant within `cublas_helper.h` and re-make before doing profiling runs. 
		- Command: ```./do_profile_metrics.sh << All Arguments >>```

### Notes:

#### Command Line Arguments
- `M, K, N`: Matrix Dimensions
- `dtype_fp_bits`: One of 8, 16, or 32. 
	- Sets dtypes for A, B, D. If use_c_matrix is set and dtype_fp_bits is 8, then C dtype is 16-bit, otherwise C dtype is also set to same as others.
- `n_sms`: The number of SMs to use. 
	- Setting to 0 implies using all SMs. Setting > 0 and below min count will set to min count number of SMs. If > min_count then number of SMs will round-down to nearest arch-specific multiple. See: [Nvidia Green Context Documentation](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html) to understand
the min counts and multiples for different arch versions. 
- `use_fp16_accum`: Choose either 0 or 1 to enable FP16 accumulation. 
	- On RTX devices FP16 has double the throughput of FP32 accumulation, while on A100/H100 FP32 has same perf as FP16.
- `log2_workspace_bytes`: Setting the workspace size as 2^this value. 
	- A reasonable range is [0, 26], with 22 as recommendation.
- `n_matmuls`: Number of distinct matrix mutliplications (unique A matrices and possibly unique B) to perform. 
	- Done in round-robin fashion to help simulate true performance with realistic caching behavior. 
- `n_warmup`: Number of total matrix matmuls to perform before starting profiling/timing. 
	- This helps simulate true performance in sense that after warmup phase clock-speed will represent realistic behavior.
- `n_compute_iters`: Number of iterations to repeat the round-robin dispatching of the `n_matmuls`
- `use_same_b_matrix`: Choose either 0 or 1 to set the same B matrix (i.e. same address) for each of the `n_matmuls`
- `use_c_matrix`: Choose either 0 or 1 to add C (different than D) to the result of AB^T before saving in D.

#### Compile-Time Constants 
- There are some hardcoded constants at the top of source file: `bench_cublaslt_matmul.c`. 
	- These are currently assigned to H100 constants, you can change an re-make for other devices
- There are constants within `cublas_helper.h` that specify whether to profile each individual matmul so you can see results within Nsys Timeline and also for the number of algorithms that cublasHeuristic search does. 

#### Profiling Script
- If you are profiling long or many matmuls, you can reduce the sampling frequency to avoid dropping data with `--gpu-metrics-frequency`. 
	- It is currently set to 20000 (e.g. 20kHz = 50 microsecond sampling), it has maximum frequency of 200000. 
- The profiling script is curparameterized for H100 machine with `--gpu-metrics-set=gh100`. 
	- For A100 you can use `ga100`, for other Ampere (e.g. GeForce RTX 4090/3090) use `ga10x`. 
