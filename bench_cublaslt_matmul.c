#include "common.h"

#include "dtype.h"
#include "backend_profile.h"
#include "cuda_drv.h"
#include "create_matrix.h"
#include "cublas_helper.h"

#define MAX_FP8_TFLOPS 1989.0f
#define MAX_FP16_TFLOPS 989.0f
#define MAX_FP32_TFLOPS 494.0f

#define MEM_BW_TBYTES_SEC 3.35f


int main (int argc, char * argv[]){

	int ret;

	if (argc != 10){
		fprintf(stderr, "Error: Usage. ./benchCublasLtMatmul <M> <K> <N> <dtype_fp_bits> <log2_workspace_bytes> <n_matmuls> <n_warmup> <use_same_b_matrix> <n_compute_iters>\n");
		return -1;
	}

	int M = atoi(argv[1]);
	int K = atoi(argv[2]);
	int N = atoi(argv[3]);

	int dtype_fp_bits = atoi(argv[4]);


	DataType dt;

	if (dtype_fp_bits == 8){
		dt = FP8;
	}
	else if (dtype_fp_bits == 16){
		dt = FP16;
	}
	else if (dtype_fp_bits == 32){
		dt = FP32;
	}
	else{
		fprintf(stderr, "Error: unsupported data type. Choose floating point # of bits as: 8, 16, 32...\n");
		return -1;
	}

	int log2_workspace_bytes = atoi(argv[5]);

	if (log2_workspace_bytes < 0){
		log2_workspace_bytes = 0;
	}
	if (log2_workspace_bytes >= 35){
		fprintf(stderr, "Error: setting workspace size too high. Choose value in range [0, 34] (=> [0, 16GB])...\n");
		return -1;
	}


	size_t workspace_bytes = 1 << log2_workspace_bytes;
	

	int n_matmuls = atoi(argv[6]);

	int n_warmup = atoi(argv[7]);

	int use_same_b_matrix = atoi(argv[8]);

	int n_compute_iters = atoi(argv[9]);


	ret = initialize_drv();
	if (ret){
		fprintf(stderr, "Error: failed to init cuda driver...\n");
		return -1;
	}


	// TODO: ENABLE MULTI THREADS FOR SUBMITTING WORK ON DIFFERENT DEVICES...
	// 			- Can start each thread indepenently from here...

	CUcontext ctx;


	// HARDCODING FOR NOW
	int device_id = 0;

	ret = initialize_ctx(device_id, &ctx);
	if (ret){
		fprintf(stderr, "Error: failed to init cuda ctx...\n");
		return -1;
	}

	const char * context_name = "Main Context";
	profile_name_context(&ctx, context_name);

	// HARDCODING FOR NOW
	int prio = 0;

	CUstream compute_stream;

	ret = initialize_stream(ctx, &compute_stream, prio);
	if (ret){
		fprintf(stderr, "Error: could not initialize stream\n");
		return -1;
	}

	const char * stream_name = "Compute Stream";

	profile_name_stream(&compute_stream, stream_name);


	cublasLtHandle_t cublas_handle;

	ret = initialize_cublas_handle(ctx, &cublas_handle);
	if (ret){
		fprintf(stderr, "Error: unable to initialize cublas handle...\n");
		return -1;
	}



	// Show Expected Outcome...
	double compute_peak_tflops;

	if (dt == FP8){
		compute_peak_tflops = MAX_FP8_TFLOPS;
	}
	else if (dt == FP16){
		compute_peak_tflops = MAX_FP16_TFLOPS;
	}
	else if (dt == FP32){
		compute_peak_tflops = MAX_FP32_TFLOPS;
	}
	else{
		fprintf(stderr, "Error: data type %d not supported...\n", dt);
		return -1;
	}

	double compute_bound_limit = compute_peak_tflops / MEM_BW_TBYTES_SEC;

	printf("Device Info:\n\tDevice Compute Peak TFLOPS (fp%d): %d\n\tMem BW (TB/sec): %.2f\n\tLower Arithmetic Intensity Bound for Full Utilization: %.2f\n\n\n", dtype_fp_bits, (int) compute_peak_tflops, MEM_BW_TBYTES_SEC, compute_bound_limit);

	uint64_t total_flops = 2 * (uint64_t) M * (uint64_t) K * (uint64_t) N;

	uint64_t total_bytes = sizeof_dtype(dt) * (((uint64_t) M * (uint64_t) K) + ((uint64_t) K * (uint64_t) N) + ((uint64_t) M * (uint64_t) N));

	double arithmetic_intensity = (double) total_flops /  (double) total_bytes;

	double frac_of_peak;

	if (arithmetic_intensity >= compute_bound_limit){
		frac_of_peak = 1;
	}
	else{
		frac_of_peak = arithmetic_intensity / compute_bound_limit;
	}

	
	double expected_tflops = compute_peak_tflops * frac_of_peak;

	double total_tflops = (double) total_flops / 1e12;

	double min_time_sec = total_tflops / expected_tflops;
	double min_time_micros = min_time_sec * 1e6;

	printf("M = %d, K = %d, N = %d\n\tArithmetic Intensity (Total FLOPs / Total Bytes): %.2f\n\tUpper Bound Throughput: %.2f%%\n\tUpper Bound TFLOPS: %.2f\n\tLower Bound Time (micros): %.2f\n\n\n", M, K, N, arithmetic_intensity, 100 * frac_of_peak, expected_tflops, min_time_micros);



	// MAKE MATRICES ON DEVICE!

	void ** h_A = malloc(n_matmuls * sizeof(void *));
	void ** h_B = malloc(n_matmuls * sizeof(void *));
	void ** h_C = malloc(n_matmuls * sizeof(void *));
	void ** h_D = malloc(n_matmuls * sizeof(void *));

	void ** d_A = malloc(n_matmuls * sizeof(void *));
	void ** d_B = malloc(n_matmuls * sizeof(void *));
	void ** d_C = malloc(n_matmuls * sizeof(void *));
	void ** d_D = malloc(n_matmuls * sizeof(void *));
	void ** d_workspace = malloc(n_matmuls * sizeof(void *));

	DataType c_dt = dt;

	if (dt == FP8){
		c_dt = FP16;
	}

	// DEFAULT INIT MATRICES USED IN DEEPSEEK
	float mean = 0;
	float std = 0.006;

	bool to_pin = false;

	// hardcoding...
	float alpha = 1.0;
	float beta = 0.0;

	uint64_t dt_size = sizeof_dtype(dt);
	uint64_t c_dt_size = sizeof_dtype(c_dt);

	uint64_t A_size = (uint64_t) M * (uint64_t) K * dt_size;
	uint64_t B_size = (uint64_t) K * (uint64_t) N * dt_size;
	uint64_t C_size = (uint64_t) M * (uint64_t) N * c_dt_size;
	uint64_t D_size = (uint64_t) M * (uint64_t) N * dt_size;

	uint64_t all_A_size = (uint64_t) n_matmuls * A_size;

	uint64_t all_B_size;
	if (use_same_b_matrix){
		all_B_size = B_size;
	}
	else{
		all_B_size = (uint64_t) n_matmuls * B_size;
	}


	uint64_t all_C_size = (uint64_t) n_matmuls * C_size;
	uint64_t all_D_size = (uint64_t) n_matmuls * D_size;


	printf("Allocating Space on Device for Matrices...\n\n");

	void * dev_all_A = alloc_dev_mem(ctx, all_A_size);
	if (!dev_all_A){
		fprintf(stderr, "Error: failed to alloc space for all A matrices on device\n\tSize: %lu\n", all_A_size);
		return -1;
	}

	void * dev_all_B = alloc_dev_mem(ctx, all_B_size);
	if (!dev_all_B){
		fprintf(stderr, "Error: failed to alloc space for all B matrices on device\n\tSize: %lu\n", all_B_size);
		return -1;
	}

	void * dev_all_C = alloc_dev_mem(ctx, all_C_size);
	if (!dev_all_C){
		fprintf(stderr, "Error: failed to alloc space for all C matrices on device\n\tSize: %lu\n", all_C_size);
		return -1;
	}

	void * dev_all_D = alloc_dev_mem(ctx, all_D_size);
	if (!dev_all_D){
		fprintf(stderr, "Error: failed to alloc space for all D matrices on device\n\tSize: %lu\n", all_D_size);
		return -1;
	}

	uint64_t all_workspace_size = (uint64_t) n_matmuls * workspace_bytes;
	void * dev_all_workspace = alloc_dev_mem(ctx, all_workspace_size);
	if (!dev_all_workspace){
		fprintf(stderr, "Error: failed to alloc space for all workspace on device\n\tSize: %lu\n", all_workspace_size);
		return -1;
	}

	// blocking call within mem alloc and mem free anyways,
	// so just gonna have long delays between each iteration...
	CUstream * inbound_stream_ref = &compute_stream;

	void * cur_A_loc = dev_all_A;
	void * cur_B_loc = dev_all_B;
	void * cur_C_loc = dev_all_C;
	void * cur_D_loc = dev_all_D;
	void * cur_workspace_loc = dev_all_workspace;

	printf("Creating Random Matrices and Copying to Device...\n\n");

	for (int i = 0; i < n_matmuls; i++){

		// REALLOC EACH ITERATION TO ENSURE FRESH CACHE...
		//printf("Creating Random Matrices and Sending To GPU Mem...\n");

		// internally does
		d_A[i] = create_rand_device_matrix(ctx, M, K, mean, std, dt, to_pin, inbound_stream_ref, &(h_A[i]), cur_A_loc);
		if (!d_A[i]){
			fprintf(stderr, "Error: failed to create A matrix on device\n\tM = %d, K = %d\n", M, K);
			return -1;
		}

		if ((i == 0) || (!use_same_b_matrix)){
			d_B[i] = create_rand_device_matrix(ctx, K, N, mean, std, dt, to_pin, inbound_stream_ref, &(h_B[i]), cur_B_loc);
			if (!d_B[i]){
				fprintf(stderr, "Error: failed to create B matrix on device...\n\tK = %d, N = %d\n", K, N);
				return -1;
			}
		}
		else{
			d_B[i] = d_B[0];
		}

		
		d_C[i] = create_rand_device_matrix(ctx, M, N, mean, 0, c_dt, to_pin, inbound_stream_ref, &(h_C[i]), cur_C_loc);
		if (!d_C[i]){
			fprintf(stderr, "Error: failed to create C matrix on device...\nM = %d, N = %d\n", M, N);
			return -1;
		}

		d_D[i] = create_zero_device_matrix(ctx, M, N, dt, to_pin, inbound_stream_ref, &(h_D[i]), cur_D_loc);
		if (!d_D[i]){
			fprintf(stderr, "Error: failed to create D matrix on device...\nM = %d, N = %d\n", M, N);
			return -1;
		}

		d_workspace[i] = cur_workspace_loc;
		if (!d_workspace[i]){
			fprintf(stderr, "Error: failed to alloc workspace bytes of size: %lu...\n", workspace_bytes);
			return -1;
		}

		cur_A_loc += A_size;
		
		cur_C_loc += C_size;
		cur_D_loc += D_size;
		cur_workspace_loc += workspace_bytes;

		if (!use_same_b_matrix){
			cur_B_loc += B_size;
		}
	}

	

	ret = stream_sync(ctx, compute_stream);
	if (ret){
		fprintf(stderr, "Error: unable to do stream_sync...\n");
		return -1;
	}

	for (int i = 0; i < n_matmuls; i++){
		free(h_A[i]);
		if ((i == 0) || (!use_same_b_matrix)){
			free(h_B[i]);
		}
		free(h_C[i]);
		free(h_D[i]);
	}

	ret = profile_start();
	if (ret){
		fprintf(stderr, "Error: unable to start profiler...\n");
		return -1;
	}


	Cublas_Matmul_Desc desc_info;

	char prof_warmup_str[256];

	sprintf(prof_warmup_str, "Warmup: %d Matmuls...", n_warmup);

	printf("Warmup: %d Matmuls...\n\n", n_warmup);

	profile_range_push(prof_warmup_str);

	int matmul_ind;
	for (int i = 0; i < n_warmup; i++){

		matmul_ind = i % n_matmuls;

		if (i == 0){
			ret = do_cublas_matmul(compute_stream, cublas_handle, d_workspace[matmul_ind], workspace_bytes, M, K, N, dt, 
							alpha, d_A[matmul_ind], d_B[matmul_ind], beta, d_C[matmul_ind], d_D[matmul_ind], NULL, &desc_info, NULL);
		}
		else{
			ret = do_cublas_matmul(compute_stream, cublas_handle, d_workspace[matmul_ind], workspace_bytes, M, K, N, dt, 
							alpha, d_A[matmul_ind], d_B[matmul_ind], beta, d_C[matmul_ind], d_D[matmul_ind], &desc_info, NULL, NULL);
		}
	}


	ret = stream_sync(ctx, compute_stream);
	if (ret){
		fprintf(stderr, "Error: unable to do stream_sync...\n");
		return -1;
	}

	profile_range_pop();




	struct timespec start_time, stop_time;
	uint64_t start_timestamp, stop_timestamp;
	
	char prof_outer_str[256];
	char prof_wrapper_str[256];
	char prof_core_str[256];

	double achieved_tflops;
	double achieved_throughput_pct;

	uint64_t elapsed_ns;
	double elapsed_micros;
	double elapsed_sec;
	
	// SUBMITTING ALL MATMULS.

	printf("Benchmark: %d Matmuls...\n\n", n_matmuls);

	sprintf(prof_outer_str, "Benchmark: %d Matmuls", n_matmuls);

	profile_range_push(prof_warmup_str);

	clock_gettime(CLOCK_MONOTONIC, &start_time);

	for (int iter = 0; iter < n_compute_iters; iter++){
		for (int i = 0; i < n_matmuls; i++){

			sprintf(prof_wrapper_str, "Main Function Iteration #: %d, Matmul #: %d", iter, i);
			sprintf(prof_core_str, "Core of Iteration #: %d, Matmul #: %d", iter, i);

			profile_range_push(prof_wrapper_str);

			// using the saved descriptor
			ret = do_cublas_matmul(compute_stream, cublas_handle, d_workspace[i], workspace_bytes, M, K, N, dt, 
								alpha, d_A[i], d_B[i], beta, d_C[i], d_D[i], &desc_info, NULL, prof_core_str);

			if (ret){
				fprintf(stderr, "Error: unable to do call cublas matmul for M = %d, K = %d, N = %d with dtype of fp%d...\n", M, K, N, dtype_fp_bits);
				return -1;
			}

			profile_range_pop();

		}
	}

	// set context to spin on sync so should be pretty accurate resolution...
	ret = stream_sync(ctx, compute_stream);
	if (ret){
		fprintf(stderr, "Error: unable to do stream_sync...\n");
		return -1;
	}

	profile_range_pop();

	clock_gettime(CLOCK_MONOTONIC, &stop_time);

	start_timestamp = start_time.tv_sec * 1e9 + start_time.tv_nsec;
	stop_timestamp = stop_time.tv_sec * 1e9 + stop_time.tv_nsec;

	elapsed_ns = stop_timestamp - start_timestamp;
	elapsed_micros = (double) elapsed_ns / 1e3;

	elapsed_sec = (double) elapsed_ns / 1e9;

	achieved_tflops = (((uint64_t) n_compute_iters * (uint64_t) n_matmuls * total_flops) / elapsed_sec) / 1e12;
	achieved_throughput_pct = achieved_tflops / compute_peak_tflops;

	printf("\nCompleted %d Matrix Multiplications!\n\tM = %d, K = %d, N = %d\n\n\t\tAvg. Elapsed Time (micros): %.2f\n\t\tOverall Achieved TFLOPS: %.3f\n\t\tAchieved Throughput: %.2f%%\n\n", n_matmuls, M, K, N, elapsed_micros / ((double) n_matmuls * (double) n_compute_iters), achieved_tflops, 100 * achieved_throughput_pct);

	ret = profile_stop();
	if (ret){
		fprintf(stderr, "Error: unable to stop profiler...\n");
		return -1;
	}

	// Cleaning up memory
	printf("\nFreeing Device Memory...\n");

	ret = free_dev_mem(ctx, dev_all_A);
	if (ret){
		fprintf(stderr, "Error: could not free all device A matrices...\n");
		return -1;
	}

	ret = free_dev_mem(ctx, dev_all_B);
	if (ret){
		fprintf(stderr, "Error: could not free all device B matrices...\n");
		return -1;
	}

	ret = free_dev_mem(ctx, dev_all_C);
	if (ret){
		fprintf(stderr, "Error: could not free all device C matrices...\n");
		return -1;
	}

	ret = free_dev_mem(ctx, dev_all_D);
	if (ret){
		fprintf(stderr, "Error: could not free all device D matrices...\n");
		return -1;
	}

	ret = free_dev_mem(ctx, dev_all_workspace);
	if (ret){
		fprintf(stderr, "Error: could not free all device workspace mem...\n");
		return -1;
	}

	printf("\nExiting...\n\n");

	return 0;
}