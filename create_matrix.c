#include "create_matrix.h"


float rand_normal(float mean, float std) {

	if ((mean == 0) && (std == 0)){
		return 0;
	}

    static float spare;
    static int has_spare = 0;

    if (has_spare) {
        has_spare = 0;
        return mean + std * spare;
    } else {
        float u, v, s;
        do {
            u = (rand() / (float)RAND_MAX) * 2.0 - 1.0;
            v = (rand() / (float)RAND_MAX) * 2.0 - 1.0;
            s = u * u + v * v;
        } while (s >= 1.0 || s == 0.0);

        s = sqrtf(-2.0 * logf(s) / s);
        spare = v * s;
        has_spare = 1;
        return mean + std * u * s;
    }
}


void * create_zero_host_matrix(uint64_t M, uint64_t N, DataType dt, bool to_pin){

	int ret;
	
	uint64_t num_els = M * N;
	uint64_t dtype_size = sizeof_dtype(dt);

	void * zero_matrix = calloc(num_els, dtype_size);
	if (!zero_matrix){
		fprintf(stderr, "Error: could not allocate zero matrix on host of size %lu...\n", num_els * dtype_size);
		return NULL;
	}

	if (to_pin){

		ret = pin_host_mem_with_cuda(zero_matrix, num_els * dtype_size);
		if (ret){
			fprintf(stderr, "Error: unable to pin zero host matrix with cuda of size: %lu...\n", num_els * dtype_size);
			return NULL;
		}

	}

	return zero_matrix;
}

void * create_rand_host_matrix(uint64_t M, uint64_t N, float mean, float std, DataType dt, bool to_pin){

	size_t dtype_size = sizeof_dtype(dt);

	uint64_t num_els = M * N;

	void * rand_matrix = malloc(num_els * dtype_size);

	if (!rand_matrix){
		fprintf(stderr, "Error: could not allocate random float matrix\n\tM: %lu\n\tN: %lu\n\n", M, N);
		return NULL;
	}

	float rand_val;

	void * cur_matrix_el = rand_matrix;
	
	for (uint64_t i = 0; i < num_els; i++){
		rand_val = rand_normal(mean, std);

		if (dt == FP8E4M3){
			*((uint8_t *) cur_matrix_el) = fp32_to_fp8(rand_val, 4, 3);
		}
		else if (dt == FP16){
			*((uint16_t *) cur_matrix_el) = fp32_to_fp16(rand_val);
		}
		else if (dt == FP32){
			*((float *) cur_matrix_el) = rand_val;
		}
		else{
			fprintf(stderr, "Error: dtype of type: %d, unsupported\n", dt);
			return NULL;
		}

		cur_matrix_el += dtype_size;
	}

	

	int ret;

	if (to_pin){

		ret = pin_host_mem_with_cuda(rand_matrix, num_els * dtype_size);
		if (ret){
			fprintf(stderr, "Error: unable to pin random host matrix with cuda of size: %lu...\n", num_els * dtype_size);
			return NULL;
		}

	}

	return rand_matrix;
}

void * load_host_matrix_from_file(char * filepath, uint64_t M, uint64_t N, DataType orig_dt, DataType dt, bool to_pin){

	FILE * fp = fopen(filepath, "rb");
	if (!fp){
		fprintf(stderr, "Error: could not load matrix from file: %s\n", filepath);
		return NULL;
	}

	uint64_t num_els = M * N;
	uint64_t orig_dtype_size = sizeof_dtype(orig_dt);
	void * orig_matrix = malloc(num_els * orig_dtype_size);
	if (!orig_matrix){
		fprintf(stderr, "Error: malloc failed to allocate memory for orig matrix (M = %lu, N = %lu)\n", M, N);
		return NULL;
	}

	size_t nread = fread(orig_matrix, orig_dtype_size, num_els, fp);
	if (nread != num_els){
		fprintf(stderr, "Error: couldn't read expected number of elements from matrix file: %s. (Expected %lu, read: %lu)\n", filepath, num_els, nread);
		free(orig_matrix);
		return NULL;
	}

	fclose(fp);

	if (orig_dt == dt){
		return orig_matrix;
	}

	
	uint64_t new_dtype_size = sizeof_dtype(dt);

	void * new_matrix = malloc(num_els * new_dtype_size);
	if (!new_matrix){
		fprintf(stderr, "Error: malloc failed to allocate memory for new matrix (M = %lu, N = %lu)\n", M, N);
		free(orig_matrix);
		return NULL;
	}

	void * cur_orig_matrix_el = orig_matrix;
	void * cur_new_matrix_el = new_matrix;

	float orig_val_fp32;
	uint16_t orig_val_fp16;
	uint8_t orig_val_fp8;

	float fp32_upcast;
	
	for (uint64_t i = 0; i < num_els; i++){

		if (orig_dt == FP32){

			orig_val_fp32 = *((float *) cur_orig_matrix_el);

			if (dt == FP16){
				*((uint16_t *) cur_new_matrix_el) = fp32_to_fp16(orig_val_fp32);
			}
			else if (dt == FP8E4M3){
				*((uint8_t *) cur_new_matrix_el) = fp32_to_fp8(orig_val_fp32, 4, 3);
			}
			else{
				fprintf(stderr, "Error: dt conversion not supported (from %d to %d)...\n", orig_dt, dt);
				free(orig_matrix);
				free(new_matrix);
				return NULL;
			}
		}
		else if (orig_dt == FP16){

			orig_val_fp16 = *((uint16_t *) cur_orig_matrix_el);

			fp32_upcast = fp16_to_fp32(orig_val_fp16);

			if (dt == FP32){
				*((float *) cur_new_matrix_el) = fp32_upcast;
			}
			else if (dt == FP8E4M3){
				*((uint8_t *) cur_new_matrix_el) = fp32_to_fp8(fp32_upcast, 4, 3);
			}
			else{
				fprintf(stderr, "Error: dt conversion not supported (from %d to %d)...\n", orig_dt, dt);
				free(orig_matrix);
				free(new_matrix);
				return NULL;
			}

		}
		else{
			fprintf(stderr, "Error: dt conversion not supported (from %d to %d)...\n", orig_dt, dt);
			free(orig_matrix);
			free(new_matrix);
			return NULL;
		}


		cur_orig_matrix_el += orig_dtype_size;
		cur_new_matrix_el += new_dtype_size;

	}

	free(orig_matrix);

	int ret;

	if (to_pin){

		ret = pin_host_mem_with_cuda(new_matrix, num_els * new_dtype_size);
		if (ret){
			fprintf(stderr, "Error: unable to pin loaded host matrix with cuda of size: %lu...\n", num_els * new_dtype_size);
			return NULL;
		}

	}

	return new_matrix;
}



void * create_zero_device_matrix(CUcontext ctx, uint64_t M, uint64_t N, DataType dt, bool to_pin, CUstream * stream_ref, void ** ret_host_matrix, void * dev_ptr) {

	int ret;

	*ret_host_matrix = NULL;

	// 1.) Allocate and Populate Host Matrix
	void * host_matrix = create_zero_host_matrix(M, N, dt, to_pin);
	if (!host_matrix){
		fprintf(stderr, "Error: unable to create zero host matrix with M= %lu, N=%lu...\n", M, N);
		return NULL;
	}

	// 2.) Potentially Allocate Device Matrix

	void * dev_matrix;

	uint64_t dtype_size = sizeof_dtype(dt);
	uint64_t num_els = M * N;
	uint64_t matrix_size_bytes = dtype_size * num_els;

	if (!dev_ptr){
		
		dev_matrix = alloc_dev_mem(ctx, matrix_size_bytes);
		if (!dev_matrix){
			fprintf(stderr, "Error: unable to allocate device matrix of size: %lu...\n", matrix_size_bytes);
			free(host_matrix);
			return NULL;
		}
	}
	else{
		dev_matrix = dev_ptr;
	}



	// 3.) Copy to Device
	ret = transfer_host_to_device(ctx, dev_matrix, host_matrix, matrix_size_bytes, stream_ref);
	if (ret){
		fprintf(stderr, "Error: unable to transfer matrix of size: %lu from host to device...\n", matrix_size_bytes);
		free(host_matrix);
		return NULL;
	}

	*ret_host_matrix = host_matrix;

	return dev_matrix;

}


void * create_rand_device_matrix(CUcontext ctx, uint64_t M, uint64_t N, float mean, float std, DataType dt, bool to_pin, CUstream * stream_ref, void ** ret_host_matrix, void * dev_ptr) {

	int ret;

	*ret_host_matrix = NULL;

	// 1.) Allocate and Populate Host Matrix
	void * host_matrix = create_rand_host_matrix(M, N, mean, std, dt, to_pin);
	if (!host_matrix){
		fprintf(stderr, "Error: unable to create random host matrix with M= %lu, N=%lu...\n", M, N);
		return NULL;
	}

	// 2.) Potentially Allocate Device Matrix

	void * dev_matrix;

	uint64_t dtype_size = sizeof_dtype(dt);
	uint64_t num_els = M * N;
	uint64_t matrix_size_bytes = dtype_size * num_els;

	if (!dev_ptr){
		
		dev_matrix = alloc_dev_mem(ctx, matrix_size_bytes);
		if (!dev_matrix){
			fprintf(stderr, "Error: unable to allocate device matrix of size: %lu...\n", matrix_size_bytes);
			free(host_matrix);
			return NULL;
		}
	}
	else{
		dev_matrix = dev_ptr;
	}



	// 3.) Copy to Device
	ret = transfer_host_to_device(ctx, dev_matrix, host_matrix, matrix_size_bytes, stream_ref);
	if (ret){
		fprintf(stderr, "Error: unable to transfer matrix of size: %lu from host to device...\n", matrix_size_bytes);
		free(host_matrix);
		return NULL;
	}

	*ret_host_matrix = host_matrix;

	return dev_matrix;
}

void * load_device_matrix_from_file(CUcontext ctx, char * filepath, uint64_t M, uint64_t N, DataType orig_dt, DataType dt, bool to_pin, CUstream * stream_ref, void ** ret_host_matrix, void * dev_ptr){

	int ret;

	*ret_host_matrix = NULL;

	// 1.) Allocate and Populate Host Matrix
	void * host_matrix = load_host_matrix_from_file(filepath, M, N, orig_dt, dt, to_pin);
	if (!host_matrix){
		fprintf(stderr, "Error: unable to create random host matrix with M= %lu, N=%lu...\n", M, N);
		return NULL;
	}

	// 2.) Potentially Allocate Device Matrix

	uint64_t dtype_size = sizeof_dtype(dt);
	uint64_t num_els = M * N;
	uint64_t matrix_size_bytes = dtype_size * num_els;

	void * dev_matrix;

	if (!dev_ptr){
		dev_matrix = alloc_dev_mem(ctx, matrix_size_bytes);
		if (!dev_matrix){
			fprintf(stderr, "Error: unable to allocate device matrix of size: %lu...\n", matrix_size_bytes);
			free(host_matrix);
			return NULL;
		}
	}
	else{
		dev_matrix = dev_ptr;
	}


	// 3.) Copy to Device
	ret = transfer_host_to_device(ctx, dev_matrix, host_matrix, matrix_size_bytes, stream_ref);
	if (ret){
		fprintf(stderr, "Error: unable to transfer matrix of size: %lu from host to device...\n", matrix_size_bytes);
		free(host_matrix);
		return NULL;
	}

	*ret_host_matrix = host_matrix;

	return dev_matrix;

}
