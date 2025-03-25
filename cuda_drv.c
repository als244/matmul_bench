#include "cuda_drv.h"

int initialize_drv(){

	CUresult result;
	const char * err;

	unsigned long flags = 0;
	result = cuInit(flags);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
    	fprintf(stderr, "Error: Could not init driver: %s\n", err);
    	return -1;
	}
	return 0;
}

int get_dev_attribute(int * ret_val, CUdevice dev, CUdevice_attribute attrib) {

	CUresult result;
	const char * err;

	result = cuDeviceGetAttribute(ret_val, attrib, dev);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: Could not get device attribute %d: %s\n", attrib, err);
		return -1;
	}

	return 0;
}


int initialize_ctx(int device_id, CUcontext * ctx, int num_sms, int * total_sms, int * used_sms){

	int ret;

	CUresult result;
	const char * err;

	CUdevice dev;
	result = cuDeviceGet(&dev, device_id);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
    	fprintf(stderr, "Error: Could not get device: %s\n", err);
    	return -1;
	}

	int major_arch_num;

	ret = get_dev_attribute(&major_arch_num, dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
	if (ret){
		fprintf(stderr, "Error: failed to get major arch num for cuda device...\n");
		return -1;
	}

	int min_count = 1;
	int multiple_factor = 1;
	int green_avail = true;
	// if num_sms > 0 then using green context
	// 6.X min count is 1
	// 7.X min count is 2 and must be multiple of 2
	// 8.X min count is 4 and must be multiple of 2
	// 9.X min count is 8 and must be multiple of 8
	switch(major_arch_num){
		case 6:
			min_count = 1;
			multiple_factor = 1;
			break;
		case 7:
			min_count = 2;
			multiple_factor = 2;
			break;
		case 8:
			min_count = 4;
			multiple_factor = 2;
		case 9:
			min_count = 8;
			multiple_factor = 8;
			break;
		default:
			min_count = 1;
			multiple_factor = 1;
			green_avail = false;
	}

	int sm_count;

	ret = get_dev_attribute(&sm_count, dev, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
	if (ret){
		fprintf(stderr, "Error: failed to get total sm count when setting device info...\n");
		return -1;
	}

	*total_sms = sm_count;


	// Set the host thread to spin waiting for completetion from GPU

	// could also use CU_CTX_SCHED_BLOCKING_SYNC to conserve CPU cycles/allow other threads...
	unsigned int ctx_flags = CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST;
	
	
	if ((num_sms <= 0) || (!green_avail)) {
		*used_sms = sm_count;

		result = cuCtxCreate(ctx, ctx_flags, dev);
		if (result != CUDA_SUCCESS){
			cuGetErrorString(result, &err);
	    	fprintf(stderr, "Error: Could not create context: %s\n", err);
	    	return -1;
		}
		return 0;
	}
	// create green context
	else{
		CUdevResource sm_resource;
		result = cuDeviceGetDevResource(dev, &sm_resource, CU_DEV_RESOURCE_TYPE_SM);
		if (result != CUDA_SUCCESS){
			cuGetErrorString(result, &err);
	    	fprintf(stderr, "Error: Could not get available sms: %s\n", err);
	    	return -1;
		}

		int cur_sm_cnt = sm_resource.sm.smCount;

		if (cur_sm_cnt != sm_count){
			fprintf(stderr, "Error: sm_resource smCount (%d) differs from dev attribute (%d)...\n", cur_sm_cnt, sm_count);
			return -1;
		}

		// No need for green context at this point
		if (num_sms >= cur_sm_cnt){
			*used_sms = sm_count;
			result = cuCtxCreate(ctx, ctx_flags, dev);
			if (result != CUDA_SUCCESS){
				cuGetErrorString(result, &err);
	    		fprintf(stderr, "Error: Could not create context: %s\n", err);
	    		return -1;
			}
			return 0;
		}

		// Abide by arch specs
		if (num_sms < min_count){
			sm_resource.sm.smCount = min_count;
		}
		else{
			int remain = num_sms % multiple_factor;
			sm_resource.sm.smCount = num_sms - remain;
		}

		printf("SM Count: %d\n", sm_resource.sm.smCount);

		*used_sms = sm_resource.sm.smCount;

		// Generate resource desc
		CUdevResourceDesc sm_resource_desc;
		unsigned int nbResources = 1;
		result = cuDevResourceGenerateDesc(&sm_resource_desc, &sm_resource, nbResources);
		if (result != CUDA_SUCCESS){
			cuGetErrorString(result, &err);
	    	fprintf(stderr, "Error: Could not generate resource desc: %s\n", err);
	    	return -1;
		}

		// Create green context
		CUgreenCtx green_ctx;
		result = cuGreenCtxCreate(&green_ctx, sm_resource_desc, dev, ctx_flags);
		if (result != CUDA_SUCCESS){
			cuGetErrorString(result, &err);
	    	fprintf(stderr, "Error: Could not create green ctx: %s\n", err);
	    	return -1;
		}

		// Convert from green context to primary
		result = cuCtxFromGreenCtx(ctx, green_ctx);
		if (result != CUDA_SUCCESS){
			cuGetErrorString(result, &err);
	    	fprintf(stderr, "Error: Could not convert from green ctx to primary: %s\n", err);
	    	return -1;
		}
	}


	return 0;
}


int initialize_stream(CUcontext ctx, CUstream * stream, int prio){

	CUresult result;
	const char * err;

	result = cuCtxPushCurrent(ctx);
	if (result != CUDA_SUCCESS){
		fprintf(stderr, "Error: could not set context...\n");
		return -1;
	}

	result = cuStreamCreateWithPriority(stream, CU_STREAM_NON_BLOCKING, prio);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: unable to create cuda stream\n");
		return -1;
	}

	cuCtxPopCurrent(NULL);

	return 0;
}

int stream_sync(CUcontext ctx, CUstream stream){

	CUresult result;
	const char * err;

	result = cuCtxPushCurrent(ctx);
	if (result != CUDA_SUCCESS){
		fprintf(stderr, "Error: could not set context...\n");
		return -1;
	}

	result = cuStreamSynchronize(stream);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: unable to do cuStreamSynchronize...\n");
		return -1;
	}

	return 0;


}

void * alloc_dev_mem(CUcontext ctx, uint64_t size_bytes) {

	CUresult result;
	const char * err;

	result = cuCtxPushCurrent(ctx);
	if (result != CUDA_SUCCESS){
		fprintf(stderr, "Error: could not set context...\n");
		return NULL;
	}


	void * dev_mem;

	result = cuMemAlloc((CUdeviceptr *) &dev_mem, size_bytes);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: Could not do cudaMemAlloc with total size %lu: %s\n", size_bytes, err);
		return NULL;
	}

	cuCtxPopCurrent(NULL);
	
	return dev_mem;
}

int free_dev_mem(CUcontext ctx, void * dev_ptr){

	CUresult result;
	const char * err;

	result = cuCtxPushCurrent(ctx);
	if (result != CUDA_SUCCESS){
		fprintf(stderr, "Error: could not set context...\n");
		return -1;
	}


	result = cuMemFree((CUdeviceptr) dev_ptr);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: Could not do cuMemFree...: %s\n", err);
		return -1;
	}

	cuCtxPopCurrent(NULL);
	
	return 0;

}

int transfer_host_to_device(CUcontext ctx, void * dev_dest, void * host_src, uint64_t size_bytes, CUstream * stream_ref){

	CUresult result;
	const char * err;

	result = cuCtxPushCurrent(ctx);
	if (result != CUDA_SUCCESS){
		fprintf(stderr, "Error: could not set context...\n");
		return -1;
	}

	if (!stream_ref){
		result = cuMemcpyHtoD((CUdeviceptr) dev_dest, host_src, size_bytes);
	}
	else{
		CUstream inbound_stream = *stream_ref;
		result = cuMemcpyHtoDAsync((CUdeviceptr) dev_dest, host_src, size_bytes, inbound_stream);
	}

	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: Could not do cudaMemcpyHtoD with total size %lu: %s\n", size_bytes, err);
		return -1;
	}

	cuCtxPopCurrent(NULL);

	return 0;
}

int transfer_device_to_host(CUcontext ctx, void * host_dest, void * dev_src, uint64_t size_bytes, CUstream * stream_ref){

	CUresult result;
	const char * err;

	result = cuCtxPushCurrent(ctx);
	if (result != CUDA_SUCCESS){
		fprintf(stderr, "Error: could not set context...\n");
		return -1;
	}

	if (!stream_ref){
		result = cuMemcpyDtoH(host_dest, (CUdeviceptr) dev_src, size_bytes);
	}
	else{
		CUstream outbound_stream = *stream_ref;
		result = cuMemcpyDtoHAsync(host_dest, (CUdeviceptr) dev_src, size_bytes, outbound_stream);
	}

	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: Could not do cudaMemcpyDtoH with total size %lu: %s\n", size_bytes, err);
		return -1;
	}

	cuCtxPopCurrent(NULL);

	return 0;
}

int transfer_device_to_device(CUcontext ctx, void * dev_dest, void * dev_src, uint64_t size_bytes, CUstream * stream_ref){

	CUresult result;
	const char * err;

	result = cuCtxPushCurrent(ctx);
	if (result != CUDA_SUCCESS){
		fprintf(stderr, "Error: could not set context...\n");
		return -1;
	}

	if (!stream_ref){
		result = cuMemcpyDtoD((CUdeviceptr) dev_dest, (CUdeviceptr) dev_src, size_bytes);
	}
	else{
		CUstream outbound_stream = *stream_ref;
		result = cuMemcpyDtoDAsync((CUdeviceptr) dev_dest, (CUdeviceptr) dev_src, size_bytes, outbound_stream);
	}

	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: Could not do cudaMemcpyDtoD with total size %lu: %s\n", size_bytes, err);
		return -1;
	}

	cuCtxPopCurrent(NULL);

	return 0;
}


int pin_host_mem_with_cuda(void * host_ptr, uint64_t size_bytes) {

	CUresult result;
	const char * err;

	// setting CU_MEMHOSTREGISTER_PORTABLE makes this memory pinned from the viewpoint of all cuda contexts
	// we already initialized this memory with MAP_POPULATE and called mlock() on it so we know it is truly pinned
	// but the cuda driver also needs to do it's page locking (under the hood it calls MAP_FIXED)
	result = cuMemHostRegister(host_ptr, size_bytes, CU_MEMHOSTREGISTER_PORTABLE);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
		fprintf(stderr, "Error: unable to regsiter system memory buffer of size %lu with cuda. Err: %s\n", size_bytes, err);
		return -1;
	}

	return 0;

}

