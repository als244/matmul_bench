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


int initialize_ctx(int device_id, CUcontext * ctx){

	CUresult result;
	const char * err;

	CUdevice dev;
	result = cuDeviceGet(&dev, device_id);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
    	fprintf(stderr, "Error: Could not get device: %s\n", err);
    	return -1;
	}


	// Set the host thread to spin waiting for completetion from GPU

	// could also use CU_CTX_SCHED_BLOCKING_SYNC to conserve CPU cycles/allow other threads...
	unsigned int ctx_flags = CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST;
	
	result = cuCtxCreate(ctx, ctx_flags, dev);
	if (result != CUDA_SUCCESS){
		cuGetErrorString(result, &err);
    	fprintf(stderr, "Error: Could not create context: %s\n", err);
    	return -1;
	}

	result = cuCtxPushCurrent(*ctx);
	if (result != CUDA_SUCCESS){
		fprintf(stderr, "Error: could not set context...\n");
		return -1;
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

