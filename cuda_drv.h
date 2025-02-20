#ifndef CUDA_DRV_H
#define CUDA_DRV_H

#include "common.h"

int initialize_drv();

int initialize_ctx(int device_id, CUcontext * ctx);

int initialize_stream(CUcontext ctx, CUstream * stream, int prio);

int stream_sync(CUcontext ctx, CUstream stream);

void * alloc_dev_mem(CUcontext ctx, uint64_t size_bytes);
int free_dev_mem(CUcontext ctx, void * dev_ptr);

int transfer_host_to_device(CUcontext ctx, void * dev_dest, void * host_src, uint64_t size_bytes, CUstream * stream_ref);
int transfer_device_to_host(CUcontext ctx, void * host_dest, void * dev_src, uint64_t size_bytes, CUstream * stream_ref);
int transfer_device_to_device(CUcontext ctx, void * dev_dest, void * dev_src, uint64_t size_bytes, CUstream * stream_ref);


int pin_host_mem_with_cuda(void * host_ptr, uint64_t size_bytes);



#endif