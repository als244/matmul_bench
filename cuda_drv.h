#ifndef CUDA_DRV_H
#define CUDA_DRV_H

#include "common.h"

int initialize_drv();

int get_dev_attribute(int * ret_val, CUdevice dev, CUdevice_attribute attrib);

// if num_sms > 0 then using green context
// 6.X min count is 1
// 7.X min count is 2 and must be multiple of 2
// 8.X min count is 4 and must be multiple of 2
// 9.X min count is 8 and must be multiple of 8
int initialize_ctx(int device_id, CUcontext * ctx, int num_sms);

int initialize_stream(CUcontext ctx, CUstream * stream, int prio);

int stream_sync(CUcontext ctx, CUstream stream);

void * alloc_dev_mem(CUcontext ctx, uint64_t size_bytes);
int free_dev_mem(CUcontext ctx, void * dev_ptr);

int transfer_host_to_device(CUcontext ctx, void * dev_dest, void * host_src, uint64_t size_bytes, CUstream * stream_ref);
int transfer_device_to_host(CUcontext ctx, void * host_dest, void * dev_src, uint64_t size_bytes, CUstream * stream_ref);
int transfer_device_to_device(CUcontext ctx, void * dev_dest, void * dev_src, uint64_t size_bytes, CUstream * stream_ref);


int pin_host_mem_with_cuda(void * host_ptr, uint64_t size_bytes);



#endif