#ifndef CREATE_MATRIX_H
#define CREATE_MATRIX_H

#include "common.h"

#include "cuda_drv.h"

void * create_zero_host_matrix(uint64_t M, uint64_t N, DataType dt, bool to_pin);
void * create_rand_host_matrix(uint64_t M, uint64_t N, float mean, float std, DataType dt, bool to_pin);
void * load_host_matrix_from_file(char * filepath, uint64_t M, uint64_t N, DataType orig_dt, DataType dt, bool to_pin);

void * create_zero_device_matrix(CUcontext ctx, uint64_t M, uint64_t N, DataType dt, bool to_pin, CUstream * stream_ref, void ** ret_host_matrix, void * dev_ptr);
void * create_rand_device_matrix(CUcontext ctx, uint64_t M, uint64_t N, float mean, float std, DataType dt, bool to_pin, CUstream * stream_ref, void ** ret_host_matrix, void * dev_ptr);
void * load_device_matrix_from_file(CUcontext ctx, char * filepath, uint64_t M, uint64_t N, DataType orig_dt, DataType dt, bool to_pin, CUstream * stream_ref, void ** ret_host_matrix, void * dev_ptr);




#endif