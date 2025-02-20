#ifndef CUBLAS_HELPER_H
#define CUBLAS_HELPER_H

#include "common.h"


#include <cublasLt.h>

typedef struct cublas_matmul_desc {
	// normally will only initialize
	// with N and K and let every stage
	// alter M/Adesc
	int M;
	int K;
	int N;
	cublasComputeType_t compute_type;
	cudaDataType scale_type;
	cublasLtMatmulDesc_t matmul_desc;
	cublasLtMatrixLayout_t Adesc;
	cublasLtMatrixLayout_t Bdesc;
	cublasLtMatrixLayout_t Cdesc;
	cublasLtMatrixLayout_t Ddesc;
	cublasLtMatmulPreference_t pref;
	cublasLtMatmulAlgo_t algo;
	int has_algo;
} Cublas_Matmul_Desc;


int initialize_cublas_handle(CUcontext ctx, cublasLtHandle_t * handle);

int do_cublas_matmul(CUstream compute_stream, cublasLtHandle_t handle, void * workspace, uint64_t workspace_bytes, int M, int K, int N, DataType dt, 
							float alpha, void * A, void * B, float beta, void * C, void * D, Cublas_Matmul_Desc * supplied_desc, Cublas_Matmul_Desc * save_desc, char * prof_label);


#endif