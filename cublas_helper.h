#ifndef CUBLAS_HELPER_H
#define CUBLAS_HELPER_H

#include "common.h"


#include <cublasLt.h>

#define TO_PROF_MATMUL 0

#define MAX_ALGO_SEARCH 5

typedef struct cublas_matmul_params {
	cublasLtMatmulDesc_t computeDesc;
	uint16_t alpha_h;
	float alpha_f;
	double alpha_d;
	void * alpha;
	void * A;
	cublasLtMatrixLayout_t Adesc;
	void * B;
	cublasLtMatrixLayout_t Bdesc;
	uint16_t beta_h;
	float beta_f;
	double beta_d;
	void * beta;
	void * C;
	cublasLtMatrixLayout_t Cdesc;
	void * D;
	bool same_cDesc;
	cublasLtMatrixLayout_t Ddesc;
	cublasLtMatmulPreference_t pref;
	cublasLtMatmulHeuristicResult_t heuristicResultsArray[MAX_ALGO_SEARCH];
	void * workspace;
	size_t workspaceBytes;
} Cublas_Matmul_Params;


int initialize_cublas_handle(CUcontext ctx, cublasLtHandle_t * cublas_handle);

int do_cublas_matmul(CUstream compute_stream, cublasLtHandle_t cublas_handle,
						DataType a_dt, DataType b_dt, DataType c_dt, DataType d_dt,
						DataType compute_dt,
						int M, int K, int N,
						float alpha, float beta,
						uint64_t workspaceBytes, void * workspace,
						void * A, void * B, void * C, void * D,
						int num_sms,
						char * prof_label);


#endif