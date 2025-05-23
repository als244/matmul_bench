#include "cublas_helper.h"



int initialize_cublas_handle(CUcontext ctx, cublasLtHandle_t * cublas_handle){

	CUresult result;
	const char * err;


	result = cuCtxPushCurrent(ctx);
	if (result != CUDA_SUCCESS){
		fprintf(stderr, "Error: could not set context...\n");
		return -1;
	}

	cublasStatus_t status;
	status = cublasLtCreate(cublas_handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "Error: could not create cublasLt handle\n");
		return -1;
	}

	cuCtxPopCurrent(NULL);

	return 0;
}

static int dtype_to_cuda_dtype(DataType dtype, cudaDataType * ret_dtype){

	switch(dtype){
		case FP8E4M3:
			*ret_dtype = CUDA_R_8F_E4M3;
			break;
		case FP8E5M2:
			*ret_dtype = CUDA_R_8F_E5M2;
			break;
		case FP16:
			*ret_dtype = CUDA_R_16F;
			break;
		case BF16:
			*ret_dtype = CUDA_R_16BF;
			break;
		case FP32:
			*ret_dtype = CUDA_R_32F;
			break;
		default:
			printf("Error: unsupported dtype to convert to cuda\n");
			return -1;
	}

	return 0;
}

static int set_cublas_compute_scale_types(cublasComputeType_t * cublas_compute_type, cudaDataType * scale_cuda_dt, DataType compute_dt, cudaDataType a_cuda_dt, cudaDataType b_cuda_dt, cudaDataType c_cuda_dt, cudaDataType d_cuda_dt){

	switch (compute_dt){
		case FP32:
			// ERROR CHECKING
			switch (a_cuda_dt){
				case CUDA_R_32F:
					if (b_cuda_dt != CUDA_R_32F){
						fprintf(stderr, "Error: when A dt = FP32, B must also be FP32...\n");
						return -1;
					}
					if (c_cuda_dt != CUDA_R_32F){
						fprintf(stderr, "Error: when A dt = FP32, C must also be FP32...\n");
						return -1;
					}
					break;
				case CUDA_R_16F:
					if (b_cuda_dt != CUDA_R_16F){
						fprintf(stderr, "Error: when A dt = FP16, B must also be FP16...\n");
						return -1;
					}
					if ((c_cuda_dt != CUDA_R_32F) && (c_cuda_dt != CUDA_R_16F)){
						fprintf(stderr, "Error: when A dt = FP16, C must either be FP16 or FP32...\n");
						return -1;
					}
					break;
				case CUDA_R_16BF:
					if (b_cuda_dt != CUDA_R_16BF){
						fprintf(stderr, "Error: when A dt = BF16, B must also be BF16...\n");
						return -1;
					}
					if ((c_cuda_dt != CUDA_R_32F) && (c_cuda_dt != CUDA_R_16BF)){
						fprintf(stderr, "Error: when A dt = BF16, C must either be BF16 or FP32...\n");
						return -1;
					}
					break;
				case CUDA_R_8F_E4M3:
					switch (b_cuda_dt){
						case CUDA_R_8F_E4M3:
							switch (c_cuda_dt){
								case CUDA_R_16F:
									if ((d_cuda_dt != CUDA_R_16F) && (d_cuda_dt != CUDA_R_8F_E4M3)){
										fprintf(stderr, "Error: when A dt = B dt FP8E4M3 and C = FP16, D must be either FP16 or FP8E4M3...\n");
										return -1;
									}
									break;
								case CUDA_R_16BF:
									if ((d_cuda_dt != CUDA_R_16BF) && (d_cuda_dt != CUDA_R_8F_E4M3)){
										fprintf(stderr, "Error: when A dt = B dt FP8E4M3 and C = BF16, D must be either BF16 or FP8E4M3...\n");
										return -1;
									}
									break;
								case CUDA_R_32F:
									if (d_cuda_dt != CUDA_R_32F){
										fprintf(stderr, "Error: when A dt = B dt FP8E4M3 and C = FP32, D must be FP32...\n");
										return -1;
									}
									break;
								default:
									fprintf(stderr, "Error: when A dt = B dt FP8E4M3, C dt must be FP16, BF16, or FP32...\n");
									return -1;
							}
							break;
						case CUDA_R_8F_E5M2:
							switch (c_cuda_dt){
								case CUDA_R_16F:
									if ((d_cuda_dt != CUDA_R_16F) && (d_cuda_dt != CUDA_R_8F_E4M3) && (d_cuda_dt != CUDA_R_8F_E5M2)){
										fprintf(stderr, "Error: when A dt = FP8E4M3, B dt = FP8E5M2 and C = FP16, D must be either FP16, FP8E4M3, or FP8E5M2...\n");
										return -1;
									}
									break;
								case CUDA_R_16BF:
									if ((d_cuda_dt != CUDA_R_16BF) && (d_cuda_dt != CUDA_R_8F_E4M3) && (d_cuda_dt != CUDA_R_8F_E5M2)){
										fprintf(stderr, "Error: when A dt = FP8E4M3, B dt = FP8E5M2 and C = BF16, D must be either BF16 or FP8E4M3, or FP8E5M2...\n");
										return -1;
									}
									break;
								case CUDA_R_32F:
									if (d_cuda_dt != CUDA_R_32F){
										fprintf(stderr, "Error: when A dt = FP8E4M3, B dt = FP8E5M2 and C = FP32, D must be FP32...\n");
										return -1;
									}
									break;
								default:
									fprintf(stderr, "Error: when when A dt = FP8E4M3, B dt = FP8E5M2, C dt must be FP16, BF16, or FP32...\n");
									return -1;
							}
							break;
						default:
							fprintf(stderr, "Error: when A dt = FP8E4M3, B dt must be FP8E4M3 or FP8E5M2...\n");
							return -1;
					}
					break;
				case CUDA_R_8F_E5M2:
					if (b_cuda_dt != CUDA_R_8F_E4M3){
						fprintf(stderr, "Error: when A = FP8E5M2, B dt must be FP8E4M3...\n");
						return -1;
					}
					switch (c_cuda_dt){
						case CUDA_R_16F:
							if ((d_cuda_dt != CUDA_R_16F) && (d_cuda_dt != CUDA_R_8F_E4M3) && (d_cuda_dt != CUDA_R_8F_E5M2)){
								fprintf(stderr, "Error: when A dt = FP8E5M2, B dt = FP8E4M3 and C = FP16, D must be either FP16, FP8E4M3, or FP8E5M2...\n");
								return -1;
							}
							break;
						case CUDA_R_16BF:
							if ((d_cuda_dt != CUDA_R_16BF) && (d_cuda_dt != CUDA_R_8F_E4M3) && (d_cuda_dt != CUDA_R_8F_E5M2)){
								fprintf(stderr, "Error: when A dt = FP8E5M2, B dt = FP8E4M3 and C = BF16, D must be either BF16 or FP8E4M3, or FP8E5M2...\n");
								return -1;
							}
							break;
						case CUDA_R_32F:
							if (d_cuda_dt != CUDA_R_32F){
								fprintf(stderr, "Error: when A dt = FP8E5M2, B dt = FP8E4M3 and C = FP32, D must be FP32...\n");
								return -1;
							}
							break;
						default:
							fprintf(stderr, "Error: when A dt = FP8E5M2, B dt = FP8E4M3, C dt must be FP16, BF16, or FP32...\n");
							return -1;
					}
					break;
				default:
					fprintf(stderr, "Error: unsupported A dt of type: %d\n", a_cuda_dt);
					return -1;
			}
			*cublas_compute_type = CUBLAS_COMPUTE_32F;
			*scale_cuda_dt = CUDA_R_32F;
			return 0;
		case FP16:
			switch (a_cuda_dt){
				case CUDA_R_16F:
					if (b_cuda_dt != CUDA_R_16F){
						fprintf(stderr, "Error: if compute type is FP16 and A dt = FP16, B dt must also be FP16...\n");
						return -1;
					}
					if (c_cuda_dt != CUDA_R_16F){
						fprintf(stderr, "Error: if compute type is FP16 and A dt = B dt = FP16, C dt must also be FP16...\n");
						return -1;
					}
					if (d_cuda_dt != CUDA_R_16F){
						fprintf(stderr, "Error: if compute type is FP16 and A dt = B dt = C dt = FP16, D dt must also be FP16...\n");
						return -1;
					}
					*cublas_compute_type = CUBLAS_COMPUTE_16F;
					*scale_cuda_dt = CUDA_R_16F;
					return 0;
				case CUDA_R_32F:
					if ((b_cuda_dt != CUDA_R_32F) || (c_cuda_dt != CUDA_R_32F) || (d_cuda_dt != CUDA_R_32F)) {
						fprintf(stderr, "Error: if compute type is FP16 and A dt = FP32, B dt, C dt, and D dt must also be FP32...\n");
						return -1;
					}
					*cublas_compute_type = CUBLAS_COMPUTE_32F_FAST_16F;
					*scale_cuda_dt = CUDA_R_32F;
					return 0;
				default:
					fprintf(stderr, "Error: if compute dtpe is FP16, A dt must be either FP16 or FP32...\n");
					return -1;
			}
			break;
		case BF16:
			if ((a_cuda_dt != CUDA_R_32F) || (b_cuda_dt != CUDA_R_32F) || (c_cuda_dt != CUDA_R_32F) || (d_cuda_dt != CUDA_R_32F)) {
				fprintf(stderr, "Error: if compute type is BF16 and A dt, B dt, C dt, and D dt must all be FP32...\n");
				return -1;
			}
			*cublas_compute_type = CUBLAS_COMPUTE_32F_FAST_16BF;
			*scale_cuda_dt = CUDA_R_32F;
			return 0;
		default:
			fprintf(stderr, "Error: compute type must be either FP32, FP16, or BF16...\n");
			return -1;
	}

	// won't get here
	return -1;
}

static int destroy_matmul_params(Cublas_Matmul_Params * matmul_params){

	int ret;

	cublasStatus_t status;

	if (matmul_params -> computeDesc){
		status = cublasLtMatmulDescDestroy(matmul_params -> computeDesc);
		if (status != CUBLAS_STATUS_SUCCESS){
			fprintf(stderr, "Error: could not destroy computeDesc...\n");
			return -1;
		}
	}

	if (matmul_params -> Adesc){
		status = cublasLtMatrixLayoutDestroy(matmul_params -> Adesc);
		if (status != CUBLAS_STATUS_SUCCESS){
			fprintf(stderr, "Error: could not destroy Adesc...\n");
			return -1;
		}
	}

	if (matmul_params -> Bdesc){
		status = cublasLtMatrixLayoutDestroy(matmul_params -> Bdesc);
		if (status != CUBLAS_STATUS_SUCCESS){
			fprintf(stderr, "Error: could not destroy Adesc...\n");
			return -1;
		}
	}

	if (matmul_params -> Ddesc){
		status = cublasLtMatrixLayoutDestroy(matmul_params -> Ddesc);
		if (status != CUBLAS_STATUS_SUCCESS){
			fprintf(stderr, "Error: could not destroy Adesc...\n");
			return -1;
		}
	}

	if ((matmul_params -> Cdesc) && (!(matmul_params -> same_cDesc))) {
		status = cublasLtMatrixLayoutDestroy(matmul_params -> Cdesc);
		if (status != CUBLAS_STATUS_SUCCESS){
			fprintf(stderr, "Error: could not destroy Ddesc...\n");
			return -1;
		}
	}

	if (matmul_params -> pref){
		status = cublasLtMatmulPreferenceDestroy(matmul_params -> pref);
		if (status != CUBLAS_STATUS_SUCCESS){
			fprintf(stderr, "Error: could not destroy pref...\n");
			return -1;
		}
	}

	return 0;
}

// ASSUMING UNDERLYING STORAGE IS ROW-MAJOR!
static int set_cublas_matmul_params(Cublas_Matmul_Params * matmul_params, cublasLtHandle_t cublas_handle,
									 DataType a_dt, DataType b_dt, DataType c_dt, DataType d_dt,
									 DataType compute_dt,
									 int M, int K, int N,
									 float alpha, float beta,
									 uint64_t workspaceBytes, void * workspace,
									 void * A, void * B, void * C, void * D,
									 int num_sms,
									 cublasLtMatmulAlgo_t * algo_to_use){

	int ret;

	cublasStatus_t status;

	cudaDataType a_cuda_dt;
	cudaDataType b_cuda_dt;
	cudaDataType c_cuda_dt;
	cudaDataType d_cuda_dt;

	// NOTE:
	// We assume underlying storage is Row-Major and we are required to use "TN" format
	// This means that from cuBLAS perspective (which assumes col-major) we can reverse the
	// ordering of A and B matrices such that we will compute C^T = B^T * A^T



	ret = dtype_to_cuda_dtype(b_dt, &a_cuda_dt);
	if (ret){
		fprintf(stderr, "Error: unsupported A dtype of %s\n", datatype_as_string(a_dt));
		return -1;
	}

	ret = dtype_to_cuda_dtype(a_dt, &b_cuda_dt);
	if (ret){
		fprintf(stderr, "Error: unsupported B dtype of %s\n", datatype_as_string(b_dt));
		return -1;
	}

	ret = dtype_to_cuda_dtype(d_dt, &d_cuda_dt);
	if (ret){
		fprintf(stderr, "Error: unsupported D dtype of %s\n", datatype_as_string(c_dt));
		return -1;
	}

	if ((c_dt == NONE) || (!C)) {
		if ((a_dt == FP8E4M3) || (a_dt == FP8E5M2)){
			c_cuda_dt = CUDA_R_16F;
		}
		else {
			c_cuda_dt = d_cuda_dt;
		}
	}
	else{
		ret = dtype_to_cuda_dtype(c_dt, &c_cuda_dt);
		if (ret){
			fprintf(stderr, "Error: unsupported C dtype of %s\n", datatype_as_string(d_dt));
			return -1;
		}
	}

	cudaDataType scale_cuda_dt;
	cublasComputeType_t cublas_compute_type;

	ret = set_cublas_compute_scale_types(&cublas_compute_type, &scale_cuda_dt, compute_dt, a_cuda_dt, b_cuda_dt, c_cuda_dt, d_cuda_dt);
	if (ret){
		fprintf(stderr, "Error: unable to get cublas compute type and scale type...\n");
		return -1;
	}

	status = cublasLtMatmulDescCreate(&(matmul_params -> computeDesc), cublas_compute_type, scale_cuda_dt);
	if (status != CUBLAS_STATUS_SUCCESS){
		fprintf(stderr, "Error: cublaslt matmul desc could not be created...\n");
		return -1;
	}

	if (num_sms > 0){
		status = cublasLtMatmulDescSetAttribute(matmul_params -> computeDesc, CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET, &num_sms, sizeof(num_sms));
		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf(stderr, "Error: matmul desc sm count attribute could not be set\n");
			return -1;
		}
	}

	cublasOperation_t transa = CUBLAS_OP_T;
	cublasOperation_t transb = CUBLAS_OP_N;

	status = cublasLtMatmulDescSetAttribute(matmul_params -> computeDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "Error: matmul desc attribute transa could not be set\n");
		return -1;
	}
	status = cublasLtMatmulDescSetAttribute(matmul_params -> computeDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "Error: matmul desc attribute transb could not be set\n");
		return -1;
	}

	// Now Adesc actually referes the B matrix. a_cuda_dt has already been set appropriately

	status = cublasLtMatrixLayoutCreate(&(matmul_params -> Adesc), a_cuda_dt, K, N, K);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "Error: Adesc matmul layout could not be created\n");
		return -1;
	}

	// Now Bdesc actually referes the A matrix. b_cuda_dt has already been set appropriately
	status = cublasLtMatrixLayoutCreate(&(matmul_params -> Bdesc), b_cuda_dt, K, M, K);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "Error: Bdesc matmul layout could not be created\n");
		return -1;
	}

	status = cublasLtMatrixLayoutCreate(&(matmul_params -> Ddesc), d_cuda_dt, N, M, N);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "Error: Cdesc matmul layout could not be created\n");
		return -1;
	}

	if ((c_dt == NONE) || (!C)){
		if ((a_dt == FP8E4M3) || (a_dt == FP8E5M2)){
			matmul_params -> same_cDesc = false;
			status = cublasLtMatrixLayoutCreate(&(matmul_params -> Cdesc), c_cuda_dt, N, M, N);
               		if (status != CUBLAS_STATUS_SUCCESS) {
                        	fprintf(stderr, "Error: Ddesc matmul layout could not be created\n");
                        	return -1;
                	}
		}
		else{
			matmul_params -> same_cDesc = true;
			memcpy(&(matmul_params -> Cdesc), &(matmul_params -> Ddesc), sizeof(cublasLtMatrixLayout_t));
		}
	}
	else{
		matmul_params -> same_cDesc = false;
		status = cublasLtMatrixLayoutCreate(&(matmul_params -> Cdesc), c_cuda_dt, N, M, N);
		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf(stderr, "Error: Ddesc matmul layout could not be created\n");
			return -1;
		}
	}


	matmul_params -> alpha_f = alpha;
	matmul_params -> beta_f = beta;

	if (scale_cuda_dt == CUDA_R_16F){
		matmul_params -> alpha_h = fp32_to_fp16(matmul_params -> alpha_f);
		matmul_params -> beta_h = fp32_to_fp16(matmul_params -> beta_f);
		matmul_params -> alpha = &(matmul_params -> alpha_h);
		matmul_params -> beta = &(matmul_params -> beta_h);
	}
	else if (scale_cuda_dt == CUDA_R_32F){
		matmul_params -> alpha = &(matmul_params -> alpha_f);
		matmul_params -> beta = &(matmul_params -> beta_f);
	}
	else{
		fprintf(stderr, "Error: unsupported scale type...\n");
		return -1;
	}

	matmul_params -> workspaceBytes = (size_t) workspaceBytes;
	matmul_params -> workspace = workspace;

	// search for algo if not provided
	if (!algo_to_use){

		status = cublasLtMatmulPreferenceCreate(&(matmul_params -> pref));
		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf(stderr, "Error: matmul pref could not be created\n");
			return -1;
		}
		// Allowing just a small amount of workspace mem (2 MB) makes a big difference
		status = cublasLtMatmulPreferenceSetAttribute(matmul_params -> pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceBytes, sizeof(workspaceBytes));
		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf(stderr, "Error: matmul pref attribute could not be set\n");
			return -1;
		}
			
		int algoCount = MAX_ALGO_SEARCH;
		int retAlgoCount = 0;

		status = cublasLtMatmulAlgoGetHeuristic(cublas_handle, matmul_params -> computeDesc, matmul_params -> Adesc, matmul_params -> Bdesc, matmul_params -> Cdesc, matmul_params -> Ddesc, matmul_params -> pref, algoCount, matmul_params -> heuristicResultsArray, &retAlgoCount);
		if ((status != CUBLAS_STATUS_SUCCESS) || (retAlgoCount == 0)) {
			fprintf(stderr, "Error: could not get matmul algo heuristic: %s\n", cublasLtGetStatusString(status));
			return -1;
		}
	}

	if ((c_dt == NONE)|| (!C)) {
		C = D;
	}

	// Ensure A and B are swapped
	// to maintain illusion of row-major computation!
	matmul_params -> A = B;
	matmul_params -> B = A;
	matmul_params -> C = C;
	matmul_params -> D = D;

	return 0;

}


// Assumes all row major
int do_cublas_matmul(CUstream compute_stream, cublasLtHandle_t cublas_handle,
						DataType a_dt, DataType b_dt, DataType c_dt, DataType d_dt,
						DataType compute_dt,
						int M, int K, int N,
						float alpha, float beta,
						uint64_t workspaceBytes, void * workspace,
						void * A, void * B, void * C, void * D,
						int num_sms,
						cublasLtMatmulAlgo_t * algo_to_use,
						cublasLtMatmulAlgo_t * algo_to_save,
						char * prof_label) {

	

	int ret;

	cublasStatus_t status;

	Cublas_Matmul_Params * matmul_params = malloc(sizeof(Cublas_Matmul_Params));
	if (!matmul_params){
		fprintf(stderr, "Error: malloc failed to alloc space to hold matmul params...\n");
		return -1;
	}

	memset(matmul_params, 0, sizeof(Cublas_Matmul_Params));

	ret = set_cublas_matmul_params(matmul_params, cublas_handle,
									a_dt, b_dt, c_dt, d_dt, compute_dt,
									M, K, N,
									alpha, beta,
									workspaceBytes, workspace,
									A, B, C, D,
									num_sms,
									algo_to_use);
	if (ret){
		fprintf(stderr, "Error: unable to set cublas matmul params...\n");
		ret = destroy_matmul_params(matmul_params);
		if (ret){
			fprintf(stderr, "Error: had error destroying matmul params...\n");
		}
		free(matmul_params);
		return -1;
	}

	if (TO_PROF_MATMUL && prof_label){
		profile_range_push(prof_label);
	}

	cublasLtMatmulAlgo_t * true_algo;
	if (algo_to_use){
		true_algo = algo_to_use;
	}
	else{
		true_algo = &(matmul_params -> heuristicResultsArray[0].algo);
		if (algo_to_save){
			memcpy(algo_to_save, true_algo, sizeof(cublasLtMatmulAlgo_t));
		}
	}

	status = cublasLtMatmul(cublas_handle,
							matmul_params -> computeDesc,
							matmul_params -> alpha,
							matmul_params -> A,
							matmul_params -> Adesc,
							matmul_params -> B,
							matmul_params -> Bdesc,
							matmul_params -> beta,
							matmul_params -> C,
							matmul_params -> Cdesc,
							matmul_params -> D,
							matmul_params -> Ddesc,
							true_algo,
							matmul_params -> workspace,
							matmul_params -> workspaceBytes,
							compute_stream);


	if (TO_PROF_MATMUL && prof_label){
		profile_range_pop();
	}


	if (status != CUBLAS_STATUS_SUCCESS){
		fprintf(stderr, "Error: cublasLtMatmul failed...\n");
		ret = destroy_matmul_params(matmul_params);
		if (ret){
			fprintf(stderr, "Error: had error destroying matmul params...\n");
		}
		free(matmul_params);
		return -1;
	}

	ret = destroy_matmul_params(matmul_params);
	if (ret){
		fprintf(stderr, "Error: had error destroying matmul params...\n");
		free(matmul_params);
		return -1;
	}

	free(matmul_params);
	return 0;

}
