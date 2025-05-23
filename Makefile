CC = gcc

DEV_CFLAGS = -g -std=c99 -Wall -pedantic
BUILD_CFLAGS = -O3 -std=c99 -Wall -pedantic
CFLAGS = ${DEV_CFLAGS}


ALL_OBJS = dtype.o cuda_drv.o backend_profile.o create_matrix.o cublas_helper.o 

EXECS = benchCublasLtMatmul benchCublasLtMatmulSpatialShare


CUDA_INCLUDE_DIR = /usr/local/cuda/include
CUDA_LIB_DIR = /usr/local/cuda/lib64
CUDA_LIB_LINKS = -lcuda -lcublasLt

LIB_LINKS = -lm -pthread -ldl


all: ${EXECS}


benchCublasLtMatmul: bench_cublaslt_matmul.c ${ALL_OBJS}
	${CC} ${CFLAGS} $^ -o $@ -I ${CUDA_INCLUDE_DIR} ${LIB_LINKS} -L${CUDA_LIB_DIR} ${CUDA_LIB_LINKS}


benchCublasLtMatmulSpatialShare: bench_cublaslt_matmul_spatial_share.c ${ALL_OBJS}
	${CC} ${CFLAGS} $^ -o $@ -I ${CUDA_INCLUDE_DIR} ${LIB_LINKS} -L${CUDA_LIB_DIR} ${CUDA_LIB_LINKS}



dtype.o: dtype.c
	${CC} ${CFLAGS} -c $^ -I ${CUDA_INCLUDE_DIR}


cuda_drv.o: cuda_drv.c
	${CC} ${CFLAGS} -c $^ -I ${CUDA_INCLUDE_DIR}


backend_profile.o: backend_profile.c
	${CC} ${CFLAGS} -c $^ -I ${CUDA_INCLUDE_DIR}


create_matrix.o: create_matrix.c
	${CC} ${CFLAGS} -c $^ -I ${CUDA_INCLUDE_DIR}


cublas_helper.o: cublas_helper.c
	${CC} ${CFLAGS} -c $^ -I ${CUDA_INCLUDE_DIR}


clean:
	rm -f ${EXECS} ${ALL_OBJS}
