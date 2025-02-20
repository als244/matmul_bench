#!/bin/bash

nsys profile -t cuda,nvtx,cublas,osrt --capture-range cudaProfilerApi --capture-range-end stop --gpu-metrics-devices=all --gpu-metrics-set=gh100 --gpu-metrics-frequency=200000 --force-overwrite true -o profiling/M_$1_K_$2_N_$3_fpType_$4_logWorkspace_$5_nummatmuls_$6_numwarmup_$7_usesameb_$8_numcomputeiters_$9 ./benchCublasLtMatmul $1 $2 $3 $4 $5 $6 $7 $8 $9
