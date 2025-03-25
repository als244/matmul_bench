#ifndef DTYPE_H
#define DTYPE_H

#include "common.h"


typedef enum data_type {
	NONE,
	VOID,
	FP4,
	FP8E4M3,
	FP8E5M2,
	FP16,
	BF16,
	FP32,
	FP64,
	U_INT4,
	U_INT8,
	U_INT16,
	U_INT32,
	U_INT64,
	REG_INT,
	REG_LONG
} DataType;


size_t sizeof_dtype(DataType dtype);

uint8_t fp32_to_fp8(float f, int e_bits, int m_bits);

uint16_t fp32_to_fp16(float f);

uint16_t fp32_to_bf16(float f);

float fp16_to_fp32(uint16_t h);

char * datatype_as_string(DataType dtype);

#endif