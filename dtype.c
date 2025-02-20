#include "dtype.h"

size_t sizeof_dtype(DataType dtype){

	switch(dtype){
		case FP8:
			return 1;
		case FP16:
			return 2;
		case BF16:
			return 2;
		case FP32:
			return 4;
		case U_INT8:
			return 1;
		case U_INT16:
			return 2;
		case U_INT32:
			return 4;
		case U_INT64:
			return 8;
		case REG_INT:
			return sizeof(int);
		case REG_LONG:
			return sizeof(long);
		default:
			return 0;
	}
}



uint16_t fp32_to_fp16(float f) {
    uint32_t x = *(uint32_t*)&f;
    int32_t sign = (x >> 31) & 1;
    int32_t exp = (x >> 23) & 0xFF;
    int32_t mant = x & 0x7FFFFF;

    uint16_t h;

    if (exp == 0) {
        h = sign << 15 | 0; 
    } else if (exp == 0xFF) {
        h = sign << 15 | 0x7C00 | (mant != 0);
    } else {
        exp = exp - 127 + 15;
        if (exp >= 0x1F) {
            h = sign << 15 | 0x7C00; 
        } else if (exp <= 0) {
            h = sign << 15 | 0;
        } else {
            mant = mant >> 13;
            h = sign << 15 | exp << 10 | mant;
        }
    }
    return h;
}

float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    uint32_t f;

    if (exp == 0) {
        f = sign << 31 | 0;
    } else if (exp == 0x1F) {
         f = sign << 31 | 0xFF << 23 | mant << 13;
    } else {
        exp = exp - 15 + 127;
        mant = mant << 13;
        f = sign << 31 | exp << 23 | mant;
    }
    return *(float*)&f;
}

// Function to convert float32 to float8 (limited range and precision)
uint8_t fp32_to_fp8(float f, int e_bits, int m_bits) {

	int bias;
	if ((e_bits == 4) && (m_bits == 3)){
		bias = 7;
	}
	else if ((e_bits == 5) && (m_bits == 2)){
		bias = 15;
	}
	else{
		fprintf(stderr, "Error: unsupprted casting to fp8 mode, only E4M3 and E5M2 supported...\n");
		return 255;
	}

	int sign_bit = 0;
	if (f < 0){
		sign_bit = 1;
	}
	else{
		sign_bit = 0;
	}

	float f_abs = fabsf(f);

	float f_log_val = floorf(log2f(f_abs));

	int exponent;

	if (f_abs == 0){
		exponent = 0;
	}
	else{
		exponent = (int) floorf(f_log_val);
	}

	float mantissa;

	if (f_abs == 0){
		mantissa = 0;
	}
	else{
		mantissa = f_abs / (float) (1 << exponent) - 1.0f;
	}

	int E = exponent + bias;
	int M = round(mantissa * (1 << m_bits));

	uint8_t result = 0;

	result = (sign_bit << 7);

	result |= (E << 3);
	result |= M;

	return result;
}