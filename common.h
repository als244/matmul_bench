#ifndef COMMON_H
#define COMMON_H


#define MY_MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MY_MAX(X, Y) (((X) > (Y)) ? (X) : (Y))
#define MY_CEIL(a, b) ((a + b - 1) / b)

#include <math.h>
#include <time.h>

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#include <linux/limits.h>
#include <unistd.h>


#include <errno.h>



#ifndef CUDA_H
#define CUDA_H
#include <cuda.h>
#endif


#define MY_MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MY_MAX(X, Y) (((X) > (Y)) ? (X) : (Y))
#define MY_CEIL(a, b) ((a + b - 1) / b)



#include "dtype.h"

#include "backend_profile.h"


#endif