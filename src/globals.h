// Header Guards
#ifndef GLOBALS_H_
#define GLOBALS_H_

// STDLIB imports that we want to apply everywhere
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <assert.h>
#include <limits>
#include <iomanip>

// CUDA Toolkit imports
#include <cuda.h>
#include <cublas.h>
#include <cublas_v2.h>
#include <device_launch_parameters.h>

// CUDA sample imports
#include <helper_cuda.h>
#include <thrust/reduce.h>

// Turn on self-intersection at compile time
//#define NO_SELF_INTERSECTION
#define DO_BACKFACE_CULLING

// This sets the namespace to std, which shortens many of our stdlib type and function references
using namespace std;

// Standard CUDA blocksize; adjusting has no noticeable effect on runtime
const static int BLOCKSIZE = 32;
const static char SELFINT_ENV_VAR[10] = "SELF_INT";
const static double PI = 3.14159265358979311599796346854;

// Time
typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::milliseconds msec;
typedef std::chrono::duration<float> fsec;

#endif  /* GLOBALS_H_ */  