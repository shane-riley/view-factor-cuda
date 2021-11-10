// Header Guards
#ifndef GLOBALS_H_
#define GLOBALS_H_

// STDLIB imports that we want to apply everywhere
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cuda.h>
#include <cublas.h>
#include <cublas_v2.h>
#include <helper_cuda.h>
#include <device_launch_parameters.h>
#include <chrono>
#include <stdexcept>

// Turn on self-intersection at compile time
//#define DO_SELF_INTERSECTION
#define DO_BACKFACE_CULLING

// This sets the namespace to std, which shortens many of our stdlib type and function references
using namespace std;

// These constants will always be availible on CPU
const static int X = 0;
const static int Y = 1;
const static int Z = 2;
const string OUT_FILE = "output.txt";
const static int BLOCKSIZE = 32;
const static double PI = 3.14159265359;

// Time
typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::seconds sec;
typedef std::chrono::duration<float> fsec;

#endif  /* GLOBALS_H_ */  