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
#include "helper_cuda.h"

// This sets the namespace to std, which shortens many of our stdlib type and function references
using namespace std;

// These constants will always be availible on CPU
const static int X = 0;
const static int Y = 1;
const static int Z = 2;
const string OUT_FILE = "output.txt";
const static int BLOCKSIZE = 32;

#endif  /* GLOBALS_H_ */  