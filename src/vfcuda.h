#ifndef _VF_CUDA_H_
#define _VF_CUDA_H_

#include "globals.h"
#include "Geometry.h"
#include "MTCalculator.cuh"

using namespace std;

double getVF(string ef, string rf, string bf);

double getVF(string ef, string rf);

#endif  /* _VF_CUDA_H_ */