// This file will contain ALL of our GPU function declarations
#ifndef KERNELS_H_
#define KERNELS_H_
#include "GPUGeometry.h"

// cudaEvaluateEmitter
__global__ void evaluateEmitter(int e, int startEmitter, GPUGeometry gpuEmitter, GPUGeometry gpuReceiver, GPUGeometry gpuBlocker, double *result);
__global__ void evaluateEmitterSelfIntEmitter(int e, int startEmitter, GPUGeometry gpuEmitter, GPUGeometry gpuReceiver, GPUGeometry gpuBlocker, double* result);
__global__ void evaluateEmitterSelfIntReceiver(int e, int startEmitter, GPUGeometry gpuEmitter, GPUGeometry gpuReceiver, GPUGeometry gpuBlocker, double* result);
__global__ void evaluateEmitterSelfIntBoth(int e, int startEmitter, GPUGeometry gpuEmitter, GPUGeometry gpuReceiver, GPUGeometry gpuBlocker, double* result);

// cudaSumVector
//__global__ void sumVector(int e, double* result, double* total);


#endif  /* KERNELS_H_ */
