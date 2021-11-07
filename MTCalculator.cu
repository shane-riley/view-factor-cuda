#include "MTCalculator.h"
#include "kernels.cuh"

MTCalculator::MTCalculator(Geometry& cpuEmitter, Geometry& cpuReceiver, Geometry& cpuBlocker, int ddeviceNum, int emitterBegin, int emitterEnd) {
	cudaSetDevice(ddeviceNum);
	gpuEmitter = GPUGeometry(cpuEmitter);
	gpuReceiver = GPUGeometry(cpuReceiver);
	gpuBlocker = GPUGeometry(cpuBlocker);
	deviceNum = ddeviceNum;
	startEmitter = emitterBegin;
	stopEmitter = emitterEnd;
}

double MTCalculator::calculateVF() {
	checkCudaErrors(cudaMalloc((void**)&result, gpuEmitter.arraySize * sizeof(double)));

	// Loop through assigned emitters
	for (int e = startEmitter; e < stopEmitter; e++) {
		cudaEvaluateEmitter(e);
	}

	// Sum all of the results
	
	// Sum on CPU for now
	//double* total;
	//checkCudaErrors(cudaMalloc((void**)&total, 1 * sizeof(double)));

	// Sum on CPU for now
	// TODO: Implement a GPU-based block reduction
	double* cpuResult = (double*)malloc(gpuEmitter.arraySize * sizeof(double));
	checkCudaErrors(cudaMemcpy(cpuResult, result, gpuEmitter.arraySize * sizeof(double), cudaMemcpyDeviceToHost));
	double cpuTotal;
	for (int e = startEmitter; e < stopEmitter; e++) {
		cpuTotal += cpuResult[e];
	}
	free(cpuResult);

	// Return sum
	return cpuTotal;
}

// Put kernel wrappers down here

double MTCalculator::cudaEvaluateEmitter(int e) {
	evaluateEmitter<<<1, gpuReceiver.arraySize>>> (e, gpuEmitter, gpuReceiver, gpuBlocker, result);
}

//double MTCalculator::cudaSumVector(int e, double* result, double* total) {
	//sumVector << <1, e >> > (e, result, total);
//}