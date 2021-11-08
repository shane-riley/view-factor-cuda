#include "MTCalculator.cuh"
#include "kernels.cuh"

MTCalculator::MTCalculator(Geometry& cpuEmitter, Geometry& cpuReceiver, Geometry& cpuBlocker, int ddeviceNum, int emitterBegin, int emitterEnd) {
	cudaSetDevice(ddeviceNum);
	gpuEmitter = GPUGeometry(cpuEmitter);
	gpuReceiver = GPUGeometry(cpuReceiver);
	gpuBlocker = GPUGeometry(cpuBlocker);
	deviceNum = ddeviceNum;
	startEmitter = emitterBegin;
	stopEmitter = emitterEnd;

	int numResults = (stopEmitter - startEmitter) * gpuReceiver.arraySize;
	checkCudaErrors(cudaMalloc((void**)&result, numResults * sizeof(double)));
}

void MTCalculator::freeMemory() {
	gpuEmitter.freeMemory();
	gpuReceiver.freeMemory();
	gpuBlocker.freeMemory();
	checkCudaErrors(cudaFree(result));
}

double MTCalculator::calculateVF() {

	// Get number of emitters evaluated
	int numEmitters = stopEmitter - startEmitter;

	// Get number of receivers evaluated
	int numReceivers = gpuReceiver.arraySize;

	long numResults = numEmitters * numReceivers;

	// Loop through assigned emitters
	for (int e = startEmitter; e < stopEmitter; e++) {
		cudaEvaluateEmitter(e, startEmitter, numEmitters);
	}

	// Sum all of the results
	
	// Sum on CPU for now
	//double* total;
	//checkCudaErrors(cudaMalloc((void**)&total, 1 * sizeof(double)));

	// Sum on CPU for now
	// TODO: Implement a GPU-based block reduction
	double* cpuResult = (double*)malloc(numResults * sizeof(double));
	checkCudaErrors(cudaMemcpy(cpuResult, result, numResults * sizeof(double), cudaMemcpyDeviceToHost));
	double cpuTotal = 0.0;


	for (int res = 0; res < numResults; res++) {
		cpuTotal += cpuResult[res];
	}
	free(cpuResult);

	// Return sum
	return cpuTotal;
}

// Put kernel wrappers down here

void MTCalculator::cudaEvaluateEmitter(int e, int start, int num) {
	int nblocks = (gpuReceiver.arraySize / BLOCKSIZE) + 1;
	evaluateEmitter<<<nblocks, BLOCKSIZE>>> (e, start, num, gpuEmitter, gpuReceiver, gpuBlocker, result);
}

//double MTCalculator::cudaSumVector(int e, double* result, double* total) {
	//sumVector << <1, e >> > (e, result, total);
//}