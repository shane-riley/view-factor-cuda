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

	int numResults = gpuReceiver.arraySize;
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

	long numResults = numReceivers;

	// Loop through assigned emitters
	auto tStart = Time::now();

	for (int e = startEmitter; e < stopEmitter; e++) {
		cudaEvaluateEmitter(e, startEmitter, numEmitters);
	}
	cout << "GPU DONE!" << endl;
	auto tGPUDone = Time::now();

	sec tGPU = chrono::duration_cast<sec>(tGPUDone - tStart);

	// Sum all of the results
	// Sum on CPU for now
	// TODO: Implement a GPU-based block reduction
	double* cpuResult = (double*)malloc(numResults * sizeof(double));
	checkCudaErrors(cudaMemcpy(cpuResult, result, numResults * sizeof(double), cudaMemcpyDeviceToHost));
	double cpuTotal = 0.0;


	for (int res = 0; res < numResults; res++) {
		cpuTotal += cpuResult[res];
	}
	free(cpuResult);

	auto tAllDone = Time::now();

	sec tCPU = chrono::duration_cast<sec>(tAllDone - tGPUDone);

	cout << deviceNum << ": (Time CPU, Time GPU) [s] => (" << tCPU.count() << ", " << tGPU.count() << ")" << endl;

	// Return sum
	return cpuTotal;
}

// Put kernel wrappers down here

void MTCalculator::cudaEvaluateEmitter(int e, int start, int num) {
	int nblocks = (gpuReceiver.arraySize / BLOCKSIZE) + 1;
	evaluateEmitter<<<nblocks, BLOCKSIZE>>> (e, start, gpuEmitter, gpuReceiver, gpuBlocker, result);
}

//double MTCalculator::cudaSumVector(int e, double* result, double* total) {
	//sumVector << <1, e >> > (e, result, total);
//}