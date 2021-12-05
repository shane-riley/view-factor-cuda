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

	// Determine which selfint to use
	char *selfIntSettingStr;
	int selfIntSetting = 0;
	selfIntSettingStr = getenv(SELFINT_ENV_VAR);
	if (selfIntSettingStr != NULL) {
		selfIntSetting = atoi(selfIntSettingStr);
	}
	// Loop through assigned emitters
	auto tStart = Time::now();
	switch (selfIntSetting)
	{
		case 0:
			for (int e = startEmitter; e < stopEmitter; e++) {
				cudaEvaluateEmitter(e, startEmitter, numEmitters);
			}
			break;
		case 1:
			for (int e = startEmitter; e < stopEmitter; e++) {
				cudaEvaluateEmitterSelfIntEmitter(e, startEmitter, numEmitters);
			}
			break;
		case 2:
			for (int e = startEmitter; e < stopEmitter; e++) {
				cudaEvaluateEmitterSelfIntReceiver(e, startEmitter, numEmitters);
			}
			break;
		case 3:
			for (int e = startEmitter; e < stopEmitter; e++) {
				cudaEvaluateEmitterSelfIntBoth(e, startEmitter, numEmitters);
			}
			break;
		default:
			return -1;
	}

	// Kernel invocations are async, wait for actions to complete
	checkCudaErrors(cudaDeviceSynchronize());
	auto tGPUDone = Time::now();

	msec tGPU = chrono::duration_cast<msec>(tGPUDone - tStart);

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

	cout << "Thread " << deviceNum << " Time [s]: " << setprecision(6) << (float) tGPU.count()/1000.0 << endl;

	// Return sum
	return cpuTotal;
}

// Put kernel wrappers down here

void MTCalculator::cudaEvaluateEmitter(int e, int start, int num) {
	int nblocks = (gpuReceiver.arraySize / BLOCKSIZE) + 1;
	evaluateEmitter<<<nblocks, BLOCKSIZE>>> (e, start, gpuEmitter, gpuReceiver, gpuBlocker, result);
}

void MTCalculator::cudaEvaluateEmitterSelfIntEmitter(int e, int start, int num) {
	int nblocks = (gpuReceiver.arraySize / BLOCKSIZE) + 1;
	evaluateEmitterSelfIntEmitter<<<nblocks, BLOCKSIZE>>> (e, start, gpuEmitter, gpuReceiver, gpuBlocker, result);
}

void MTCalculator::cudaEvaluateEmitterSelfIntReceiver(int e, int start, int num) {
	int nblocks = (gpuReceiver.arraySize / BLOCKSIZE) + 1;
	evaluateEmitterSelfIntReceiver<<<nblocks, BLOCKSIZE>>> (e, start, gpuEmitter, gpuReceiver, gpuBlocker, result);
}

void MTCalculator::cudaEvaluateEmitterSelfIntBoth(int e, int start, int num) {
	int nblocks = (gpuReceiver.arraySize / BLOCKSIZE) + 1;
	evaluateEmitterSelfIntBoth<<<nblocks, BLOCKSIZE>>> (e, start, gpuEmitter, gpuReceiver, gpuBlocker, result);
}

//double MTCalculator::cudaSumVector(int e, double* result, double* total) {
	//sumVector << <1, e >> > (e, result, total);
//}