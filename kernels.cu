// This file will contain ALL of our GPU function implementations
#include "kernels.cuh"

__device__ double vectorMagnitude(double x, double y, double z) {
	return sqrt(x * x + y * y + z * z);
}

__global__ void evaluateEmitter(int e, GPUGeometry gpuEmitter, GPUGeometry gpuReceiver, GPUGeometry gpuBlocker, double* result) {
	size_t r = blockIdx.x * blockDim.x + threadIdx.x;

	if (r < gpuReceiver.arraySize)
	{
		// Cast ray
		double rayX = gpuReceiver.centerX[r] - gpuEmitter.centerX[e];
		double rayY = gpuReceiver.centerY[r] - gpuEmitter.centerY[e];
		double rayZ = gpuReceiver.centerZ[r] - gpuEmitter.centerZ[e];
		double rayMagnitude = vectorMagnitude(rayX, rayY, rayZ);

		// Check for blocking blockers
		for (int b = 0; b < gpuBlocker.arraySize; b++) {
			double dist = intersectionDistance(b, rayX, rayY, rayZ);

			// If intersected, kill the thread
			if (dist != 0 && dist <= rayMagnitude) {
				result
			}
		}

		// Check for emitting blockers
	}
}

