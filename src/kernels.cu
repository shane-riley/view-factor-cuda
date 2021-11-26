// This file will contain ALL of our GPU function implementations
#include "kernels.cuh"
#define pi 3.14159265359

__device__ double vectorMagnitude(double3 r) {
	return sqrt(r.x * r.x + r.y * r.y + r.z * r.z);
}

__device__ double vectorDot(double3 r1, double3 r2) {
	return r1.x * r2.x + r1.y * r2.y + r1.z * r2.z;
}

__device__ double3 vectorSub(double3 r1, double3 r2) {
	double3 r3 = { r1.x - r2.x, r1.y - r2.y, r1.z - r2.z };
	return r3;
}

__device__ double3 vectorCross(double3 r1, double3 r2) {
	double3 r3 = { r1.y * r2.z - r1.z * r2.y,
					r1.z * r2.x - r1.x * r2.z,
					r1.x * r2.y - r1.y * r2.x };
	return r3;
}

__device__ double intersectionDistance(int ei, int bi, double3 r, GPUGeometry& e, GPUGeometry &b) {

	// r cross edge
	double3 pvec = vectorCross(r, b.edgeCA[bi]);

	double det = vectorDot(b.edgeBA[bi], pvec);

	// CULLING (no blocking behind the emitter)
#ifdef DO_BACKFACE_CULLING
	if (det < 0) {
		return 0;
	}
#endif

	// Ray is parallel to plane
	if (det < 1e-8 && det > -1e-8) {
		return 0;
	}

	double invDet = 1.0 / det;

	double3 tvec = vectorSub(e.center[ei], b.vertexA[bi]);

	double u = invDet * vectorDot(tvec, pvec);

	if (u < 0 || u > 1) {
		return 0;
	}
	
	double3 qvec = vectorCross(tvec, b.edgeBA[bi]);

	double v = invDet * vectorDot(r, qvec);

	if (v < 0 || (u + v) > 1) {
		return 0;
	}
	else {
		return vectorDot(b.edgeCA[bi], qvec) * invDet;
	}
}

__global__ void evaluateEmitter(int e, int startEmitter, GPUGeometry gpuEmitter, GPUGeometry gpuReceiver, GPUGeometry gpuBlocker, double* result) {
	size_t r = blockIdx.x * blockDim.x + threadIdx.x;

	if (r < gpuReceiver.arraySize)
	{
		// Cast ray
		double3 ray = vectorSub(gpuReceiver.center[r], gpuEmitter.center[e]);

		double rayMagnitude = vectorMagnitude(ray);

		// Check for blocking blockers
		for (int b = 0; b < gpuBlocker.arraySize; b++) {
			double dist = intersectionDistance(e, b, ray, gpuEmitter, gpuBlocker);

			// If intersected, kill the thread
			if (dist != 0 && dist <= rayMagnitude) {
				result[r] += 0;
				return;
			}
		}


#ifdef NO_SELF_INTERSECTION
		// Do nothing
#else

		// Check for self-intersection of emitters
		for (int b = 0; b < gpuEmitter.arraySize; b++) {
			if (e == b) continue;

			double dist = intersectionDistance(e, b, ray, gpuEmitter, gpuEmitter);

			// If intersected, kill the thread
			if (dist != 0 && dist <= rayMagnitude) {
				result[r] += 0;
				return;
			}
		}

		// Check for self-intersection of receivers
		// for (int b = 0; b < gpuReceiver.arraySize; b++) {
		// 	if (r == b) continue;

		// 	double dist = intersectionDistance(e, b, ray, gpuEmitter, gpuReceiver);

		// 	// If intersected, kill the thread
		// 	if (dist != 0 && dist <= rayMagnitude) {
		// 		result[r] += 0;
		// 		return;
		// 	}
		// }

#endif

		double3 eNormal = gpuEmitter.normal[e];
		double3 rNormal = gpuReceiver.normal[r];
		double emitterDenominator = vectorMagnitude(eNormal) * rayMagnitude;
		double receiverDenominator = vectorMagnitude(rNormal) * rayMagnitude;
		double emitterNormalDotRay = vectorDot(eNormal, ray);
		double receiverNormalDotRay = vectorDot(rNormal, ray);

		double cosThetaOne = abs(emitterNormalDotRay / emitterDenominator);
		double cosThetaTwo = abs(receiverNormalDotRay / receiverDenominator);

		result[r] += cosThetaOne * cosThetaTwo * gpuEmitter.area[e] * gpuReceiver.area[r]
			/ (pi * rayMagnitude * rayMagnitude);

		// Check for emitting blockers
	}
}
