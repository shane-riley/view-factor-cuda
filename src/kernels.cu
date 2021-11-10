// This file will contain ALL of our GPU function implementations
#include "kernels.cuh"
#define pi 3.14159265359

__device__ double vectorMagnitude(double3 r) {
	return sqrt(r.x * r.x + r.y * r.y + r.z * r.z);
}

__device__ double vectorDot(double3 r1, double3 r2) {
	return r1.x * r2.x + r1.y * r2.y + r1.z * r2.z;
}

__device__ double intersectionDistance(int bi, int ei, double3 r, GPUGeometry& e, GPUGeometry &b) {

	double3 pvec = { r.y * b.edgeCAZ[bi] - r.z * b.edgeCAY[bi],
					r.z * b.edgeCAX[bi] - r.x * b.edgeCAZ[bi],
					r.x * b.edgeCAY[bi] - r.y * b.edgeCAX[bi] };


	double det = b.edgeBAX[bi] * pvec.x
		+ b.edgeBAY[bi] * pvec.y
		+ b.edgeBAZ[bi] * pvec.z;

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

	double invDet = 1 / det;

	double3 tvec = { e.centerX[ei] - b.vertexAX[bi],
					e.centerY[ei] - b.vertexAY[bi],
					e.centerZ[ei] - b.vertexAZ[bi] };

	double u = invDet * vectorDot(tvec, pvec);

	if (u < 0 || u > 1) {
		return 0;
	}

	double3 qvec = { tvec.y * b.edgeBAZ[bi] - tvec.z * b.edgeBAY[bi],
					tvec.z * b.edgeBAX[bi] - tvec.x * b.edgeBAZ[bi],
					tvec.x * b.edgeBAY[bi] - tvec.y * b.edgeBAX[bi] };

	double v = invDet * vectorDot(r, qvec);

	if (v < 0 || (u + v) > 1) {
		return 0;
	}
	else {
		return (b.edgeCAX[bi] * qvec.x + b.edgeCAY[bi] * qvec.y + b.edgeCAZ[bi] * qvec.z) * invDet;
	}
}

__global__ void evaluateEmitter(int e, int startEmitter, GPUGeometry gpuEmitter, GPUGeometry gpuReceiver, GPUGeometry gpuBlocker, double* result) {
	size_t r = blockIdx.x * blockDim.x + threadIdx.x;

	if (r < gpuReceiver.arraySize)
	{
		// Cast ray
		double3 ray = { gpuReceiver.centerX[r] - gpuEmitter.centerX[e],
						gpuReceiver.centerY[r] - gpuEmitter.centerY[e],
						gpuReceiver.centerZ[r] - gpuEmitter.centerZ[e] };
		double rayMagnitude = vectorMagnitude(ray);

		// Check for blocking blockers
		for (int b = 0; b < gpuBlocker.arraySize; b++) {
			double dist = intersectionDistance(b, e, ray, gpuEmitter, gpuBlocker);

			// If intersected, kill the thread
			if (dist != 0 && dist <= rayMagnitude) {
				result[r] += 0;
				return;
			}
		}


#ifdef DO_SELF_INTERSECTION

		// Check for self-intersection of emitters
		for (int b = 0; b < gpuEmitter.arraySize; b++) {
			if (e == b) continue;

			double dist = intersectionDistance(b, e, ray, gpuEmitter, gpuEmitter);

			// If intersected, kill the thread
			if (dist != 0 && dist <= rayMagnitude) {
				result[r] += 0;
				return;
			}
		}

		// Check for self-intersection of receivers
		for (int b = 0; b < gpuReceiver.arraySize; b++) {
			if (r == b) continue;

			double dist = intersectionDistance(b, e, ray, gpuEmitter, gpuReceiver);

			// If intersected, kill the thread
			if (dist != 0 && dist <= rayMagnitude) {
				result[r] += 0;
				return;
			}
		}

#endif

		double3 eNormal = { gpuEmitter.normalX[e], gpuEmitter.normalY[e], gpuEmitter.normalZ[e] };
		double3 rNormal = { gpuReceiver.normalX[r], gpuReceiver.normalY[r], gpuReceiver.normalZ[r] };
		double emitterDenominator = vectorMagnitude(eNormal) * rayMagnitude;
		double receiverDenominator = vectorMagnitude(rNormal) * rayMagnitude;
		double emitterNormalDotRay = vectorDot(eNormal, ray);
		double receiverNormalDotRay = vectorDot(rNormal, ray);

		double cosThetaOne = emitterNormalDotRay / emitterDenominator;
		double cosThetaTwo = receiverNormalDotRay / receiverDenominator;

		if (cosThetaOne < 0) {
			cosThetaOne = -cosThetaOne;
		}
		if (cosThetaTwo < 0) {
			cosThetaTwo = -cosThetaTwo;
		}

		result[r] += cosThetaOne * cosThetaTwo * gpuEmitter.area[e] * gpuReceiver.area[r]
			/ (pi * rayMagnitude * rayMagnitude);

		// Check for emitting blockers
	}
}
