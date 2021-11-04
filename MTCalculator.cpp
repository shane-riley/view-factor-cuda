#include "MTCalculator.h"

MTCalculator::MTCalculator(Geometry& cpuEmitter, Geometry& cpuReceiver, Geometry& cpuBlocker, int ddeviceNum, int emitterBegin, int emitterEnd) {
	gpuEmitter = GPUGeometry(cpuEmitter);
	gpuReceiver = GPUGeometry(cpuReceiver);
	gpuBlocker = GPUGeometry(cpuBlocker);
	deviceNum = ddeviceNum;
	startEmitter = emitterBegin;
	stopEmitter = emitterEnd;
}

double MTCalculator::calculateVF() {
	// Implement me
}

// Put kernel wrappers down here