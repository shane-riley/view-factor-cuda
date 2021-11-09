// Header Guards
#ifndef MT_CALC_H_
#define MT_CALC_H_

#include "Geometry.h"
#include "GPUGeometry.h"

class MTCalculator
{
public:

	// Instance variables
	int deviceNum;
	int startEmitter;
	int stopEmitter;

	// GPU Pointers (for this device)
	GPUGeometry gpuEmitter;
	GPUGeometry gpuReceiver;
	GPUGeometry gpuBlocker;
	
	// Emitter-length
	double* result;

	// Constructor
	MTCalculator(Geometry &cpuEmitter, Geometry &cpureceiver, Geometry & cpublocker, int deviceNum, int emitterBegin, int emitterEnd);

	// Calculate method
	double calculateVF();

	// Wrapper methods for gpu kernels
	void cudaEvaluateEmitter(int e, int start, int num);

	void freeMemory();

	//double MTCalculator::cudaSumVector(int e, double* result, double* total);
};

#endif  /* MT_CALC_H_ */

