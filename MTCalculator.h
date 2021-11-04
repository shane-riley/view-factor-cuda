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

	// Constructor
	MTCalculator(Geometry &cpuEmitter, Geometry &cpureceiver, Geometry & cpublocker, int deviceNum, int emitterBegin, int emitterEnd);

	// Calculate method
	double calculateVF();

	// Wrapper methods for gpu kernels
};

#endif  /* MT_CALC_H_ */

