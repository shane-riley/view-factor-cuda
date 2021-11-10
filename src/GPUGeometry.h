// Header Guards
#ifndef GPUGEOMETRY_H_
#define GPUGEOMETRY_H_

#include "globals.h"
#include "STLReader.h"
#include "Geometry.h"

class GPUGeometry
{

public:
	// Size
	int arraySize;

	// NOTE: It might be possible to use std::vectors for these arrays and then copy them over--later problem
	// size-length pointers
	double3 *normal;

	double3 *vertexA;

	double3 *edgeBA;

	double3 *edgeCA;

	double3 *center;

	double *area;

	double *result;

	// GPUGeometry constructor from regular Geometry
	GPUGeometry(Geometry &geom);

	GPUGeometry();

	// Free memory
	void freeMemory();
};

#endif /* GPUGEOMETRY_H_ */