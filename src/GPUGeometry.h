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
	float3 *normal;

	float3 *vertexA;

	float3 *edgeBA;

	float3 *edgeCA;

	float3 *center;

	double *area;

	double *result;

	// GPUGeometry constructor from regular Geometry
	GPUGeometry(Geometry &geom);

	GPUGeometry();

	// Free memory
	void freeMemory();
};

#endif /* GPUGEOMETRY_H_ */