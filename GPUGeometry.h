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
	double* normalX;
	double* normalY;
	double* normalZ;

	double* vertexAX;
	double* vertexAY;
	double* vertexAZ;

	double* edgeBAX;
	double* edgeBAY;
	double* edgeBAZ;

	double* edgeCAX;
	double* edgeCAY;
	double* edgeCAZ;

	double* centerX;
	double* centerY;
	double* centerZ;

	double* area;

	// GPUGeometry constructor from regular Geometry
	GPUGeometry(Geometry &geom);

	// Free memory
	void freeMemory();

};

#endif  /* GPUGEOMETRY_H_ */