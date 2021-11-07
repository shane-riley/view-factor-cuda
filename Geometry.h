// Header Guards
#ifndef GEOMETRY_H_
#define GEOMETRY_H_

#include "globals.h"
#include "STLReader.h"

class Geometry
{

// I am making everything public for now--we can handle visibility later
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

	// Geometry constructor from STL reader
	Geometry(STLReader &reader);

	// Someday we will make a constructor that takes a TEGDefinition as an input

	// initWithSize method (allocates all arrays as size-length and full of zeroes)
	void initWithSize(int newSize);

	// freeMemory method (releases memory from allocation, will error if memory was never allocated in the first place)
	void freeMemory();

	// areaOf method (same as Aparapi)
	static double areaOf(vector<double3> triangle);

	// Size method
	int size();
};

#endif  /* GEOMETRY_H_ */