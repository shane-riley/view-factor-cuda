#include "Geometry.h"

Geometry::Geometry(STLReader &reader) {
	// TODO: Implement me
	// Return NULL if NULL
	//initWithSize();
}

void Geometry::initWithSize(int newSize) {
	arraySize = newSize;
	normalX = (double*)calloc(arraySize, sizeof(double));
	normalY = (double*)calloc(arraySize, sizeof(double));
	normalZ = (double*)calloc(arraySize, sizeof(double));

	vertexAX = (double*)calloc(arraySize, sizeof(double));
	vertexAY = (double*)calloc(arraySize, sizeof(double));
	vertexAZ = (double*)calloc(arraySize, sizeof(double));

	edgeBAX = (double*)calloc(arraySize, sizeof(double));
	edgeBAY = (double*)calloc(arraySize, sizeof(double));
	edgeBAZ = (double*)calloc(arraySize, sizeof(double));

	edgeCAX = (double*)calloc(arraySize, sizeof(double));
	edgeCAY = (double*)calloc(arraySize, sizeof(double));
	edgeCAZ = (double*)calloc(arraySize, sizeof(double));

	centerX = (double*)calloc(arraySize, sizeof(double));
	centerY = (double*)calloc(arraySize, sizeof(double));
	centerZ = (double*)calloc(arraySize, sizeof(double));

	area = (double*)calloc(arraySize, sizeof(double));
}

void Geometry::freeMemory() {
	free(normalX);
	free(normalY);
	free(normalZ);

	free(vertexAX);
	free(vertexAY);
	free(vertexAZ);

	free(edgeBAX);
	free(edgeBAY);
	free(edgeBAZ);

	free(edgeCAX);
	free(edgeCAY);
	free(edgeCAZ);

	free(centerX);
	free(centerY);
	free(centerZ);

	free(area);
}

double Geometry::areaOf(double *triangle) {
	// TODO: Implement me
}

int Geometry::size() { return arraySize; }