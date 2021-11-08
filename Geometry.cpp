#include "Geometry.h"

Geometry::Geometry(STLReader &reader) {
	
	// Open file
	reader.openFile();

	initWithSize(reader.getNumFacets());
	reader.resetFile();

	for (int i = 0; i < reader.getNumFacets(); i++) {
		vector<double3> info = reader.getNextFacet();
		normalX[i] = info[0].x;
		normalY[i] = info[0].y;
		normalZ[i] = info[0].z;

		vertexAX[i] = info[1].x;
		vertexAY[i] = info[1].y;
		vertexAZ[i] = info[1].z;

		edgeBAX[i] = info[2].x - vertexAX[i];
		edgeBAY[i] = info[2].y - vertexAY[i];
		edgeBAZ[i] = info[2].z - vertexAZ[i];

		edgeCAX[i] = info[3].x - vertexAX[i];
		edgeCAY[i] = info[3].y - vertexAY[i];
		edgeCAZ[i] = info[3].z - vertexAZ[i];

		centerX[i] = (info[1].x + info[2].x + info[3].x) / 3;
		centerY[i] = (info[1].y + info[2].y + info[3].y) / 3;
		centerZ[i] = (info[1].z + info[2].z + info[3].z) / 3;
	
		area[i] = areaOf(info);
	}

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

double Geometry::areaOf(vector<double3> triangle ) {
	
	double x1 = triangle[1].x - triangle[0].x;
	double x2 = triangle[1].y - triangle[0].y;
	double x3 = triangle[1].z - triangle[0].z;

	double y1 = triangle[2].x - triangle[0].x;
	double y2 = triangle[2].y - triangle[0].y;
	double y3 = triangle[2].z - triangle[0].z;

	return .5 * sqrt((x2 * y3 - x3 * y2) * (x2 * y3 - x3 * y2) + (x3 * y1 - x1 * y3) * (x3 * y1 - x1 * y3) + (x1 * y2 - x2 * y1) * (x1 * y2 - x2 * y1));
}

int Geometry::size() { return arraySize; }