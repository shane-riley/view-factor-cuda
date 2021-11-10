#include "Geometry.h"

Geometry::Geometry()
{
	arraySize = 0;
}

Geometry::Geometry(STLReader &reader)
{

	// Open file
	reader.openFile();
	if (!reader.file.is_open())
	{
		cout << "File not found: " << reader.filename << endl;
		cout << "Terminating..." << endl;
		exit(0);
	}

	int numFacets = reader.getNumFacets();
	initWithSize(numFacets);
	reader.resetFile();

	for (int i = 0; i < numFacets; i++)
	{
		vector<double3> info = reader.getNextFacet();
		normal[i].x = info[0].x;
		normal[i].y = info[0].y;
		normal[i].z = info[0].z;

		vertexA[i].x = info[1].x;
		vertexA[i].y = info[1].y;
		vertexA[i].z = info[1].z;

		edgeBA[i].x = info[2].x - vertexA[i].x;
		edgeBA[i].y = info[2].y - vertexA[i].y;
		edgeBA[i].z = info[2].z - vertexA[i].z;

		edgeCA[i].x = info[3].x - vertexA[i].x;
		edgeCA[i].y = info[3].y - vertexA[i].y;
		edgeCA[i].z = info[3].z - vertexA[i].z;

		center[i].x = (info[1].x + info[2].x + info[3].x) / 3;
		center[i].y = (info[1].y + info[2].y + info[3].y) / 3;
		center[i].z = (info[1].z + info[2].z + info[3].z) / 3;

		area[i] = areaOf(info);
	}
}

void Geometry::initWithSize(int newSize)
{
	arraySize = newSize;
	normal = (double3 *)calloc(arraySize, sizeof(double3));

	vertexA = (double3 *)calloc(arraySize, sizeof(double3));

	edgeBA = (double3 *)calloc(arraySize, sizeof(double3));

	edgeCA = (double3 *)calloc(arraySize, sizeof(double3));

	center = (double3 *)calloc(arraySize, sizeof(double3));

	area = (double *)calloc(arraySize, sizeof(double3));
}

void Geometry::freeMemory()
{
	free(normal);

	free(vertexA);

	free(edgeBA);

	free(edgeCA);

	free(center);

	free(area);
}

double Geometry::areaOf(vector<double3> triangle)
{

	double x1 = triangle[2].x - triangle[1].x;
	double x2 = triangle[2].y - triangle[1].y;
	double x3 = triangle[2].z - triangle[1].z;

	double y1 = triangle[3].x - triangle[1].x;
	double y2 = triangle[3].y - triangle[1].y;
	double y3 = triangle[3].z - triangle[1].z;

	return .5 * sqrt((x2 * y3 - x3 * y2) * (x2 * y3 - x3 * y2) + (x3 * y1 - x1 * y3) * (x3 * y1 - x1 * y3) + (x1 * y2 - x2 * y1) * (x1 * y2 - x2 * y1));
}

int Geometry::size() { return arraySize; }