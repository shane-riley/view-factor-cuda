#include "Geometry.h"

Geometry::Geometry()
{
	arraySize = 0;
}

Geometry::Geometry(STLReader &reader)
{

	// Check mode
	char* binaryStr;
	int binarySetting = 0;
	bool binaryMode = false;
	binaryStr = getenv(BINARY_ENV_VAR);
	if (binaryStr != NULL) {
		binarySetting = atoi(binaryStr);
	}
	if (binarySetting == 1) {
		binaryMode = true;
	}

	// Open file
	reader.openFile();
	if (!reader.file.is_open())
	{
		throw runtime_error(reader.filename);
	}

	unsigned int numFacets = reader.getNumFacets(binaryMode);
	initWithSize(numFacets);
	reader.resetFile();
	if (binaryMode) { reader.getToFacets(); }
	

	vector<float3> info(4);
	for (int i = 0; i < numFacets; i++)
	{
		reader.getNextFacet(binaryMode, info);
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

void Geometry::initWithSize(unsigned int newSize)
{
	arraySize = newSize;
	normal = (float3*)calloc(arraySize, sizeof(float3));

	vertexA = (float3*)calloc(arraySize, sizeof(float3));

	edgeBA = (float3*)calloc(arraySize, sizeof(float3));

	edgeCA = (float3*)calloc(arraySize, sizeof(float3));

	center = (float3*)calloc(arraySize, sizeof(float3));

	area = (double *)calloc(arraySize, sizeof(float3));
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

double Geometry::areaOf(vector<float3> triangle)
{

	double x1 = triangle[2].x - triangle[1].x;
	double x2 = triangle[2].y - triangle[1].y;
	double x3 = triangle[2].z - triangle[1].z;

	double y1 = triangle[3].x - triangle[1].x;
	double y2 = triangle[3].y - triangle[1].y;
	double y3 = triangle[3].z - triangle[1].z;

	return .5 * sqrt((x2 * y3 - x3 * y2) * (x2 * y3 - x3 * y2) + (x3 * y1 - x1 * y3) * (x3 * y1 - x1 * y3) + (x1 * y2 - x2 * y1) * (x1 * y2 - x2 * y1));
}

double Geometry::totalArea()
{
	double totalArea = 0.0;
	for (int i = 0; i < arraySize; i++) {
		totalArea += area[i];
	}
	return totalArea;
	
}

int Geometry::size() { return arraySize; }