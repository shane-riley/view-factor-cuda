#include "GPUGeometry.h"

GPUGeometry::GPUGeometry(Geometry& geom) {

	arraySize = geom.arraySize;
	// Implement the CUDA copy
	
	// Allocate the memory

	cudaMalloc((void**)&normalX, arraySize * sizeof(double));
	cudaMalloc((void**)&normalY, arraySize * sizeof(double));
	cudaMalloc((void**)&normalZ, arraySize * sizeof(double));

	cudaMalloc((void**)&vertexAX, arraySize * sizeof(double));
	cudaMalloc((void**)&vertexAY, arraySize * sizeof(double));
	cudaMalloc((void**)&vertexAZ, arraySize * sizeof(double));

	cudaMalloc((void**)&edgeBAX, arraySize * sizeof(double));
	cudaMalloc((void**)&edgeBAY, arraySize * sizeof(double));
	cudaMalloc((void**)&edgeBAZ, arraySize * sizeof(double));

	cudaMalloc((void**)&edgeCAX, arraySize * sizeof(double));
	cudaMalloc((void**)&edgeCAY, arraySize * sizeof(double));
	cudaMalloc((void**)&edgeCAZ, arraySize * sizeof(double));

	cudaMalloc((void**)&centerX, arraySize * sizeof(double));
	cudaMalloc((void**)&centerY, arraySize * sizeof(double));
	cudaMalloc((void**)&centerZ, arraySize * sizeof(double));

	cudaMalloc((void**)&area, arraySize * sizeof(double));

	// Copy the memory
	cudaMemcpy(normalX, geom.normalX, arraySize * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(normalY, geom.normalY, arraySize * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(normalZ, geom.normalZ, arraySize * sizeof(double), cudaMemcpyHostToDevice);

	cudaMemcpy(vertexAX, geom.normalX, arraySize * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(vertexAY, geom.normalY, arraySize * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(vertexAZ, geom.normalZ, arraySize * sizeof(double), cudaMemcpyHostToDevice);

	cudaMemcpy(edgeBAX, geom.normalX, arraySize * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(edgeBAY, geom.normalY, arraySize * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(edgeBAZ, geom.normalZ, arraySize * sizeof(double), cudaMemcpyHostToDevice);

	cudaMemcpy(edgeCAX, geom.normalX, arraySize * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(edgeCAY, geom.normalY, arraySize * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(edgeCAZ, geom.normalZ, arraySize * sizeof(double), cudaMemcpyHostToDevice);

	cudaMemcpy(centerX, geom.normalX, arraySize * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(centerY, geom.normalY, arraySize * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(centerZ, geom.normalZ, arraySize * sizeof(double), cudaMemcpyHostToDevice);

	cudaMemcpy(area, geom.normalZ, arraySize * sizeof(double), cudaMemcpyHostToDevice);
}

void GPUGeometry::freeMemory() {

	// FREE THE MEMORY
	cudaFree(normalX);
	cudaFree(normalY);
	cudaFree(normalZ);

	cudaFree(vertexAX);
	cudaFree(vertexAY);
	cudaFree(vertexAZ);

	cudaFree(edgeBAX);
	cudaFree(edgeBAY);
	cudaFree(edgeBAZ);

	cudaFree(edgeCAX);
	cudaFree(edgeCAY);
	cudaFree(edgeBAZ);

	cudaFree(centerX);
	cudaFree(centerY);
	cudaFree(centerZ);

	cudaFree(area);
}
