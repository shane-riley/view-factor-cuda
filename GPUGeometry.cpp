#include "GPUGeometry.h"

GPUGeometry::GPUGeometry() {}

GPUGeometry::GPUGeometry(Geometry& geom) {

	arraySize = geom.arraySize;
	// Implement the CUDA copy
	
	// Allocate the memory

	checkCudaErrors(cudaMalloc((void**)&normalX, arraySize * sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&normalY, arraySize * sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&normalZ, arraySize * sizeof(double)));

	checkCudaErrors(cudaMalloc((void**)&vertexAX, arraySize * sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&vertexAY, arraySize * sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&vertexAZ, arraySize * sizeof(double)));

	checkCudaErrors(cudaMalloc((void**)&edgeBAX, arraySize * sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&edgeBAY, arraySize * sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&edgeBAZ, arraySize * sizeof(double)));

	checkCudaErrors(cudaMalloc((void**)&edgeCAX, arraySize * sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&edgeCAY, arraySize * sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&edgeCAZ, arraySize * sizeof(double)));

	checkCudaErrors(cudaMalloc((void**)&centerX, arraySize * sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&centerY, arraySize * sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&centerZ, arraySize * sizeof(double)));

	checkCudaErrors(cudaMalloc((void**)&area, arraySize * sizeof(double)));

	// Copy the memory
	checkCudaErrors(cudaMemcpy(normalX, geom.normalX, arraySize * sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(normalY, geom.normalY, arraySize * sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(normalZ, geom.normalZ, arraySize * sizeof(double), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(vertexAX, geom.normalX, arraySize * sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(vertexAY, geom.normalY, arraySize * sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(vertexAZ, geom.normalZ, arraySize * sizeof(double), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(edgeBAX, geom.normalX, arraySize * sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(edgeBAY, geom.normalY, arraySize * sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(edgeBAZ, geom.normalZ, arraySize * sizeof(double), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(edgeCAX, geom.normalX, arraySize * sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(edgeCAY, geom.normalY, arraySize * sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(edgeCAZ, geom.normalZ, arraySize * sizeof(double), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(centerX, geom.normalX, arraySize * sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(centerY, geom.normalY, arraySize * sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(centerZ, geom.normalZ, arraySize * sizeof(double), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(area, geom.normalZ, arraySize * sizeof(double), cudaMemcpyHostToDevice));
}

void GPUGeometry::freeMemory() {

	// FREE THE MEMORY
	checkCudaErrors(cudaFree(normalX));
	checkCudaErrors(cudaFree(normalY));
	checkCudaErrors(cudaFree(normalZ));

	checkCudaErrors(cudaFree(vertexAX));
	checkCudaErrors(cudaFree(vertexAY));
	checkCudaErrors(cudaFree(vertexAZ));

	checkCudaErrors(cudaFree(edgeBAX));
	checkCudaErrors(cudaFree(edgeBAY));
	checkCudaErrors(cudaFree(edgeBAZ));

	checkCudaErrors(cudaFree(edgeCAX));
	checkCudaErrors(cudaFree(edgeCAY));
	checkCudaErrors(cudaFree(edgeBAZ));

	checkCudaErrors(cudaFree(centerX));
	checkCudaErrors(cudaFree(centerY));
	checkCudaErrors(cudaFree(centerZ));

	checkCudaErrors(cudaFree(area));
}
