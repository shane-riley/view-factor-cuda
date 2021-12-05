#include "GPUGeometry.h"

GPUGeometry::GPUGeometry() {}

GPUGeometry::GPUGeometry(Geometry &geom)
{

	arraySize = geom.arraySize;
	// Implement the CUDA copy

	// Allocate the memory

	checkCudaErrors(cudaMalloc((void **)&normal, arraySize * sizeof(float3)));

	checkCudaErrors(cudaMalloc((void **)&vertexA, arraySize * sizeof(float3)));

	checkCudaErrors(cudaMalloc((void **)&edgeBA, arraySize * sizeof(float3)));

	checkCudaErrors(cudaMalloc((void **)&edgeCA, arraySize * sizeof(float3)));

	checkCudaErrors(cudaMalloc((void **)&center, arraySize * sizeof(float3)));

	checkCudaErrors(cudaMalloc((void **)&area, arraySize * sizeof(double)));

	// Copy the memory
	checkCudaErrors(cudaMemcpy(normal, geom.normal, arraySize * sizeof(float3), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(vertexA, geom.vertexA, arraySize * sizeof(float3), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(edgeBA, geom.edgeBA, arraySize * sizeof(float3), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(edgeCA, geom.edgeCA, arraySize * sizeof(float3), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(center, geom.center, arraySize * sizeof(float3), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(area, geom.area, arraySize * sizeof(double), cudaMemcpyHostToDevice));
}

void GPUGeometry::freeMemory()
{

	// FREE THE MEMORY
	checkCudaErrors(cudaFree(normal));

	checkCudaErrors(cudaFree(vertexA));

	checkCudaErrors(cudaFree(edgeBA));

	checkCudaErrors(cudaFree(edgeCA));

	checkCudaErrors(cudaFree(center));

	checkCudaErrors(cudaFree(area));
}
