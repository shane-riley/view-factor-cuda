#include "globals.h"
#include "STLReader.h"
#include "Geometry.h"
#include "MTCalculator.cuh"
#include <thread>
#include <iostream>
#include <vector>

void runMT(Geometry &emitterGeometry, Geometry &receiverGeometry, Geometry &blockerGeometry, int i, int start, int end, double *vfs) {
	MTCalculator MT(emitterGeometry,
		receiverGeometry,
		blockerGeometry,
		i,
		start,
		end);
	double vf = MT.calculateVF();
	MT.freeMemory();
	vfs[i] = vf;
}

int main(int argc, char** args)
{
	// Arguments
	// 1: Emitting Geometry
	// 2: Receiving Geometry
	// 3: Blocking Geometry (Optional)

	// Check arguments
	//if (argc < 4 || argc > 4) {
		// Wrong number of inputs
		//cout << "Usage: vfcuda [Emitter STL] [Receiving STL] (Blocking STL)" << endl;
		//exit(0);
	//}

	string emitterFilename = "F:/Shane/view-factor-cuda/test/bottom.stl"; //string(args[1]);
	string receiverFilename = "F:/Shane/view-factor-cuda/test/top.stl"; //string(args[2]);
	string blockerFilename = "F:/Shane/view-factor-cuda/test/none.stl"; //string(args[3]);

	// Specify GPU's

	// Create STL readers
	STLReader emitterReader(emitterFilename);
	STLReader receiverReader(receiverFilename);
	STLReader blockerReader(blockerFilename);

	// Create Geometries
	Geometry emitterGeometry(emitterReader);
	Geometry receiverGeometry(receiverReader);
	Geometry blockerGeometry(blockerReader);

	// Create an MTCalculator per device
	// TODO: Actually implement more than one GPU and have them get executed concurrently
	// For now, just do one 

	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	cout << "Number of available devices is: " << deviceCount << endl;


	vector<thread> threads;
	double* vfs = (double*)calloc(deviceCount, sizeof(double));

	int s = emitterGeometry.size();
	for (int i = 0; i < deviceCount; i++) {
		int start = (i * s / deviceCount);
		int end = ((i+1) * s / deviceCount);
		threads.push_back(thread(runMT, emitterGeometry, receiverGeometry, blockerGeometry, i, start, end, vfs));
	}
	for (auto& seg : threads) {
		seg.join();
	}
	double vf = 0.0;
	for (int i = 0; i < deviceCount; i++) {
		cout << "Device,VF: " << i << "," << vfs[i] << endl;
		vf += vfs[i];
	}

	cout << "VF: " << vf << endl;

}