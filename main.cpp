#include "globals.h"
#include "STLReader.h"
#include "Geometry.h"
#include "MTCalculator.h"
#include <thread>
#include <iostream>
#include <vector>



int main(int argc, char **args)
{
	// Arguments
	// 1: Emitting Geometry
	// 2: Receiving Geometry
	// 3: Blocking Geometry (Optional)
	
	// Check arguments
	if (argc < 3 || argc > 4) {
		// Wrong number of inputs
		cout << "Usage: vfcuda [Emitter STL] [Receiving STL] (Blocking STL)" << endl;
		exit(0);
	}

	string emitterFilename = string(args[1]);
	string receiverFilename = string(args[2]);
	string blockerFilename = "";
	bool is_blocker = false;

	if (argc == 4) {
		blockerFilename = string(args[3]);
		is_blocker = true;
	}

	// Specify GPU's
	// TODO: Make the GPU target an input
	vector<int> targetDevices = { 0 };
	int numDevices = targetDevices.size();


	// Create STL readers
	STLReader emitterReader(emitterFilename);
	STLReader receiverReader(emitterFilename);
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
	cout << "Number of available devices is: " << deviceCount;
	

	std::vector <std::thread> threads;

	for (int i = 0; i < deviceCount; i++) {
		std::thread seg();
		threads.push_back(std::move(seg));
	}
	for (auto& seg : threads) {
		seg.join();
	}
	void create()
	MTCalculator MT0(emitterGeometry,
		receiverGeometry,
		blockerGeometry,
		0,
		0,
		emitterGeometry.size());

	// Run the MTCalculator
	double vf = MT0.calculateVF();

	cout << "VF: " << vf << endl;

}