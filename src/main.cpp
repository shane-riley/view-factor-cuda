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

double getVF(string ef, string rf, string bf) {
	bool do_blocker = (bf.size() > 0);

	STLReader emitterReader(ef);
	STLReader receiverReader(rf);

	STLReader blockerReader;
	if (do_blocker) blockerReader = STLReader(bf);

	Geometry emitterGeometry(emitterReader);
	Geometry receiverGeometry(receiverReader);
	Geometry blockerGeometry;
	if (do_blocker) blockerGeometry = Geometry(blockerReader);

	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	
	cout << "VFCUDA" << endl;
	cout << endl;

	cout << "Input files:" << endl;
	cout << "Emitter: " << ef << endl;
	cout << "Receiver: " << rf << endl;
	if (do_blocker) cout << "Blocker: " << bf << endl;

	cout << "------------------------------------------" << endl;
	cout << endl;
	cout << "Tesselation counts: " << endl;
	cout << "ET: " << emitterGeometry.arraySize << endl;
	cout << "RT: " << receiverGeometry.arraySize << endl;
	if (do_blocker) cout << "BT: " << blockerGeometry.arraySize << endl;

	cout << "------------------------------------------" << endl;
	cout << endl;

	cout << "Number of available devices: " << deviceCount << endl;

	vector<thread> threads;
	double* vfs = (double*)calloc(deviceCount, sizeof(double));

	int s = emitterGeometry.size();
	for (int i = 0; i < deviceCount; i++) {
		int start = (i * s / deviceCount);
		int end = ((i + 1) * s / deviceCount);
		cout << "Thread " << i << ": (" << start << ", " << end << ")" << endl;
		threads.push_back(thread(runMT, emitterGeometry, receiverGeometry, blockerGeometry, i, start, end, vfs));
	}
	cout << "------------------------------------------" << endl;
	cout << endl;
	cout << "Threads started..." << endl;
	for (auto& seg : threads) {
		seg.join();
	}
	cout << "Complete!" << endl;
	cout << "------------------------------------------" << endl;
	cout << endl;
	double vf = 0.0;
	for (int i = 0; i < deviceCount; i++) {
		cout << "VF per thread:" << endl;
		cout << "Thread " << i << ": " << vfs[i] << endl;
		vf += vfs[i];
	}
	cout << "------------------------------------------" << endl;
	cout << endl;
	return vf;
}

double getVF(string ef, string rf) {
	string bf = "";
	return (getVF(ef, rf, bf));
}


int main(int argc, char** args)
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

	double vf;
	if (argc == 4) {
		string blockerFilename = string(args[3]);
		
		vf = getVF(emitterFilename, receiverFilename, blockerFilename);
	}
	else {
		vf = getVF(emitterFilename, receiverFilename);
	}
	
	cout << "VF: " << vf << endl;

}