#include "vfcuda.h"

void runMT(Geometry& emitterGeometry, Geometry& receiverGeometry, Geometry& blockerGeometry, int i, int start, int end, double* vfs) {
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

	cout << "VFCUDA" << endl;
	cout << endl;

	cout << "Input files:" << endl;
	cout << "Emitter: " << ef << endl;
	cout << "Receiver: " << rf << endl;
	if (do_blocker) cout << "Blocker: " << bf << endl;

	STLReader emitterReader(ef);
	STLReader receiverReader(rf);

	STLReader blockerReader;
	if (do_blocker) blockerReader = STLReader(bf);

	Geometry emitterGeometry;
	Geometry receiverGeometry;
	Geometry blockerGeometry;
	cout << "Reading files..." << endl;
	try {
		emitterGeometry = Geometry(emitterReader);
		receiverGeometry = Geometry(receiverReader);
		if (do_blocker) blockerGeometry = Geometry(blockerReader);
	}
	catch (runtime_error e) {
		cout << "ERROR! File not found: " << e.what() << endl;
		cout << "Terminating..." << endl;
		return -1;
	}
	cout << "Files loaded." << endl;

	int deviceCount;
	cudaGetDeviceCount(&deviceCount);

	cout << "------------------------------------------" << endl;
	cout << endl;
	cout << "Tesselation counts: " << endl;
	cout << "ET: " << emitterGeometry.arraySize << endl;
	cout << "RT: " << receiverGeometry.arraySize << endl;
	if (do_blocker) cout << "BT: " << blockerGeometry.arraySize << endl;

	// Calculate areas
	double eArea = emitterGeometry.totalArea();
	double rArea = receiverGeometry.totalArea();
	double bArea = (do_blocker) ? blockerGeometry.totalArea() : 0.0;

	cout << "Total areas: " << endl;
	cout << "ET: " << eArea << endl;
	cout << "RT: " << rArea << endl;
	if (do_blocker) cout << "BT: " << bArea << endl;
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
		double thisVF = vfs[i] / eArea;
		cout << "Thread " << i << ": " << thisVF << endl;
		vf += thisVF;
	}
	cout << "------------------------------------------" << endl;
	cout << endl;
	return vf;
}

double getVF(string ef, string rf) {
	string bf = "";
	return (getVF(ef, rf, bf));
}