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

	auto tStart = Time::now();

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
	double eArea, rArea, bArea;

	cout << "Reading files..." << endl;
	try {
		emitterGeometry = Geometry(emitterReader);
		receiverGeometry = Geometry(receiverReader);
		if (do_blocker) blockerGeometry = Geometry(blockerReader);
		eArea = emitterGeometry.totalArea();
		rArea = receiverGeometry.totalArea();
		bArea = (do_blocker) ? blockerGeometry.totalArea() : 0.0;
	}
	catch (runtime_error e) {
		cout << "ERROR! File not found: " << e.what() << endl;
		cout << "Terminating..." << endl;
		return -1;
	}
	cout << "Files loaded." << endl;
	msec tFiles = chrono::duration_cast<msec>(Time::now() - tStart);
	cout << "File import Time [s]: " << setprecision(3) << (float) tFiles.count() / 1000.0 << endl;

	int deviceCount;
	cudaGetDeviceCount(&deviceCount);

	cout << "------------------------------------------" << endl;
	cout << endl;
	cout << "Tesellations_ET: " << emitterGeometry.arraySize << endl;
	cout << "Area_ET: " << eArea << endl;
	cout << "Tesellations_RT: " << receiverGeometry.arraySize << endl;
	cout << "Area_RT: " << rArea << endl;
	if (do_blocker) cout << "Tesellations_BT: " << blockerGeometry.arraySize << endl;
	if (do_blocker) cout << "Area_BT: " << bArea << endl;

	cout << "------------------------------------------" << endl;
	cout << endl;

	// Determine selfint setting

	char* selfIntSettingStr;
	int selfIntSetting = 0;
	selfIntSettingStr = getenv(SELFINT_ENV_VAR);
	if (selfIntSettingStr != NULL) {
		selfIntSetting = atoi(selfIntSettingStr);
	}
	switch (selfIntSetting) {
		case 0:
			cout << "Self intersection DISABLED" << endl;
			break;
		case 1:
			cout << "Emmiter self intersection ENABLED" << endl;
			break;
		case 2:
			cout << "Receiver self intersection ENABLED" << endl;
			break;
		case 3:
			cout << "Emitter & Receiver self intersection ENABLED" << endl;
			break;
		break;
	}

	cout << "Number of available devices: " << deviceCount << endl;

	// Begin threads
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
	cout.precision(17);
	for (auto& seg : threads) {
		seg.join();
	}
	cout << "Complete!" << endl;
	cout << "------------------------------------------" << endl;
	cout << endl;
	double vf = 0.0;
	cout << "VF per thread:" << endl;
	for (int i = 0; i < deviceCount; i++) {
		double thisVF = vfs[i] / eArea;
		cout << "Thread " << i << ": " << setprecision(17) << thisVF << endl;
		vf += thisVF;
	}
	cout << "------------------------------------------" << endl;
	cout << endl;
	msec tEnd = chrono::duration_cast<msec>(Time::now() - tStart);
	cout << "Total Time [s]: " << setprecision(3) << (float)tEnd.count() / 1000.0 << endl;
	cout << "VF: " << setprecision(17) << vf << endl;
	return vf;
}

double getVF(string ef, string rf) {
	string bf = "";
	return (getVF(ef, rf, bf));
}