#include "vfcuda.h"

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