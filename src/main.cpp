#include "vfcuda.h"

int main(int argc, char** args)
{
	// Arguments
	// 1: Emitting Geometry
	// 2: Receiving Geometry
	// 3: Blocking Geometry (default none)
	// Defaults to not running self-intersection (controlled using SELF_INT)
	// = 0: OFF
	// = 1: ON (Emitters)
	// = 2: ON (Receivers)
	// = 3: ON (Both)
	// Uses all devices availible in CUDA (controlled using CUDA_VISIBLE_DEVICES)

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
}