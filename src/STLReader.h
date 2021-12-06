// Header Guards
#ifndef STL_READER_H_
#define STL_READER_H_

#include "globals.h"

class STLReader
{

public:

	// Filename of STL
	string filename;

	// File stream
	ifstream file;

	bool binaryMode;

	// Constructor
	STLReader(string ffilename, bool mode);

	// Void constructor
	STLReader();

	// Open file
	void openFile();

	// Reset file
	void resetFile();

	// Close file
	void closeFile();

	// Get total number of facets
	unsigned int getNumFacets();

	void getToFacets();

	// Get next Facet (normal vector and 3 vertices)
	// NOTE: Follows Aparapi format
	void getNextFacet(vector<float3> &coords);
};

#endif  /* STL_READER_H_ */
