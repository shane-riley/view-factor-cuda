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

	// Constructor
	STLReader(string ffilename);

	// Get total number of facets
	int getNumFacets();

	// Get next Facet (normal vector and 3 vertices)
	// NOTE: Follows Aparapi format
	void getNextFacet(double* normal, double* vertices);
};

#endif  /* STL_READER_H_ */
