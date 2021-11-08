#include "STLReader.h"

std::istream& safeGetline(std::istream& is, std::string& t)
{
    t.clear();

    // The characters in the stream are read one-by-one using a std::streambuf.
    // That is faster than reading them one-by-one using the std::istream.
    // Code that uses streambuf this way must be guarded by a sentry object.
    // The sentry object performs various tasks,
    // such as thread synchronization and updating the stream state.

    std::istream::sentry se(is, true);
    std::streambuf* sb = is.rdbuf();

    for (;;)
    {
        int c = sb->sbumpc();
        switch (c)
        {
        case '\n':
            return is;
        case '\r':
            if (sb->sgetc() == '\n')
                sb->sbumpc();
            return is;
        case std::streambuf::traits_type::eof():
            // Also handle the case when the last line has no line ending
            if (t.empty())
                is.setstate(std::ios::eofbit);
            return is;
        default:
            t += (char)c;
        }
    }
}

STLReader::STLReader(string ffilename) {
    filename = ffilename;
}

void STLReader::openFile() {
    file.open(filename);
}

void STLReader::resetFile() {
    file.clear();
    file.seekg(0);
}

void STLReader::closeFile() {
    file.close();
}

int STLReader::getNumFacets() {
    
    string temp;

    // Reset file
    resetFile();
    
    // Skip header
    safeGetline(file, temp);

    // Count the number of instances of 'facet normal'
    int numFacets = 0;
    while (!file.eof()) {
        safeGetline(file, temp);
        if (temp.find("facet normal") != string::npos) {
            numFacets++;
        }
    }
    return numFacets;
}

vector<double3> STLReader::getNextFacet() {
	
    string line = "";
    vector<double3> result;

    // Look for next facet normal
    while (!file.eof()) {
        safeGetline(file, line);
        if (line.find("facet normal") != string::npos) {
            break;
        }
    }
    if (file.eof()) {
        return result;
    }

    // Line contains normal
    stringstream ss(line);
    string temp;
    double3 coord;
    
    // Place normal
    ss >> temp >> coord.x >> coord.y >> coord.z;
    result.push_back(coord);

    // Move forward two lines
    safeGetline(file, line);
    safeGetline(file, line);

    for (int i = 0; i < 3; i++) {
        ss >> temp >> coord.x >> coord.y >> coord.z;
        result.push_back(coord);
    }

    // Move forward two lines
    safeGetline(file, line);
    safeGetline(file, line);
    return result;
}
