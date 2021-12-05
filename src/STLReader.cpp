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

STLReader::STLReader() {}

void STLReader::openFile() {
    file.open(filename, ios::in);
}

void STLReader::resetFile() {
    file.clear();
    file.seekg(0);
}

void STLReader::closeFile() {
    file.close();
}

unsigned int STLReader::getNumFacets() {
    return getNumFacets(false);
}

unsigned int STLReader::getNumFacets(bool binaryMode) {
    
    if (binaryMode) {

        // Reset file
        resetFile();

        char buffer[100];

        unsigned int numFacets;
        // Skip 80 bytes
        file.read(buffer, 80);

        // Read int
        file.read(reinterpret_cast<char *>(&numFacets), 4);
        return numFacets;
    }
    else {
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
}

void STLReader::getToFacets() {
    char buffer[100];
    // Skip 84 bytes
    file.read(buffer, 84);
}

void STLReader::getNextFacet(float3 &coords) {
    getNextFacet(coords);
}

void STLReader::getNextFacet(bool binaryMode, vector<float3> &coords) {
	
    if (binaryMode) {
        for (int j = 0; j < 4; j++) {
            file.read(reinterpret_cast<char*>(&coords[j].x), 4);
            file.read(reinterpret_cast<char*>(&coords[j].y), 4);
            file.read(reinterpret_cast<char*>(&coords[j].z), 4);
        }
        // Skip two bytes
        char buffer[3];
        file.read(buffer, 2);
    }
    else {

        string line = "";

        // Look for next facet normal
        while (!file.eof()) {
            safeGetline(file, line);
            if (line.find("facet normal") != string::npos) {
                break;
            }
        }
        if (file.eof()) {
            return;
        }

        // Line contains normal
        stringstream ss(line);
        string temp = "";

        // Place normal
        ss >> temp >> temp >> coords[0].x >> coords[0].y >> coords[0].z;

        // Move forward two lines
        safeGetline(file, line);

        for (int i = 1; i < 4; i++) {
            safeGetline(file, line);
            ss = stringstream(line);
            ss >> temp >> coords[i].x >> coords[i].y >> coords[i].z;
        }

        // Move forward two lines
        safeGetline(file, line);
        safeGetline(file, line);
        return;
    }
}
