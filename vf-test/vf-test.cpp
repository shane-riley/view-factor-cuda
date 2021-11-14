#include "pch.h"
#include "CppUnitTest.h"
#include "../src/vfcuda.h"
#include <iostream>
#define TEST_CASE_DIRECTORY GetDirectoryName(__FILE__)

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace vftest
{
	TEST_CLASS(vftest)
	{
	public:

		string GetDirectoryName(string path) {
			const size_t last_slash_idx = path.rfind('\\');
			if (std::string::npos != last_slash_idx)
			{
				return path.substr(0, last_slash_idx + 1);
			}
			return "";
		}

		
		TEST_METHOD(ParallelPlates_CoarseSmall)
		{

			// Definition
			string folder = "/stl/ParallelPlates/CoarseSmall";
			string ef = (string) TEST_CASE_DIRECTORY + folder + "/Single_bottom_1024T.STL";
			string rf = (string) TEST_CASE_DIRECTORY + folder + "/Single_top_1024T.STL";
			double expectedVF = 0.003162057;
			double tolerance = 0.0000001;

			// Run VF
			// TODO: Add timeout
			double vf = getVF(ef, rf);

			Assert::AreEqual(expectedVF, vf, tolerance);
		}

		TEST_METHOD(Spheres_Coarse)
		{

			// Definition
			string folder = "/stl/Spheres/Coarse";
			string ef = (string)TEST_CASE_DIRECTORY + folder + "/outer_4478T.STL";
			string rf = (string)TEST_CASE_DIRECTORY + folder + "/inner_4470T.STL";
			double expectedVF = 1.00;
			double tolerance = 0.01;

			// Run VF
			// TODO: Add timeout
			double vf = getVF(ef, rf);

			Assert::AreEqual(expectedVF, vf, tolerance);
		}
	};
}
