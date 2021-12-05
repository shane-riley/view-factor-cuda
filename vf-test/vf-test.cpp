#include "pch.h"
#include "CppUnitTest.h"
#include "../src/vfcuda.h"
#include <iostream>
#include <stdlib.h>
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
			_putenv_s(SELFINT_ENV_VAR, "0");
			_putenv_s(BINARY_ENV_VAR, "0");
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

		TEST_METHOD(ParallelPlates_CoarseSmall_Binary)
		{
			_putenv_s(SELFINT_ENV_VAR, "0");
			_putenv_s(BINARY_ENV_VAR, "1");
			// Definition
			string folder = "/stl/ParallelPlates/CoarseSmall";
			string ef = (string)TEST_CASE_DIRECTORY + folder + "/Single_bottom_1024TB.STL";
			string rf = (string)TEST_CASE_DIRECTORY + folder + "/Single_top_1024TB.STL";
			double expectedVF = 0.003162057;
			double tolerance = 0.0000001;

			// Run VF
			// TODO: Add timeout
			double vf = getVF(ef, rf);

			Assert::AreEqual(expectedVF, vf, tolerance);
		}

		TEST_METHOD(ParallelPlates_Blocked)
		{
			_putenv_s(SELFINT_ENV_VAR, "0");
			_putenv_s(BINARY_ENV_VAR, "0");
			// Definition
			string folder = "/stl/ParallelPlates/CoarseSmall";
			string ef = (string)TEST_CASE_DIRECTORY + folder + "/Single_bottom_1024T.STL";
			string bf = (string)TEST_CASE_DIRECTORY + folder + "/Single_middle_2T.STL";
			string rf = (string)TEST_CASE_DIRECTORY + folder + "/Single_top_1024T.STL";
			double expectedVF = 0.0;
			double tolerance = 0.00001;

			// Run VF
			// TODO: Add timeout
			double vf = getVF(ef, rf, bf);

			Assert::AreEqual(expectedVF, vf, tolerance);
		}

		TEST_METHOD(ParallelPlates_Blocked2)
		{
			_putenv_s(SELFINT_ENV_VAR, "0");
			_putenv_s(BINARY_ENV_VAR, "0");
			// Definition
			string folder = "/stl/ParallelPlates/CoarseSmall";
			string ef = (string)TEST_CASE_DIRECTORY + folder + "/Single_bottom_1024T.STL";
			string bf = (string)TEST_CASE_DIRECTORY + folder + "/Single_middle2_2T.STL";
			string rf = (string)TEST_CASE_DIRECTORY + folder + "/Single_top_1024T.STL";
			double expectedVF = 0.0;
			double tolerance = 0.00001;

			// Run VF
			// TODO: Add timeout
			double vf = getVF(ef, rf, bf);

			Assert::AreEqual(expectedVF, vf, tolerance);
		}

		TEST_METHOD(Spheres_Coarse)
		{
			_putenv_s(SELFINT_ENV_VAR, "1");
			_putenv_s(BINARY_ENV_VAR, "0");
			// Definition
			string folder = "/stl/Spheres/Coarse";
			string ef = (string)TEST_CASE_DIRECTORY + folder + "/inner_4470T.STL";
			string rf = (string)TEST_CASE_DIRECTORY + folder + "/outer_4478T.STL";
			double expectedVF = 1.00;
			double tolerance = 0.01;

			// Run VF
			// TODO: Add timeout
			double vf = getVF(ef, rf);

			Assert::AreEqual(expectedVF, vf, tolerance);
		}
	};
}
