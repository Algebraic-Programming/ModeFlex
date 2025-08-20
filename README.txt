================================================================================
Compilation
================================================================================

Compile with 'make' after modifying the Makefile with the location of dependencies, specified below.

Dependencies:
	- GCC
	- Intel MKL
	- Libxsmm*, https://github.com/hfp/libxsmm

* Please specify the installation location at line 39 in the Makefile

Versions the code was tested with:
- gcc/6.1.0
- intel/mkl/intel64/2018.2.199
- openmpi/gcc/64/2.0.1
- libxsmm/master-1.9-864

================================================================================
Quick start
================================================================================
	
To run a set of algorithms for a preset tensor sizes of dimensionalities ranging from dim_min until dim_max call:
	./tensorlib_test (dim_min) (dim_max)
	
Preceed the call with the following flags (OMP_NUM_THREADS is used to specify the number of threads):
	OMP_DYNAMIC=false MKL_DYNAMIC=false OMP_NESTED=true OMP_PROC_BIND=close OMP_NUM_THREADS=15

Due to a high number of variables in the program, and the fact that high dimensionalities quickly saturate the machine's memory,
the user must specify block sizes, and the number of stripes, for each number of threads, and number of dimensions 
(integer factoring was not implemented):
- Thread count: 10, 15, 20, 60, 120
- Dimensions: 2, 3, 4, and 5

These may be extended manually in the test_openmp.c file.

The results are stored in data\master directory within the code's directory.

