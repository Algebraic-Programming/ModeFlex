#include<string.h> // for memcmp
#include<assert.h>
#include<stdlib.h> // for free

#include<algorithms.h>
#include<gen_utils.h> // for reset_array_sizet
#include<gen_data.h> // get_vector, gen_vector, gen_block_tensor
#include<file_utils.h> // for save_to_file
#include<time_meas.h>

#include<bench_utils.h>

// This macro creates a proper filename for the results folder
#define FILENAME(x); snprintf(filename, BUFSIZE, "%s/%.0f_dimmin_%d_dimmax_%d_nmin_%d_nmax_%d_modemin_%d_modemax_%d_blockn_%d.csv", RESULTS_FOLDER, timespec_to_microseconds(time), dim_min, dim_max, n_min, n_max, mode_min, mode_max, block_n)
#define TEST(x); assert( memcmp(model_result->lin.data, x->lin.data, x->lin.size*sizeof(DTYPE))== equality )

int scenario1(int argc, char ** argv) {

	int dim_min, dim_max, n_min, n_max;
	int mode_min, mode_max;
	int block_n;

	// we must provide default arguments
	dim_min = 3;
	dim_max = 3;
	n_min = 3;
	n_max = 128;
	mode_min = 0;
	mode_max = -1;

	//int block_n_min = 1;
	//int block_n_max = n_max;

	// could be problematic...
	block_n = 8;

	// if an odd number:
	// -> the last element is the specific value for block_n
	// block_n = argv-1 (last element)
	if ((argc % 2) != 0) {
		//printf("block_n=%s\n", *(argv+argc--));
		// CONVERT string representation to integer
		sscanf (*(argv+argc--), "%d", &block_n);
		// we did -- to decrease used argument count (to say we used this el)
	}
	
	switch (argc) {
		case 6:
			// mode
			sscanf (*(argv+argc--), "%d", &mode_max);
			sscanf (*(argv+argc--), "%d", &mode_min);
		case 4:
			// dim, n 
			sscanf (*(argv+argc--), "%d", &n_max);
			sscanf (*(argv+argc--), "%d", &n_min);
			sscanf (*(argv+argc--), "%d", &dim_max);
			sscanf (*(argv+argc--), "%d", &dim_min);
	}

	if (mode_max == -1) {
		// default stayed, so we must finish the default max
		mode_max = dim_max-1;
	}

	printf("int dim_min=%d\n", dim_min);
	printf("int dim_max=%d\n", dim_max);
	printf("int n_min=%d\n", n_min);
	printf("int n_max=%d\n", n_max);
	printf("int mode_min=%d\n", mode_min);
	printf("int mode_max=%d\n", mode_max);
	printf("int block_n=%d\n", block_n);

	char filename[BUFSIZE];
	struct timespec time;
	clock_gettime(CLOCK_MONOTONIC, &time);
	FILENAME("results");

	printf("filename=%s\n", filename);
	FILE * file = fopen(filename, "a");
	if (file == NULL) {
		perror("Error opening file.\n");
	}
	//write_header(file);

	// improvement: could include the numbered versions for completeness
	// +1 FROM TESTS: we include the model algorithm here (tvm_tesor_major)
	const int algos = 1;
	TVM unfold_unfold_algorithms[1] = {
		tvm_output_major_BLAS_row
	};

	const int blocked_algos = 2;
	TVM block_block_algorithms[2] = {
		tvm_block_major_input_aligned_output_aligned_BLAS_POWERS,
		tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS
		//tvm_output_major_BLAS_row,
		//tvm_output_major_BLAS_row
		//tvm_block_major_input_aligned_output_aligned_BLAS_POWERS,
	};

	//double time = 0;

	//TVM block_unfold_algorithm = tvm_block_major_input_aligned;
	//TVM block_block_algorithm = tvm_block_major_input_aligned_output_aligned;
	//TVM morton_block_morton_block_algorithm = tvm_morton_block_major_input_aligned_output_aligned;
	//TVM morton_block_morton_block_algorithm_POWERS = tvm_morton_block_major_input_aligned_output_aligned_POWERS;
	//TVM morton_block_unfold_algorithm = tvm_morton_block_major_input_aligned;


	// parameters' loops ordered according to their dependency
	for (size_t dim=(size_t) dim_min; dim<=(size_t) dim_max; ++dim) {
		printf("dim=%zu:\n", dim);

		size_t block_layout[dim];
		size_t tensor_layout[dim];

		size_t temp_mode_max;
		if (dim-1 < (size_t) mode_max) {
			temp_mode_max = dim-1;
		} else {
			temp_mode_max = mode_max;
		}
		for (size_t mode=(size_t) mode_min; mode<=temp_mode_max; ++mode) {
			printf("    mode=%zu:\n", mode);

			for (size_t n=(size_t) n_min; n<=(size_t) n_max; n*=2) {			
				printf("        n=%zu:\n", n);

				// BEFORE: temp_block_n was meant to reduce the block size so its not more than n
				size_t temp_block_n;
				if ((size_t) block_n > n) {
					temp_block_n = n;
				} else {
					temp_block_n = block_n;
				}
				size_t block_size = 1;
				for (size_t d=0; d<dim; ++d) {
					block_size *= block_n;
				}

				// put n as each element of tensor_layout
				reset_array_sizet(tensor_layout, dim, n);	
				// or, for a change:
				//randomize_array_int(tensor_layout, dim, n);
				//printf("        n = ");
				//print_to_console_sizet(tensor_layout, dim);

				//for (size_t block_n=1; block_n<=(n+1/2); ++block_n) {
				//for (size_t block_n=block_n_min; block_n<=block_n_max && block_n<=n; block_n*=2) {

				printf("            block_n=%d:\n", block_n);

				// put block_n as each element of block_layout
				reset_array_sizet(block_layout, dim, temp_block_n);
				// allocate tensor,vector,result on the stack
				// allocate tensor,vector,result on the stack
				struct tensor_storage *tensor = gen_block_tensor(dim, tensor_layout, block_layout);
				// allocate tensor,vector,result on the stack
				struct lin_storage *vector = gen_vector(tensor->layout[mode]);
				// allocate tensor,vector,result on the stack
				struct tensor_storage  *result = get_block_result_tensor(tensor, mode);
				// allocate tensor,vector,result on the stack

				//const long FLOPS = tensor->lin.size*2;
				//printf("Number of FLOPS: %ld\n", FLOPS);

				// run all algorithms in a loop
				for (int algo=0; algo<algos; ++algo) {
					reset_array(result->lin.data, result->lin.size, 0);
					measure(
						unfold_unfold_algorithms[algo], tensor, vector, &result->lin, mode,
						file, n, temp_block_n, block_size);
				}

				fflush(file);

				// run all blocked algorithms in a loop
				for (int algo=0; algo<blocked_algos; ++algo) {
					reset_array(result->lin.data, result->lin.size, 0);
					measure(
						block_block_algorithms[algo], tensor, vector, &result->lin, mode,
						file, n, temp_block_n, block_size);
				}

				fflush(file);

				/////////////// 
				free_tensor_storage(result);
				free_tensor_storage(tensor);
				free_lin_storage(vector);



#if 0
				struct timespec start;
				struct timespec stop;
				reset_array(result->lin.data, result->lin.size, 0);
				double average_time = 0;
				for (int i=0; i<TIMES; ++i) {
					sleep(1);
					clock_gettime(CLOCK_MONOTONIC, &start);
					tvm_tensor_major_m(tensor, vector, &result->lin, mode, 1);
					clock_gettime(CLOCK_MONOTONIC, &stop);
					average_time += timespec_to_microseconds(timespec_diff(start, stop)); 
					printf("%lf \n", timespec_to_microseconds(timespec_diff(start,stop)));
				}
				average_time /= TIMES;
				printf("averaged normalized total_time: %f microsec\n", average_time);
				write_perf_result(file, time, dim, mode, n, temp_block_n, 99);
#endif
#if 0
				int out_algo = algos;
				// NOW, let's add the LAPACK to the testing
				struct tensor_storage_double *lapack_tensor = get_unfold(tensor, mode);
				struct lin_storage_double *lapack_vector = gen_vector_double(tensor->layout[mode]);
				struct tensor_storage_double *lapack_result = get_double_tensor(tensor, mode);
				
				time = measure(tvm_lapack_dgemv_input_aligned, lapack_tensor, lapack_vector, &lapack_result->lin, mode);
				printf("performance: FLOPS/time(ns) = %d/%lf = %lf\n",
						FLOPS,
						time*1000,
						(FLOPS/(time*1000)));

				write_perf_result(file, time, dim, mode, n, temp_block_n, out_algo);

				// sanity check that we are calculating something (!)
				// should be the same as the output major input aligned version
				//print_to_console_double(lapack_result->lin.data, lapack_result->lin.size);

				free_tensor_storage_double(lapack_tensor);
				free_tensor_storage_double(lapack_result);
				free_lin_storage_double(lapack_vector);
#endif
#if 0
				/////////////// block storage algorithms

				// SETUP
				int out_algo = algos;
				//struct tensor_storage *blocked_tensor = get_block_tensor(tensor, 0, 0); // block - normal
				// we don't need tensor anymore!
				// free_tensor_storage(tensor);

				// CASE 2: block -> block algorithm
				// run on blocked tensor 
				reset_array(result->lin.data, result->lin.size, 0);
				time = measure(block_block_algorithm, tensor, vector, &result->lin, mode);
				write_perf_result(file, time, dim, mode, n, temp_block_n, out_algo);
				// DESTROY
				// free_tensor_storage(blocked_tensor);

				// SETUP
				++out_algo;
				// struct tensor_storage *morton_blocked_tensor = get_block_tensor(tensor, 0, 1); // block - morton
				// CASE 2: morton_block -> morton_block
				// run on morton blocked tensor
				reset_array(result->lin.data, result->lin.size, 0);
				time = measure(morton_block_morton_block_algorithm, tensor, vector, &result->lin, mode);
				write_perf_result(file, time, dim, mode, n, temp_block_n, out_algo);

				// POWERS algorithm
				++out_algo;
				// run on morton blocked tensor
				reset_array(result->lin.data, result->lin.size, 0);
				time = measure(morton_block_morton_block_algorithm_POWERS, tensor, vector, &result->lin, mode);
				write_perf_result(file, time, dim, mode, n, temp_block_n, out_algo);
				// DESTROY
				// free_tensor_storage(morton_blocked_tensor);

#endif

			}
			
		}
	}



#if 0

	// parameters' loops ordered according to their dependency
	for (int dim=dim_min; dim<=dim_max; ++dim) {
		printf("dim=%d:\n", dim);

		int block_layout[dim];
		int tensor_layout[dim];
		
		int temp_mode_max;
		if (dim-1 < mode_max) {
			printf("set to mode_dim-1\n");
			temp_mode_max = dim-1;
		} else {
			temp_mode_max = mode_max;
		}
		
		for (int mode=mode_min; mode<=temp_mode_max; ++mode) {
			printf("    mode=%d:\n", mode);

			for (int n=n_min; n<=n_max; n*=2) {			

					printf("        n=%d:\n", n);
					// put n as each element of tensor_layout
					reset_array_sizet(tensor_layout, dim, n);

					int temp_block_n;
					if (block_n > n) {
						temp_block_n = n;
					} else {
						temp_block_n = block_n;
					}

					printf("            block_n=%d:\n", temp_block_n);
					// put block_n as each element of block_layout
					reset_array_sizet(block_layout, dim, temp_block_n);

					// allocate tensor,vector,result on the stack
					struct tensor_storage *tensor = gen_block_tensor(dim, tensor_layout, block_layout);
					// allocate tensor,vector,result on the stack
					struct lin_storage *vector = gen_vector(tensor->layout[mode]);
					// allocate tensor,vector,result on the stack
					struct tensor_storage  *result = get_block_result_tensor(tensor, mode);
					// allocate tensor,vector,result on the stack

					//const long FLOPS = tensor->lin.size*2;
					//printf("Number of FLOPS: %ld\n", FLOPS);

					// run all algorithms in a loop
					for (int algo=0; algo<algos; ++algo) {
						reset_array(result->lin.data, result->lin.size, 0);
						measure(
							unfold_unfold_algorithms[algo], tensor, vector, &result->lin, mode,
							file, n, temp_block_n);
					}

					fflush(file);

					// run all blocked algorithms in a loop
					for (int algo=0; algo<blocked_algos; ++algo) {
						reset_array(result->lin.data, result->lin.size, 0);
						measure(
							block_block_algorithms[algo], tensor, vector, &result->lin, mode,
							file, n, temp_block_n);
					}

					fflush(file);

#if 0
					struct timespec start;
					struct timespec stop;
					reset_array(result->lin.data, result->lin.size, 0);
					double average_time = 0;
					for (int i=0; i<TIMES; ++i) {
						sleep(1);
						clock_gettime(CLOCK_MONOTONIC, &start);
						tvm_tensor_major_m(tensor, vector, &result->lin, mode, 1);
						clock_gettime(CLOCK_MONOTONIC, &stop);
						average_time += timespec_to_microseconds(timespec_diff(start, stop)); 
						printf("%lf \n", timespec_to_microseconds(timespec_diff(start,stop)));
					}
					average_time /= TIMES;
					printf("averaged normalized total_time: %f microsec\n", average_time);
					write_perf_result(file, time, dim, mode, n, temp_block_n, 99);
#endif
#if 0
					int out_algo = algos;
					// NOW, let's add the LAPACK to the testing
					struct tensor_storage_double *lapack_tensor = get_unfold(tensor, mode);
					struct lin_storage_double *lapack_vector = gen_vector_double(tensor->layout[mode]);
					struct tensor_storage_double *lapack_result = get_double_tensor(tensor, mode);
					
					time = measure(tvm_lapack_dgemv_input_aligned, lapack_tensor, lapack_vector, &lapack_result->lin, mode);
					printf("performance: FLOPS/time(ns) = %d/%lf = %lf\n",
							FLOPS,
							time*1000,
							(FLOPS/(time*1000)));

					write_perf_result(file, time, dim, mode, n, temp_block_n, out_algo);

					// sanity check that we are calculating something (!)
					// should be the same as the output major input aligned version
					//print_to_console_double(lapack_result->lin.data, lapack_result->lin.size);

					free_tensor_storage_double(lapack_tensor);
					free_tensor_storage_double(lapack_result);
					free_lin_storage_double(lapack_vector);
#endif
#if 0
					/////////////// block storage algorithms

					// SETUP
					int out_algo = algos;
					//struct tensor_storage *blocked_tensor = get_block_tensor(tensor, 0, 0); // block - normal
					// we don't need tensor anymore!
					// free_tensor_storage(tensor);

					// CASE 2: block -> block algorithm
					// run on blocked tensor 
					reset_array(result->lin.data, result->lin.size, 0);
					time = measure(block_block_algorithm, tensor, vector, &result->lin, mode);
					write_perf_result(file, time, dim, mode, n, temp_block_n, out_algo);
					// DESTROY
					// free_tensor_storage(blocked_tensor);

					// SETUP
					++out_algo;
					// struct tensor_storage *morton_blocked_tensor = get_block_tensor(tensor, 0, 1); // block - morton
					// CASE 2: morton_block -> morton_block
					// run on morton blocked tensor
					reset_array(result->lin.data, result->lin.size, 0);
					time = measure(morton_block_morton_block_algorithm, tensor, vector, &result->lin, mode);
					write_perf_result(file, time, dim, mode, n, temp_block_n, out_algo);

					// POWERS algorithm
					++out_algo;
					// run on morton blocked tensor
					reset_array(result->lin.data, result->lin.size, 0);
					time = measure(morton_block_morton_block_algorithm_POWERS, tensor, vector, &result->lin, mode);
					write_perf_result(file, time, dim, mode, n, temp_block_n, out_algo);
					// DESTROY
					// free_tensor_storage(morton_blocked_tensor);

#endif
					/////////////// 
					free_tensor_storage(tensor);
					free_tensor_storage(result);
					free_lin_storage(vector);
			}
		}
	}
#endif

	if (file != NULL) {
		fclose(file);
	}
	return 0;
}

