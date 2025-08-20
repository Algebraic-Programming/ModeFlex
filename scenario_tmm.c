// HT requirements
#define _GNU_SOURCE

#include <string.h> // for memcmp
#include <assert.h>
#include <stdlib.h> // for free
#include <math.h> // for pow, round

#include <algorithms.h>
#include <gen_utils.h> // for reset_array_sizet
#include <gen_data.h> // get_vector, gen_vector, gen_block_tensor
#include <file_utils.h> // for save_to_file
#include <time_meas.h>
#include <bench_utils.h>

// HT requirements
#include <tensorlibthreads.h>
#include <pthread.h>
#include <unistd.h> // for _SC_NPROCESSORS_ONLN

#define L1 4096.0 // 32 KB
#define L1_measured 2048.0
#define L2 32768.0 // 256 KB
#define L2_measured 16384.0
#define L3 3276800.0 
#define L3_measured 2097152.0 // 16 MB of L3 taken for the measurement (not half...)
#define RAM 4294967296.0 // not the actual max of RAM, just 32 GB (double RAM_measured)
#define RAM_measured 2684354560.0 // 20 GB

// This macro creates a proper filename for the results folder
#define FILENAME(x); snprintf(filename, BUFSIZE, "%s/%s_%.0f_dimmin_%d_dimmax_%d_nmin_%d_nmax_%d_modemin_%d_modemax_%d_blockn_%d.csv", RESULTS_FOLDER, hostname, timespec_to_microseconds(time), dim_min, dim_max, n_min, n_max, mode_min, mode_max, block_n)
#define TEST(x); assert( memcmp(model_result->lin.data, x->lin.data, x->lin.size*sizeof(DTYPE))== equality )

int scenario_tmm(int argc, char ** argv) {

	size_t EXP_MIN, EXP_MAX;

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
	block_n = 8;

	if ((argc % 2) != 0) {
		sscanf (*(argv+argc--), "%d", &block_n);
	}
	
	switch (argc) {
		case 6:
			// mode
			sscanf (*(argv+argc--), "%d", &mode_max);
			sscanf (*(argv+argc--), "%d", &mode_min);
			printf("int mode_min=%d\n", mode_min);
		case 4:
			// dim, n 
			sscanf (*(argv+argc--), "%d", &n_max);
			sscanf (*(argv+argc--), "%d", &n_min);
			sscanf (*(argv+argc--), "%d", &dim_max);
			sscanf (*(argv+argc--), "%d", &dim_min);
			printf("int n_min=%d\n", n_min);
			printf("int n_max=%d\n", n_max);
			printf("int dim_min=%d\n", dim_min);
			printf("int dim_max=%d\n", dim_max);
		case 2:
			sscanf (*(argv+argc--), "%d", &dim_max);
			sscanf (*(argv+argc--), "%d", &dim_min);
			printf("int dim_min=%d\n", dim_min);
			printf("int dim_max=%d\n", dim_max);
	}

	if (mode_max == -1) {
		mode_max = dim_max-1;
	}

	printf("int mode_max=%d\n", mode_max);
	printf("int block_n=%d\n", block_n);

	char hostname[1024];
	gethostname(hostname, 1024);
	char filename[BUFSIZE];
	struct timespec time;
	clock_gettime(CLOCK_MONOTONIC, &time);
	FILENAME("results");

	printf("filename=%s\n", filename);
	FILE * file = fopen(filename, "w"); // r+ if you want to update without deleteing it
	if (file == NULL) {
		perror("Error opening file.\n");
	}

	write_tmm_header(file);

	const int algo_to_test = 4;
	TVM algorithms_tmm[] = {
		tmm_blocked_mkl,
		tmm_looped_mkl,
		tmm_mortonblocked_mkl,
		tmm_blocked_libx
	};
	
	size_t memory_avail_tensor;
	
	if (TEST_ENV) {
		memory_avail_tensor = L3;
	} else {
		memory_avail_tensor = RAM_measured;
	}
	// Assume 1D(!)
	// gen_block_tensor_clean - copies the content instead of using the pointer(!) of max_length
	struct tensor_storage *tensor = gen_block_tensor_clean(1, &memory_avail_tensor, &memory_avail_tensor);
	struct tensor_storage  *result = gen_block_tensor_clean(1, &memory_avail_tensor, &memory_avail_tensor);
	printf("size of result is %zu while max tensro size per dim is %zu\n", result->lin.size, memory_avail_tensor);
	struct lin_storage *matrix = gen_vector(memory_avail_tensor);

	// parameters' loops ordered according to their dependency
	for(size_t dim=(size_t) 2; dim<=(size_t) 10; ++dim) {

		printf("dim=%zu:\n", dim);

		size_t block_layout[dim];
		size_t tensor_layout[dim];

		result->dim = dim;
		tensor->dim = dim;

		size_t block_array[3];
		size_t block_array_size = 3;

		if (dim == 2) {
			block_array[0] =    44;
			block_array[1] =   124;
			block_array[2] =   572;//1276;
		} else if (dim == 3) {
			block_array[0] =   12;
			block_array[1] =   24;
			block_array[2] =   68;//116;
		} else if (dim == 4) {
			block_array[0] =   6; 
			block_array[1] =  10;
			block_array[2] =  34;//24;
		} else if (dim == 5) {
			block_array[0] =  4;
			block_array[1] =  6;
			block_array[2] = 16;//12;
		} else if (dim == 6) {
			block_array[0] =   3;
			block_array[1] =   4;
			block_array[2] =   10;//8;
		} else if (dim == 7) {
			block_array[0] =   2;
			block_array[1] =   4;
			block_array[2] =   7;//6;
		} else if (dim == 8) {
			block_array[0] =   2;
			block_array[1] =   3;
			block_array[2]  =  5;
		} else if (dim == 9) {
			block_array[0] =   2;
			block_array[1] =   3;
			block_array[2]  =  4;
		} else if (dim == 10) {
			block_array[0] =  2;
			block_array[1]  = 4;
			block_array[2]  = 0; // will be ignored(!)
		}

		printf("Block sizes (different):\n");
		print_to_console_sizet(block_array, block_array_size);
		printf("Original result size is %zu\n", result->lin.size);

		// size_t n = floor(pow(memory_avail_tensor, 1/(double)dim))-1;
		// EXP_MIN = floor(pow(RAM_measured, 1/(double)dim))-1;
		// EXP_MAX = ceil(pow(RAM, 1/(double)dim));
		size_t per_dim_size = ceil(pow(memory_avail_tensor, 1/(double)dim));

		for(size_t i=0; i < block_array_size; ++i) {

			size_t block_n = block_array[i];
			// if (dim == 10) {
			// 	block_n = block_array[1];
			// } else {
			// 	block_n = block_array[2];
			// }
			if (block_n == 0) {
				continue;
			}
			printf("        	block_n=%zu:\n", block_n);
			reset_array_sizet(block_layout, dim, block_n);

			// Initialize tenosr, result, matrix (!)
			// tensor result of the same size (rule l_dim cannot be larger than n-K otherwise w ehave a problem!)
			// matrix is just of some smaller size but still very large
	
			int temp = per_dim_size/block_n;
			assert(temp!=0);

			size_t current_per_dim_size = temp*block_n;
			while (temp>1 && pow(current_per_dim_size,dim)>=memory_avail_tensor) {
				temp--;
				current_per_dim_size = current_per_dim_size - block_n;
				printf("decreasing the temp size to match available tensor memory.\n", temp);
			}
			assert(pow(current_per_dim_size,dim)<=memory_avail_tensor);

			reset_array_sizet(tensor_layout, dim, current_per_dim_size);
			printf("        n=%zu:\n", current_per_dim_size);

			free(tensor->layout);
			free(tensor->layout_perm);
			set_tensor_layout(tensor, tensor_layout);
			printf("Tenor layout:\n");
			print_to_console_sizet(tensor->layout, dim);
			// printf("Fixed tensor_size: %d\n", tensor_layout[0]);
	
			// Update block_layout + result block _ layout reset back to normal(!!)
			// for now just assume result is the same (in terms of block layout!)
			free(tensor->block_layout);
			tensor->block_layout = copy_array_int(block_layout, tensor->dim);

			for(size_t mode=0; mode<=dim-1; ++mode) {
				printf("\nmode %zu\n", mode);

				// go through variety of sizes, but not too many, also note this in the database?

				// FIRST CONDITION: WE NEVER WANT RESULT LARGER THAN INPUT TENSOR(!) hence the condition
				for (int l_dim_length=24; ((l_dim_length<=tensor->layout[mode]) && (l_dim_length<=48)); l_dim_length+=12) {

					int l_block_length = 12;
					// Find optimal size of block for l_dim
					// for (l_block_length=(l_dim_length)/2; l_block_length>0; l_block_length++) {
					// 	if (l_dim_length % l_block_length == 0) {
					// 		break;
					// 	}
					// }
					// if (l_block_length == 1) {
					// 	l_block_length = l_dim_length;
					// }
					printf("\nL = %zu\n", l_dim_length);
					printf("=========================== Optimal block chosen for this l_dim_length is %zu\n", l_block_length);

					matrix->size = tensor->layout[mode] * l_dim_length;

					free(result->layout);
					free(result->block_layout);
					result->layout = copy_array_int_replace_mode(tensor->layout, result->dim, mode, l_dim_length);
					result->block_layout = copy_array_int_replace_mode(tensor->block_layout, result->dim, mode, l_block_length);
					// Update the mode dimension in the result
					result->lin.size = get_size(result->layout, result->dim);

					printf("Result (size %zu) layout:\n", result->lin.size);
					print_to_console_sizet(result->layout, dim);
					printf("Result block layout:\n");
					print_to_console_sizet(result->block_layout, dim);
					
					// one algorith is independent of blocking(!)
					for (int algo=0; algo<algo_to_test; ++algo) {
						if ((algorithms_tmm[algo] == tmm_looped_mkl) && i!=0) {
							continue;
						}
						printf("RUN ALGORITHM %d\n", algo);
						reset_array(result->lin.data, result->lin.size, 0);
						measure_tmm(
							algorithms_tmm[algo], tensor, matrix, &result->lin, mode,
							file, tensor_layout[0], block_n, block_n, l_dim_length, l_block_length);
						fflush(file);
					}
				}
			}
		}
	}

	free_tensor_storage(tensor);
	free_tensor_storage(result);
	free_lin_storage(matrix);

	if (file != NULL) {
		fclose(file);
	}

	return 0;
}

