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
#include <mkl.h>

#define L1 4096.0 // 32 KB
#define L1_measured 2048.0
#define L2 32768.0 // 256 KB
#define L2_measured 16384.0
#define L3 3276800.0 
#define L3_measured 2097152.0 // 16 MB of L3 taken for the measurement (not half...)
#define RAM 4294967296.0 // not the actual max of RAM, just 32 GB (double RAM_measured)
#define RAM_measured 2684354560.0 // 20 GB


// #define RAM_measured 2147483648.0 // 16 GB
// #define RAM 1073741824.0 // 8 GB
// #define RAM_measured 536870912.0 // 4 GB
// #define RAM_measured 2147483648.0 // 16 GB
	   
// This macro creates a proper filename for the results folder
#define FILENAME(x); snprintf(filename, BUFSIZE, "%s/%s_%.0f_dimmin_%d_dimmax_%d_nmin_%d_nmax_%d_modemin_%d_modemax_%d_blockn_%d.csv", RESULTS_FOLDER, hostname, timespec_to_microseconds(time), dim_min, dim_max, n_min, n_max, mode_min, mode_max, block_n_in)
#define TEST(x); assert( memcmp(model_result->lin.data, x->lin.data, x->lin.size*sizeof(DTYPE))== equality )

int isvalueinarray(int val, int *arr, int size){
    int i;
    for (i=0; i < size; i++) {
        if (arr[i] == val) {
        	// printf("we found the blcok size in the array!\n");
            return 1;
        }
        if (arr[i] == 0) {
        	// printf("encoutered a 0, so a market of the end of array; set the block size in the array and move on\n");
        	arr[i] = val;
        	return 0;
        }
    }
    // printf(" we should actually never reach here! \n");
    return 0;
}

int scenario0(int argc, char ** argv) {

	mkl_set_num_threads(1);
	
	long a = 10;
	double b = 2;
	size_t test = a/b;
	printf("This is the result: %zu", test);


	size_t EXP_MIN, EXP_MAX;
	// Find out EXP and EXP_MIN here by calculation
	// const double gigabyte = 1073741824.0; // bytes
	//const double l3CacheDoubles = 3276800.0;
	// const double l3CacheDoubles = 3538944.0; // 2 MB more than L3 just to have a full picture
	// const double l2CacheDoubles = 32768.0;
	// const double doublesin4GB = (2*gigabyte)/8;
	// const double doublesin3GB = (3*gigabyte)/8;
	// const double doublesin6GB = (6*gigabyte)/8;

	// const long num_threads = sysconf( _SC_NPROCESSORS_ONLN );
	// int rc;
	// cpu_set_t mask;
	// pthread_attr_t attr; // Initialise pthread attribute object
	// pthread_t producer_thread;
	// ///////////////////// SETUP HT PROPERLY HERE
	// CPU_ZERO( &mask ); // Clears set so that it contains no CPUs
	// CPU_SET( 0, &mask ); // Set the mask to 0
	// rc = pthread_setaffinity_np( pthread_self(), sizeof(cpu_set_t), &mask );
	// if( rc != 0 ) {
	// 	fprintf( stderr, "Error during setting of the consumer thread affinity.\n" );
	// 	return 1;
	// }

	// ///////////////////// SETUP THREAD ATTRIBUTES
	// CPU_ZERO( &mask );
	// CPU_SET(num_threads/2+2, &mask);
	// rc = pthread_attr_init( &attr );
	// if( rc != 0 ) {
	// 	fprintf( stderr, "Could not initialise pthread attributes.\n" );
	// 	return 2;
	// }
	// rc = pthread_attr_setaffinity_np( &attr, sizeof(cpu_set_t), &mask );
	// if( rc != 0 ) {
	// 	fprintf( stderr, "Error during setting of affinity.\n" );
	// 	return 3;
	// }
	
	///////////////////// START THREAD HERE
	buffer_t buffer = {
		.monitor_on_main = PTHREAD_MUTEX_INITIALIZER,
		.monitor_begin = PTHREAD_MUTEX_INITIALIZER,
		.monitor_end = PTHREAD_MUTEX_INITIALIZER,
		.steady_state = PTHREAD_COND_INITIALIZER,
		.preface = PTHREAD_COND_INITIALIZER,
		.tensor = NULL,
		.unfold_1 = NULL,
		.unfold_2 = NULL,
		.mode = -1
	}; // or below: other init method

	// // set the monitor_on_main
	// mythread_mutex_lock(&buffer.monitor_on_main);
	// mythread_create(&producer_thread, &attr, tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_producer, (void*)&buffer);

	///////////////////// START TESTING CODE HERE
	/////////////////////
	/////////////////////
	/////////////////////

	int dim_min, dim_max, n_min, n_max;
	int mode_min, mode_max;
	int block_n_in;

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
	block_n_in = 8;

	// if an odd number:
	// -> the last element is the specific value for block_n_in
	// block_n_in = argv-1 (last element)
	if ((argc % 2) != 0) {
		//printf("block_n_in=%s\n", *(argv+argc--));
		// CONVERT string representation to integer
		sscanf (*(argv+argc--), "%d", &block_n_in);
		// we did -- to decrease used argument count (to say we used this el)
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
		// default stayed, so we must finish the default max
		mode_max = dim_max-1;
	}
	printf("int mode_max=%d\n", mode_max);
	printf("int block_n=%d\n", block_n_in);

	char hostname[1024];
	gethostname(hostname, 1024);
	// fprintf(file_handle, "hostname=%s\n", hostname);
	char filename[BUFSIZE];
	struct timespec time;
	clock_gettime(CLOCK_MONOTONIC, &time);
	FILENAME("results");

	printf("filename=%s\n", filename);
	FILE * file = fopen(filename, "w"); // r+ if you want to update without deleteing it
	if (file == NULL) {
		perror("Error opening file.\n");
	}

	write_header(file);

	// improvement: could include the numbered versions for completeness
	// +1 FROM TESTS: we include the model algorithm here (tvm_tesor_major)

	// const int algos_unfold = 0;
	// TVM algorithms_unfold[2] = { 
	// 	tvm_vector_major_BLIS_col_mode,
	// 	tvm_tensor_major_mine
	// };
	
	const int algos_block_unfold = 0;
	TVM algorithms_block_unfold[11] = {

		tvm_vector_major_BLAS_col_benchmarkable,
		tvm_vector_major_BLAS_col_mode2,
		// tvm_taco,
		tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_v2_unfold,
		tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_partcopy,
		tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_intocol_linin3,
		tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_intocol_linout,
		tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_intorow,
		tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal

		// out-of-place copy (destructive) -> blockRowNoTransDUnfold
		// tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_v2_unfold,

		// out-of-place copy (intrinsic version) (linear access) -> blockRowNoTransLinear
		// tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine,

		// out-of-place copy (intrinsic version - nontemporals) (just a copy - not an unfold) -> blockRowNoTransNontemporal
		// out-of-place copy (destructive) -> mortonRowNoTransDUnfold
		// tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold
	};
	
	const int algos_block_unfold_ht = 0;
	TVM algorithms_block_unfold_ht[2] = {
		tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_consumer,
		tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_consumer_prodonly
	};
	
	const int algos_block_unfold_small = 0;
	TVM algorithms_block_unfold_small[2] = {
		tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_mode1,
		tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_mode1
	};
	
	const int algos_parallel = 0;
	TVM algorithms_parallel[3] = {
		tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_v3_mine, // MKL
		tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_libx_tensor, // LIBX
		tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_libx_result2 // MKL
	};
	
	// column-major

	///////////////////
	// WHGAT THE FUCK IS WRONG WITH ME(!!!!!!!!!!!!!!!!)
	// LOL
	// OBVIOUSLY LIBX CANNOT HANDLE LARGE TENSORS>>>>>>
	const int algos_block = 3;
	TVM algorithms_block[3] = {
		// tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_libx, // 2 cases
		// tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3, // 2 cases
		// tvm_vector_major_BLAS_col_mode, // MKL

		// tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_mklgemm,

		// // tvm_vector_major_BLAS_col_mode_libx, // LIBX (does not work on large sizes !!!!)

		// tvm_hilbert_POWERS_mkl,


		// tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_v3, // 2 cases
		// tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_v3_libx, // 2 cases

		// tvm_hilbert_POWERS_libx,
		

		// tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_libx_vector3,

		// tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_libx_vector2,
		// tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_libx_vector,
		// tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_libx_tensor2,
		// tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_libx_result,
		// tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_libx_result2,
		
		tvm_output_major_input_aligned,
		tvm_vector_major_input_aligned,
		tvm_tensor_major,

		// THE BELOW IS NON BLOCKED SO RUN IT IN STREAM BENCHMARK (!)
		// tvm_vector_major_BLAS_col_mode,
		// blockRowTransPerf
		// tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_v3,
		// mortonRowTransPerf
	};

	const int algos_unfoldmem = 0;
	TVM algorithms_unfoldmem[1] = {
		block_array_int
	};

	size_t dim_size, dim_size_max, initial_perdim_size;

	if (TEST_ENV) {
		dim_size_max = ceil(pow(L3, 1/(double)10));
		initial_perdim_size = ceil(pow(L3, 1/(double)2));
	} else {
		dim_size_max = 10;
		initial_perdim_size = 71832;
		// dim_size_max = 5;
		// initial_perdim_size = 4000;
	}
	
	size_t memory_avail_tensor = initial_perdim_size*initial_perdim_size;

	// JUST FOR MEMORY ALLOCATION (hence, ceiling function)
	printf("dim_size=%zu, dim_size_max=%zu\n", dim_size, dim_size_max);
	// ASSUME 1 D ARRAY FOR THE ALLOCATION STEP(!!!)

	size_t init_block_layout[2] = {initial_perdim_size,initial_perdim_size};
	size_t init_tensor_layout[2] = {initial_perdim_size,initial_perdim_size};
	print_to_console_sizet(init_block_layout, 2);
	struct tensor_storage *tensor = gen_block_tensor(2, &init_tensor_layout, &init_block_layout);
	// So vector is max at this point(!)
	struct lin_storage *vector = gen_vector(initial_perdim_size);
	// Result also assumes max (EXP_MAX)

	tensor->dim = 10;
	size_t tensor_layoutX[10];

	// if (TEST_ENV) {
	reset_array_sizet(tensor_layoutX, 10, dim_size_max);
	// } else {
	// 	reset_array_sizet(tensor_layoutX, 10, 10);
	// }

	// block layout of tensor does not have 10D(!!!!)
	set_tensor_layout(tensor, tensor_layoutX);
	free(tensor->block_layout);
	tensor->block_layout = copy_array_int(tensor->layout, tensor->dim);
	print_to_console_sizet(tensor->block_layout, tensor->dim);
	printf("finished printing the block layout!!\n");
	print_to_console_sizet(tensor->layout, 10);
	struct tensor_storage  *result = get_block_result_tensor(tensor, 0);
	printf("we allocated result size of %zu\n", result->lin.size);
	printf("tensor_size is %zu\n", tensor->lin.size);

	// IMPORTANT STEP(!!!)
	// BECAUSE
	// blocked_tensor will get the same block layout as tensor to which it will be unfolded
	// struct tensor_storage *blocked_tensor = get_block_tensor(tensor, 0, 0);

	// buffer.tensor = tensor;
	// DTYPE * unfold   = get_aligned_memory(sizeof(DTYPE) * tensor->lin.size, ALIGNMENT_BLOCK);
	// DTYPE * unfold_2 = get_aligned_memory(sizeof(DTYPE) * L3, ALIGNMENT_BLOCK);
	// buffer.unfold_1 = unfold;
	// buffer.unfold_2 = unfold_2;

	// parameters' loops ordered according to their dependency
	for(size_t dim=(size_t) dim_min; dim<=(size_t) dim_max; ++dim) {

		printf("dim=%zu:\n", dim);

		size_t block_layout[dim];
		size_t tensor_layout[dim];

		result->dim = dim-1;
		tensor->dim = dim;

		size_t block_array[3];
		size_t block_array_size = 1;

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
		// MAKE IT 12(!!!!)
		// block_array[0] = 12;
		// int myArray[4] = {12, 24, 12, 12};
		// int myArray[6] = {12, 10, 10, 10, 8, 4};
		// int myArray[5] = {18, 16, 14, 12, 10};
		// int myArray[6] = {8, 10, 12, 10, 8, 12};
		print_to_console_sizet(block_array, block_array_size);
		// size_t temp_mode_max;
		// temp_mode_max = dim-1;

		printf("Original result size is %zu\n", result->lin.size);
		// Clear block array because we have different block sizes below (...)
		size_t block_array_copy[block_array_size];
		memset(block_array_copy, 0, block_array_size*sizeof(size_t));

		// int at_least_one_in = 0;

		// file = fopen("file.txt","a");

		// for(size_t n=EXP_MIN; n<=EXP_MIN; n+=1) { // CHANGE (1)
		// for(size_t n=EXP_MIN; n<=EXP_MAX; n+=1) {

			// if (at_least_one_in == 1) {
			// 	printf("we found a size matching!\n");
			// 	break;
			// }
			
			// Adjust both containers to have correct sizes
			// TENSOR ONLY 
			// (1) Free
			// free(tensor->layout);
			// free(tensor->layout_perm);
			// (2)
			// reset_array_sizet(tensor_layout, dim, EXP_MAX);  // CHANGE (1) -> uncomment this
			// for (int j=0; j<dim; ++j) {
			// 	int temp = n/myArray[j];
			// 	if (temp == 0) {
			// 		tensor_layout[j] = myArray[j];
			// 	} else {
			// 		tensor_layout[j] = temp*myArray[j];
			// 	}
			// }
			// set_tensor_layout(tensor, tensor_layout);

			size_t per_dim_size = ceil(pow(memory_avail_tensor, 1/(double)dim));
			int firstblock = 1;
			for(size_t i=0; i < block_array_size; ++i) {   // CHANGE (2) -> replace this with below
			// for(size_t i = 4; i < 17; i+=4) {

				// size_t block_n = i;
				size_t block_n = block_array[2];
				if (block_n == 0) {
					// continue;
					// block_n = block_array[1];
				}
				printf("        	block_n=%zu:\n", block_n);
				reset_array_sizet(block_layout, dim, block_n);

				size_t temp = per_dim_size/block_n;
				if (temp == 0) {
					temp = 1;
				}
				// assert(temp!=0);

				// find temp which works for all
				while (temp>1 && pow(temp*block_n,dim)>memory_avail_tensor) {
					temp--;
					printf("decreasing the temp size to match available tensor memory.\n", temp);
				}
				// for example, it turns out that 441 is the number of bloks which fits
				// we want to now determine which power of 2 (highest fitting it is!)
				int which_power = 0;
				while (temp >>= 1) { ++which_power; }
				temp = pow(2,which_power);
				printf("FINAL VALUE OF TEMP=%d\n", temp);
				// repeat until find one whcih is good(!)

				size_t current_per_dim_size = temp*block_n;
				assert(pow(current_per_dim_size,dim)<=memory_avail_tensor);
				reset_array_sizet(tensor_layout, dim, current_per_dim_size);
				printf("        n=%zu:\n", current_per_dim_size);

				// printf("Memory available for the tensor is %zu, while memory for the planned tensor is %zu\n", memory_avail, pow(temp*block_n,dim));
				free(tensor->layout);
				free(tensor->layout_perm);
				set_tensor_layout(tensor, tensor_layout);
				printf("Fixed tensor_size: %d\n", tensor_layout[0]);

				// IF A BLOCK PERFECTLY DIVIDES THIS SIZE -> execute all algorithms with this block size(!)
				// We found a block_n which perfectly divides the tensor
				// This should be a loop
				int all_divisible = 1;
				for (size_t d=0; d<dim; ++d) {
					if (tensor->layout[d] % block_layout[d] != 0) {
						all_divisible = 0;
						printf("went into here, or no?\n");
					}
					printf("check for d=%zu, tensor->layout[d]=%zu, block_layout[d]=%zu\n",
						d, tensor->layout[d], block_layout[d]);
				}
				if (all_divisible) {

					// REMOVE THIS(!!!) because otherwise vlaue is in array already
					if (isvalueinarray(block_n, block_array_copy, block_array_size)) { // CHANGE (so we dont do all tests in a loop!)
						printf("decided to conitnue!!\n");
						continue;
					}				

					// calculate the block size and allocate required for unfold
					// make tensor layout a multiple of that
					size_t block_size = 1;
					for (size_t d=0; d<dim; ++d) {
						block_size *= block_layout[d];
					}
					// for unfold-like algorithms
					//DTYPE * const restrict unfold = calloc(block_size, sizeof(DTYPE));
					// DTYPE * unfold = get_aligned_memory(sizeof(DTYPE) * block_size, ALIGNMENT_BLOCK);
					// memset(unfold, 0, block_size);
					// memset(unfold_2, 0, block_size);
					// DTYPE * unfold_2 = get_aligned_memory(sizeof(DTYPE) * block_size, ALIGNMENT_BLOCK);
					// memset(unfold_2, 0, block_size);
					// TENSOR AND RESULT
					// (1) Free
					free(tensor->block_layout);
					tensor->block_layout = copy_array_int(block_layout, tensor->dim);
					// printf("Tensor=\n");
					// print_to_console_sizet(tensor->layout, tensor->dim);
					// print_to_console_sizet(tensor->block_layout, tensor->dim);
					// Mode can be iterate over here without getting worked up over generating new storage (?)
					for(size_t mode=0; mode<=dim-1; ++mode) {

						printf("mode; begin\n");
						
						free(result->block_layout);
						// Result is of dim-1 so we have to adjust this function appopriately
						result->block_layout = copy_array_int_except_mode(tensor->block_layout, result->dim, mode); // mode = 0

						int result_size = 1;
						for (int i=0; i<=dim-1; ++i) {
							if (i != mode) {
								result_size *= tensor->layout[i];
							}
						}

						result->lin.size = result_size;
						printf("result size = %zu\n", result->lin.size);

						// VECTOR(!)
						vector->size = tensor->layout[mode];

						printf("    mode=%zu:\n", mode);
						// buffer.mode = mode;

						size_t mat_size = 1;
						for (size_t d=dim-1; d<dim; --d) {
							if (d > mode) {
								mat_size *= block_layout[d];
							}
						}
						// times vector size...
						mat_size *= block_layout[mode];

						// small unfold (of the required size -> just right size)
						// DTYPE * unfold_small = get_aligned_memory(sizeof(DTYPE) * mat_size, ALIGNMENT_BLOCK);
						// memset(unfold_small, 0, mat_size);

						// RESULT TENSOR
						// (1) Free
						free(result->layout);
						free(result->layout_perm);
						// (2)
						result->layout = copy_array_int_except_mode(tensor->layout, result->dim, mode);
						result->layout_perm = copy_array_int_except_mode(tensor->layout_perm, result->dim, mode);
						
						// print_to_console_sizet(result->layout, result->dim);
						// print_to_console_sizet(result->block_layout, result->dim);	

						// run only for a single block size
						// if (i == 0) {
						// 	// run all algorithms in a loop
						// 	for (int algo=0; algo<algos_block_unfold; ++algo) {
						// 		// if(mode==0 || mode==dim-1) {
						// 		// 	continue;
						// 		// }
						// 		reset_array(result->lin.data, result->lin.size, 0);
						// 		measure_unfold(
						// 			algorithms_block_unfold[algo], tensor, vector, &result->lin, mode,
						// 			file, tensor_layout[0], block_n, unfold, block_size);
						// 		fflush(file);
						// 	}
						// }

						// run all algorithms in a loop
						for (int algo=0; algo<algos_block_unfold_ht; ++algo) {
							if(mode==0 || mode==dim-1) {
								continue;
							}
							reset_array(result->lin.data, result->lin.size, 0);
							measure_unfold_ht(
								algorithms_block_unfold_ht[algo], tensor, vector, &result->lin, mode,
								file, tensor_layout[0], block_n, NULL, &buffer, block_size);
							fflush(file);
						}

						// run all algorithms in a loop
						for (int algo=0; algo<algos_block; ++algo) {
							// if ((algorithms_block[algo] == tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_libx_tensor || 
							// 	algorithms_block[algo] == tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_libx_vector2 ||
							// 	algorithms_block[algo] == tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_libx_vector ||
							// 	// algorithms_block[algo] == tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_libx_vector3 ||
							// 	algorithms_block[algo] == tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_libx_tensor2 ||
							// 	algorithms_block[algo] == tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_libx_result ||
							// 	algorithms_block[algo] == tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_libx_result2)
							// 	&& (mode > dim/2)) {
							// 	// prefetch ones skip if mode>1
							// 	printf("skip\n");
							// 	continue;
							// }
							if ((algorithms_block[algo] == tvm_vector_major_BLAS_col_mode || 
								algorithms_block[algo] == tvm_vector_major_BLAS_col_mode_libx) && (firstblock == 0)) {
								printf("Passing on the looped algorithms for all block sizes other than the first one\n");
								continue;
							}
							// printf("n=%zu,block_n=%zu\n",tensor_layout[0], block_n);
							printf("before\n");
							// fprintf(file, "mode=%zu,n=%zu,block_n=%zu,algo_id=%d\n", mode, tensor_layout[0], block_n);
							reset_array(result->lin.data, result->lin.size, 0);
							printf("after: measuring the next algorithm\n");
							measure(
								algorithms_block[algo], tensor, vector, &result->lin, mode,
								file, tensor_layout[0], block_n, block_size);	
							fflush(file);
						}

						for (int algo=0; algo<algos_parallel; ++algo) {
							reset_array(result->lin.data, result->lin.size, 0);
							measure(
								algorithms_parallel[algo], tensor, vector, &result->lin, mode,
								file, tensor_layout[0], block_n, block_size);	
							fflush(file);
						}
		
						// run all algorithms in a loop
						// for (int algo=0; algo<algos_unfoldmem; ++algo) {
						// 	// Cannot unfold into the result(!!!) Because it's too small
						// 	reset_array(blocked_tensor->lin.data, blocked_tensor->lin.size, 0);
						// 	measure_unfold_mem(algorithms_unfoldmem[algo], &blocked_tensor->lin, tensor, 0,
						// 		mode, file, n, block_n, block_size);
						// 	fflush(file);
						// }

						// break;

						// run all algorithms in a loop
						// for (int algo=0; algo<algos_block_unfold_small; ++algo) {
						// 	if (mode!=tensor->dim-1) {
						// 	reset_array(result->lin.data, result->lin.size, 0);
						// 	measure_unfold(
						// 		algorithms_block_unfold_small[algo], tensor, vector, &result->lin, mode,
						// 		file, tensor_layout[0], block_n, unfold_small, block_size);
						// 	fflush(file);
						// 	}
						// }

						// free(unfold_small);

					}
					firstblock = 0;
					// Finished the mode loop
					// free(unfold);
					// free(unfold_2);

				} // for mode finish (above }); that is end of if all divisible statement

				// if ((block_n>l2) && (inc_block_n>1)) {
				// 	// printf("|WE SHOULD MAKE NEXT ONE even??? ");
				// 	if ((block_n + inc_block_n) % 2 == 1) {
				// 		block_n += inc_block_n-2;
				// 	} else {
				// 		block_n += inc_block_n-1;
				// 	}
				// }

			// }
			// if (at_least_one_in == 1) {
			// 	printf("Break this dimension\n");
			// 	break;
			// }

#if 0
			// Mode can be iterate over here without getting worked up over generating new storage (?)
			for(size_t mode=(size_t) mode_min; mode<=temp_mode_max; ++mode) {
				printf("    mode=%zu:\n", mode);

				// RESULT TENSOR
				// (1) Free
				free(result->layout);
				free(result->layout_perm);
				// (2)
				result->layout = copy_array_int_except_mode(tensor->layout, result->dim, mode);
				result->layout_perm = copy_array_int_except_mode(tensor->layout_perm, result->dim, mode);

				// run all algorithms in a loop
				for (int algo=0; algo<algos_unfold; ++algo) {
					reset_array(result->lin.data, result->lin.size, 0);
					measure(
						algorithms_unfold[algo], tensor, vector, &result->lin, mode,
						file, n, block_n);
					fflush(file);
				}

			}
#endif

		}

	}

	// fclose(file);

	// free(unfold);
	// free(unfold_2);

	free_tensor_storage(tensor);
	free_tensor_storage(result);
	free_lin_storage(vector);
	// free_tensor_storage(blocked_tensor);

	if (file != NULL) {
		fclose(file);
	}

	buffer.tensor = NULL;
	// mythread_mutex_unlock(&buffer.monitor_on_main);
	// mythread_join(producer_thread, NULL);
	// pthread_attr_destroy(&attr);

	return 0;
}

