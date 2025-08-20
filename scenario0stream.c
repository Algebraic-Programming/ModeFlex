// HT requirements
#define _GNU_SOURCE

#include <string.h> // for memcmp
#include <assert.h>
#include <stdlib.h> // for free
#include <math.h> // for round and pow

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

#define L1 4096.0
#define L1_measured 2048.0
#define L2 32768.0
#define L2_measured 16384.0
#define L3 3276800.0 
#define L3_measured 2097152.0 // 16 MB of L3 taken for the measurement (not half...)
#define RAM 4294967296.0 // not the actual max of RAM, just 32 GB (double RAM_measured)
// #define RAM_measured 2147483648.0 // 16 GB
#define RAM_measured 2684354560.0 // 20 GB

// This macro creates a proper filename for the results folder
#define FILENAME(x); snprintf(filename, BUFSIZE, "%s/%s_%.0f_dimmin_%d_dimmax_%d_nmin_%d_nmax_%d_modemin_%d_modemax_%d_blockn_%d.csv", RESULTS_FOLDER, hostname, timespec_to_microseconds(time), dim_min, dim_max, n_min, n_max, mode_min, mode_max, block_n)
#define TEST(x); assert( memcmp(model_result->lin.data, x->lin.data, x->lin.size*sizeof(DTYPE))== equality )

int scenario0stream(int argc, char ** argv) {

	// initialize LIBXSMM
	libxsmm_dmmfunction a_kernel;

	size_t EXP_MIN, EXP_MAX;
	// Find out EXP and EXP_MIN here by calculation
	//const double l3CacheDoubles = 3276800.0;
	const double l3CacheDoubles = 3538944.0; // 2 MB more than L3 just to have a full picture
	const double l2CacheDoubles = 32768.0;
	
	const long num_threads = sysconf( _SC_NPROCESSORS_ONLN );
	int rc;
	cpu_set_t mask;
	pthread_attr_t attr; // Initialise pthread attribute object
	pthread_t producer_thread;
	///////////////////// SETUP HT PROPERLY HERE
	CPU_ZERO( &mask ); // Clears set so that it contains no CPUs
	CPU_SET( 0, &mask ); // Set the mask to 0
	rc = pthread_setaffinity_np( pthread_self(), sizeof(cpu_set_t), &mask );
	if( rc != 0 ) {
		fprintf( stderr, "Error during setting of the consumer thread affinity.\n" );
		return 1;
	}

	///////////////////// SETUP THREAD ATTRIBUTES
	CPU_ZERO( &mask );
	CPU_SET(num_threads/2, &mask);
	rc = pthread_attr_init( &attr );
	if( rc != 0 ) {
		fprintf( stderr, "Could not initialise pthread attributes.\n" );
		return 2;
	}
	rc = pthread_attr_setaffinity_np( &attr, sizeof(cpu_set_t), &mask );
	if( rc != 0 ) {
		fprintf( stderr, "Error during setting of affinity.\n" );
		return 3;
	}
	
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

	// set the monitor_on_main
	mythread_mutex_lock(&buffer.monitor_on_main);
	mythread_create(&producer_thread, &attr, tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_producer, (void*)&buffer);

	///////////////////// START TESTING CODE HERE
	/////////////////////
	/////////////////////
	/////////////////////

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

	char hostname[1024];
	gethostname(hostname, 1024);
	// fprintf(file_handle, "hostname=%s\n", hostname);
	char filename[BUFSIZE];
	struct timespec time;
	clock_gettime(CLOCK_MONOTONIC, &time);
	FILENAME("results");

	printf("filename=%s\n", filename);
	FILE * file = fopen(filename, "a");
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
	TVM algorithms_block_unfold[4] = {
		// out-of-place copy (destructive) -> blockRowNoTransDUnfold
		// tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_v2_unfold,

		// out-of-place copy (intrinsic version) (linear access) -> blockRowNoTransLinear
		// tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine,

		// out-of-place copy (intrinsic version - nontemporals) (just a copy - not an unfold) -> blockRowNoTransNontemporal
		tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_partcopy_stream,
		tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_partcopy,
		tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_intorow,
		tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal

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
		tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_mode1,
		tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_mode1
	};

	// column-major
	const int algos_block = 1;
	TVM algorithms_block[5] = {
		tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_libx,
		tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3,
		tvm_vector_major_BLAS_col_mode_libx,
		tvm_vector_major_BLAS_col_mode,
		// tvm_taco,
		// tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_v3_libx,
		tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_v3,
		// blockRowTransPerf
		// mortonRowTransPerf
	};

	const int algos_nomode = 0;
	TVM algorithms_nomode[4] = {
		tvm_output_major_BLAS_row_BLAS,
		tvm_vector_major_BLAS_col_GEMM_libx,
		tvm_output_major_BLAS_row_libx,
		tvm_vector_major_BLAS_col_BLAS,
	};

	// column-major
	// const int algos_block_libx = 0;
	// TVM algorithms_block_libx[1] = {
	// 	tvm_vector_major_BLAS_col_mode_libx
	// };

	// parameters' loops ordered according to their dependency
	for(size_t dim=(size_t) dim_min; dim<=(size_t) dim_max; ++dim) {
		printf("inside this\n");
		printf("dim=%zu:\n", dim);

		// size_t twice_size_meaning_cache = L2;
		// size_t size_to_fit = L2_measured;
		// // // Scenario 2: Not even numbers + Half the size for dim 1
		// size_t calc0 = floor(pow(size_to_fit,1/(double)dim));
		// size_t calc = pow(calc0, dim);
		// // If too large, take the next smaller even number and recalculate things
		// if (calc >= twice_size_meaning_cache) {
		// 	calc0 = calc0 - 1;
		// 	calc = pow(calc0, dim);
		// }
		// if (dim == 1) {
		// 	calc0 = calc0/2;
		// 	calc = pow(calc0, dim);
		// }
		// size_t next_calc = pow(calc0+1, dim);
		// printf("calc0=%zu, calc=%zu, next_calc=%zu\n", calc0, calc, next_calc);
		// printf("all together its %zu doubles\n", (next_calc + next_calc/(calc0+1) + (calc0+1)));
		// while ((next_calc + next_calc/(calc0+1) + (calc0+1)) < (0.85*twice_size_meaning_cache)) {
		// 	printf("we are not saturating even if we increase! so increase, calc0+1=%zu\n", calc0+1);
		// 	calc0 = calc0+1;
		// 	next_calc = pow(calc0+1, dim);
		// }

		// if (calc0 % 2 != 0) {
		// 	calc = calc0-1;
		// } else {
		// 	calc = calc0;
		// }
		// printf("calc=%zu, calc0=%zu\n", calc, calc0);

		// Scenario 1: We need sizes for L3
		// size_t calc0 = ceil(pow(size_to_fit,1/(double)dim));
		// if (dim <= 20) {
		// 	// Get the nearest even number
		// 	if (calc0 % 2 != 0) {
		// 		calc0 = calc0 - 1;
		// 	}
		// 	size_t calc = pow(calc0, dim);
		// 	// If too large, take the next smaller even number and recalculate things
	// 	if (calc >= twice_size_meaning_cache) {
		// 		calc0 = calc0 - 2;
		// 		calc = pow(calc0, dim);
		// 	}
		// } else {
		// 	size_t calc = pow(calc0, dim);
		// 	// If too large, take the next smaller even number and recalculate things
		// 	if (calc >= twice_size_meaning_cache) {
		// 		calc0 = calc0 - 1;
		// 		calc = pow(calc0, dim);
		// 	}
		// 	if (dim == 1) {
		// 		calc0 = calc0/2;
		// 		calc = pow(calc0, dim);
		// 	}
		// }

		// double mem_in_kb = (8*(calc+calc0+pow(calc0,dim-1))/(double)1024.0);
		// double mem_in_mb = mem_in_kb / (double) 1024;
		// printf("1=%d, 2=%d, 3=%d, 8=%d\n", calc, 2*calc, 3*calc, 8*calc);
	    // printf("%d = root is %f then we use %d (so after pow it's %d doubles) so %d bytes and total mem usage is %f KB or %f MB\n",
	    	// dim, pow(size_to_fit,1/dim), calc0, calc, 8*calc, mem_in_kb, mem_in_mb); 

		EXP_MIN = 2;
		
		
		// size_t l3_max = pow(2, 31);
		// size_t l3_max = round(pow(l3CacheDoubles, 1.0/dim) + 1);
		// size_t l3_max = round(pow(l3CacheDoubles, 1.0/dim) + 1);
		// size_t l2 = round(pow(l2CacheDoubles, 1.0/dim));
		// size_t inc_exp_n = (EXP_MAX - l2) / 20.0;
		// printf("EXP_MAX=%zu, EXP_MIN=%zu\n",
			// EXP_MAX, EXP_MIN);

		size_t block_array[3];
		size_t block_array_size = 3;
		if (dim == 2) {
			block_array[0] =    44;
			block_array[1] =   124;
			block_array[2] =   572;
		} else if (dim == 3) {
			block_array[0] =   12; //12;
			block_array[1] =   24;
			block_array[2] =   68;
		} else if (dim == 4) {
			block_array[0] =   6; //6;
			block_array[1] =  10;
			block_array[2] =  24;
		} else if (dim == 5) {
			block_array[0] =  4;//10; //4;
			block_array[1] =  6; //6;
			block_array[2] = 12;
		} else if (dim == 6) {
			block_array[0] =   3;//10;//3;
			block_array[1] =   4; //4;
			block_array[2] =   8;
		} else if (dim == 7) {
			block_array[0] =   2;
			block_array[1] =   4;
			block_array[2] =   6;
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

		size_t n;
		// if (dim == 10) {
		// 	n = block_array[1];
		// } else {
		// 	n = block_array[2];
		// }
		// EXP_MAX = n;

		// MAKE IT 
		// size_t block_array[28] = {64,128,256,512,1024,1536,2048,3072,4096,6144,8192,
		// 	12288,16384,32768,65536,131072,262144,524288,1048576,2097152,
		// 	4194304,8388608,16777216,33554432,67108864,134217728,268435456,536870912};
		// size_t block_array_size = 28;

		// EXP MAX IS TE MAXIMAL SIZE (1D) so in bytes how much we shjould reserve
		// EXP_MAX = L3;
		EXP_MAX = block_array[block_array_size-1];
		n = EXP_MAX;
		printf("EXP_MAX is seleced to be %zu\n", EXP_MAX);

		size_t block_layout[dim];
		size_t tensor_layout[dim];
		
		size_t temp_mode_max;
		if (dim-1 < (size_t) mode_max) {
			temp_mode_max = dim-1;
		} else {
			temp_mode_max = mode_max;
		}
		
		// Initialize with dummy block_layout
		// initialize with dummy tensor_layout (kind of max one)
		reset_array_sizet(tensor_layout, dim, n);
		reset_array_sizet(block_layout, dim, 1);
		
		// Mode: 0 (here it doesn't matter)
		struct tensor_storage *tensor = gen_block_tensor(dim, tensor_layout, block_layout);
		struct lin_storage *vector = gen_vector(tensor->layout[0]);
		struct tensor_storage  *result = get_block_result_tensor(tensor, 0);
		
		buffer.tensor = tensor;
		
		size_t tensor_size = 1;
		for (size_t d=0; d<dim; ++d) {
			tensor_size *= EXP_MAX;
		}
		
		printf("Original result size is %zu\n", result->lin.size);
		DTYPE * unfold = get_aligned_memory(sizeof(DTYPE) * tensor_size, ALIGNMENT_BLOCK);
		memset(unfold, 0, tensor_size);
		buffer.unfold_1 = unfold;

		DTYPE * unfold_2 = get_aligned_memory(sizeof(DTYPE) * tensor_size, ALIGNMENT_BLOCK);
		memset(unfold_2, 0, tensor_size);
		buffer.unfold_2 = unfold_2;
		
		// for(size_t n=EXP_MIN; n<=EXP_MAX; n+=1) {
			// printf("        n=%zu:\n", n);
		for(size_t i=0; i < block_array_size; ++i) {
			printf("        i=%zu:\n", i);
			
			// we must use n in another meaning, which is the actual size of the tensor taken form the block array
			n = block_array[i];
			// if (dim == 10) {
			// 	n = block_array[1];
			// } else {
			// 	n = block_array[2];
			// }
			if (n == 0) {
				continue;
			}
			
			// Adjust both containers to have correct sizes
			// TENSOR ONLY 
			// (1) Free
			free(tensor->layout);
			free(tensor->layout_perm);
			// (2)
			reset_array_sizet(tensor_layout, dim, n);
			set_tensor_layout(tensor, tensor_layout);
			
			int result_size = 1;
			for (int i=1; i<=dim-1; ++i) {
				result_size *= n;
			}
			result->lin.size = result_size;
			printf("result size =%zu, while tensor size=%zu\n", result->lin.size, tensor->lin.size);
			
			// VECTOR(!)
			vector->size = n;
			size_t block_n = n;
			size_t block_size = 1;
			for (size_t d=0; d<dim; ++d) {
				block_size *= block_n;
			}
			
			// TENSOR AND RESULT
			// (1) Free
			free(tensor->block_layout);
			free(result->block_layout);
			// // (2)
			reset_array_sizet(block_layout, dim, block_n);
			
			// printf("error\n");
			tensor->block_layout = copy_array_int(block_layout, tensor->dim);
			
			// Result is of dim-1 so we have to adjust this function appopriately
			result->block_layout = copy_array_int_except_mode(tensor->block_layout, result->dim, 0); // mode = 0
			
			// Mode can be iterate over here without getting worked up over generating new storage (?)
			for(size_t mode=(size_t) mode_min; mode <= temp_mode_max; ++mode) {
				
				printf("    mode=%zu:\n", mode);
				
				// buffer.mode = mode;
				
				// RESULT TENSOR
				// (1) Free
				free(result->layout);
				free(result->layout_perm);
				// (2)
				result->layout = copy_array_int_except_mode(tensor->layout, result->dim, mode);
				result->layout_perm = copy_array_int_except_mode(tensor->layout_perm, result->dim, mode);
				
				// run all algorithms in a loop
				for (int algo=0; algo<algos_block_unfold; ++algo) {
					reset_array(result->lin.data, result->lin.size, 0);
					measure_unfold(
						algorithms_block_unfold[algo], tensor, vector, &result->lin, mode,
						file, n, block_n, unfold, block_size);
					fflush(file);
				}

				// run all algorithms in a loop
				for (int algo=0; algo<algos_block_unfold_ht; ++algo) {
					reset_array(result->lin.data, result->lin.size, 0);
					measure_unfold_ht(
						algorithms_block_unfold_ht[algo], tensor, vector, &result->lin, mode,
						file, n, block_n, NULL, &buffer, block_size);
					fflush(file);
				}

				// run all algorithms in a loop
				for (int algo=0; algo<algos_block; ++algo) {
					reset_array(result->lin.data, result->lin.size, 0);
					measure(
						algorithms_block[algo], tensor, vector, &result->lin, mode,
						file, n, block_n, block_size);
					fflush(file);
				}

				// run all algorithms in a loop
				// for (int algo=0; algo<algos_block_libx; ++algo) {
				// 	reset_array(result->lin.data, result->lin.size, 0);
				// 	measure_libx(
				// 		algorithms_block_libx[algo], tensor, vector, &result->lin, mode,
				// 		file, n, block_n, block_size, &a_kernel);
				// 	fflush(file);
				// }

			}
			// run all algorithms in a loop
			for (int algo=0; algo<algos_nomode; ++algo) {
				reset_array(result->lin.data, result->lin.size, 0);
				measure(
					algorithms_nomode[algo], tensor, vector, &result->lin, 0,
					file, n, block_n, block_size);
				fflush(file);
			}

			// // Move faster after sizes until l2 are in (so we don't spend all day benchmarking this)
			// if ((n>l2) && (inc_exp_n>1)) {
			// 	// printf("|WE SHOULD MAKE NEXT ONE even??? ");
			// 	if ((n) % 2 == 1) {
			// 		n += inc_exp_n-2;
			// 	} else {
			// 		n += inc_exp_n-1;
			// 	}
			// }
			// break;
		}
		
		free_tensor_storage(tensor);
		free_tensor_storage(result);
		free_lin_storage(vector);

	}

	if (file != NULL) {
		fclose(file);
	}

	buffer.tensor = NULL;
	mythread_mutex_unlock(&buffer.monitor_on_main);
	mythread_join(producer_thread, NULL);
	pthread_attr_destroy(&attr);

	return 0;
}

