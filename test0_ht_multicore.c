#define _GNU_SOURCE
#include <algorithms.h>
#include <gen_utils.h> // for randomize array int 
#include <gen_data.h> // get_vector, gen_vector, gen_block_tensor
#include <file_utils.h> // for save_to_file
#include <test.h> // for inline functions
#include <rand_utils.h>
#include <stdlib.h> // for free
#include <string.h>
#include <assert.h>

// HT requirements
#include <tensorlibthreads.h>
#include <pthread.h>
#include <unistd.h> // for _SC_NPROCESSORS_ONLN

#define TOTAL_RUNS 20

int test0_ht_multicore(int argc, char ** argv) {

	// printf("parent: begin\n");
	// get the number of threads on CPU
	const long num_threads = sysconf( _SC_NPROCESSORS_ONLN );

	int rc;
	cpu_set_t mask;
	pthread_attr_t attr; // Initialise pthread attribute object
	pthread_t producer_thread;

	///////////////////// SETUP HT PROPERLY HERE

	CPU_ZERO( &mask ); // Clears set so that it contains no CPUs
	CPU_SET( 0, &mask ); // Set the mask to 0

	// Set affinity of the current thread to mask (0)
	// The connection is like this: mask -> affinity -> attribute -> thread
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
	// Set affinity of the process to &mask in attribute
	rc = pthread_attr_setaffinity_np( &attr, sizeof(cpu_set_t), &mask );
	if( rc != 0 ) {
		fprintf( stderr, "Error during setting of affinity.\n" );
		return 3;
	}
	
	///////////////////// START THREAD HERE

	buffer_t_multicore buffer = {
		.monitor_on_main = PTHREAD_MUTEX_INITIALIZER,
		.monitor_begin = PTHREAD_MUTEX_INITIALIZER,
		.monitor_end = PTHREAD_MUTEX_INITIALIZER,
		.steady_state = PTHREAD_COND_INITIALIZER,
		.preface = PTHREAD_COND_INITIALIZER,
		.tensor = NULL,
		.unfold_1 = NULL,
		.unfold_2 = NULL,
		.mode = -1,
		.buffer_0 = 0
	}; // or below: other init method
	// Initializes a mutex
	// if (pthread_mutex_init(&data.lock1, NULL) != 0) {
	// 		printf("\n mutex init failed\n");
	// 		return 6;
	// }

	// set the monitor_on_main
	mythread_mutex_lock(&buffer.monitor_on_main);

	// Create a thread according to that &attr and with that &data object
	mythread_create(&producer_thread, &attr, tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_producer_multicore, (void*)&buffer);

	///////////////////// START TESTING CODE HERE
	/////////////////////
	/////////////////////
	/////////////////////
	/////////////////////
	/////////////////////
	/////////////////////

	int dim_min, dim_max, n_min, n_max;
	int mode_min, mode_max;
	int block_n_min, block_n_max;

	// we must provide default arguments
	dim_min = 4;
	dim_max = 4;
	mode_min = 0;
	mode_max = dim_max-1;
	n_min = 1;
	n_max = 64;
	block_n_min = 1;
	block_n_max = n_max;
	
	// if an odd number:
	// -> the last element is the specific value for block_n
	// block_n = argv-1 (last element)
	if ((argc % 2) != 0) {
		//printf("block_n=%s\n", *(argv+argc--));
		// CONVERT string representation to integer
		sscanf (*(argv+argc), "%d", &block_n_min);
		sscanf (*(argv+argc), "%d", &block_n_max);
		argc--;
		// we did -- to decrease used argument count (to say we used this el)
	}
	switch (argc) {
		case 6:
			// mode
			sscanf (*(argv+argc--), "%d", &mode_max);
			sscanf (*(argv+argc--), "%d", &mode_min);
			printf("int mode_min=%d\n", mode_min);
			printf("int mode_max=%d\n", mode_max);
		case 4:
			// dim, n 
			sscanf (*(argv+argc--), "%d", &n_max);
			sscanf (*(argv+argc--), "%d", &n_min);
			sscanf (*(argv+argc--), "%d", &dim_max);
			sscanf (*(argv+argc--), "%d", &dim_min);
	}

	if (dim_max != 3) {
		mode_max = dim_max - 1;
	}
	
	printf("int dim_min=%d\n", dim_min);
	printf("int dim_max=%d\n", dim_max);
	printf("int n_min=%d\n", n_min);
	printf("int n_max=%d\n", n_max);
	printf("int mode_min=%d\n", mode_min);
	printf("int block_n_min=%d\n", block_n_min);
	printf("int block_n_max=%d\n", block_n_max);
	printf("int mode_max=%d\n", mode_max);
			
	char filename[BUFSIZE];
	char filename2[BUFSIZE];

	typedef void (*TVM)();
	// PROBLEM: we don't have this algorithm in external code (yet)
	TVM model_algorithm = tvm_tensor_major;
	
	TVM unfold_unfold_algorithms[] = {
		tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_consumer_multicore,
		tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_intorow
	};

	int count[] = {
		1, 1
	};
	
	for (int runs=1; runs<TOTAL_RUNS; ++runs) {
		printf("runs=%d\n", runs);

	for (size_t dim=(size_t) dim_min; dim<=(size_t) dim_max; ++dim) {
		printf("dim=%zu:\n", dim);
		size_t block_layout[dim];
		size_t tensor_layout[dim];

		for (size_t mode=mode_min; mode <= (size_t) mode_max; ++mode) {

			printf("    mode=%zu:\n", mode);
			randomize_array_int(block_layout, dim, rand_int(1,rand_int(runs, runs*3)));
			printf("            block_layout = ");
			print_to_console_sizet(block_layout, dim);

			// make tensor layout a multiple of that
			size_t block_size = 1;
			size_t mat_size = 1;
			for (size_t d=dim-1; d<dim; --d) {
				if (d > mode) {
					mat_size *= block_layout[d];
				}
			}
			mat_size *= block_layout[mode];
			for (size_t d=0; d<dim; ++d) {
				tensor_layout[d] = block_layout[d] * (int) rand_int(1,4);
				block_size *= block_layout[d];
			}

			DTYPE * unfold = get_aligned_memory(sizeof(DTYPE) * block_size, ALIGNMENT_BLOCK);
			memset(unfold, 0, block_size);

			DTYPE * unfold_2 = get_aligned_memory(sizeof(DTYPE) * block_size, ALIGNMENT_BLOCK);
			memset(unfold_2, 0, block_size);

			printf("            tensor_layout = ");
			print_to_console_sizet(tensor_layout, dim);

			struct tensor_storage *tensor = gen_block_tensor(dim, tensor_layout, block_layout);
			struct lin_storage *vector = gen_vector(tensor->layout[mode]);
			
			/////////////////////
			struct tensor_storage  *result = get_block_result_tensor(tensor, mode);
			struct tensor_storage *model_result = get_block_result_tensor(tensor, mode);	

			// Perform the model algorithm TMV
			model_algorithm(tensor, vector, &model_result->lin, mode);

			int out_algo = -1;
			int algo_counter = 0;

			size_t n = 0;
			size_t block_n = 0;

			//////////////////////////////////////////////////////////////////// BLOCK
			
			struct tensor_storage *blocked_tensor = get_block_tensor(tensor, 0, 0);
			struct tensor_storage *unblocked_result = get_block_tensor(model_result, 0, 0);

			// Operation on shared memory: tensor, vector pointers are copied over (?)
			buffer.tensor = blocked_tensor;
			buffer.unfold_1 = unfold;
			buffer.unfold_2 = unfold_2;
			buffer.mode = mode;

			if (mode == dim-1) {
				out_algo = test_algorithms(unfold_unfold_algorithms, algo_counter, count[0], &result->lin, unblocked_result->lin.data, blocked_tensor, vector, mode,
					filename, filename2, dim, n, block_n, out_algo, NULL, &buffer, NULL);
			}
			algo_counter += count[0];
			
			if (mode == dim-1) {
				out_algo = test_algorithms(unfold_unfold_algorithms, algo_counter, count[1], &result->lin, unblocked_result->lin.data, blocked_tensor, vector, mode,
					filename, filename2, dim, n, block_n, out_algo, unfold, NULL, NULL);
			}
			algo_counter += count[1];

			free_tensor_storage(unblocked_result);
			free_tensor_storage(blocked_tensor);

			if (out_algo != -1 && DUMP) {
				snprintf(filename, BUFSIZE, "%zu %zu %d", dim, mode, -1);
				SAVE(model_result->lin);
			}

			free(unfold);
			free(unfold_2);
			free_tensor_storage(tensor);
			free_tensor_storage(result);
			free_tensor_storage(model_result);
			free_lin_storage(vector);
		}
	}
	}

	// Prepare for closure: first set tensor to NULL then unlock so producer can realise it's time to finish(!)
	buffer.tensor = NULL;
	mythread_mutex_unlock(&buffer.monitor_on_main);
	mythread_join(producer_thread, NULL);
	// printf( "Joining producer thread...\n" );
	// printf("parent: end\n");
	pthread_attr_destroy(&attr);

	return 0;
}
