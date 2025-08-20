#include <stdio.h>
#include <stdlib.h>
#include <structures.h>
#include <rand_utils.h>
#include <gen_utils.h>
#include <math.h>
#include <file_utils.h>
#include <numa.h>
#include <omp.h>
#include <assert.h>

void
reset_struct_int(struct lin_storage * storage) {
	for (size_t i=0; i<storage->size; ++i) {
		storage->data[i] = 0;
	}
}

int
verify_data_parallel_results(const struct lin_storage * const result, const size_t stripe_size) {

	int correct = 1;
	size_t pos = 0;

	#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		int nthreads = omp_get_num_threads();
		size_t partition_size = result->size / nthreads;
		// size_t partition_size = result->
		// printf("partition_size = %zu / %d = %zu\n", result->size, nthreads, result->size / nthreads); 
		
		#pragma omp for ordered schedule(static,1)
	    for (int t=0; t<nthreads; ++t)
	    {
	        assert( t==tid );
	        #pragma omp ordered
	        {
    			size_t local_pos = 0;
				while ( fabs( result->master_data[pos] - result->local_data[tid][local_pos] ) < 1e-6 ) {
					++pos;
					++local_pos;
					// printf("el at local_pos %zu (abs pos=%zu) correct.\n", local_pos, pos);
					if (local_pos == partition_size) {
						break;
					}
				}
				if (local_pos != partition_size) {
					printf("ERROR: result1=%f, result2=%f\n", result->master_data[pos], result->local_data[tid][local_pos]);
					correct = 0;
				} else {
					printf("INFO: verify_data_parallel_results: CORRECT (part %d)\n", tid);
				}
	        }
        }
    }
    
    return correct;
}

int
verify_data_master_data_results(const struct lin_storage * const result) {

	int correct = 1;
	size_t pos = 0;

	while ( fabs( result->master_data[pos] - result->data[pos] ) < 1e-6 ) {
		++pos;
		if (pos == result->size) {
			break;
		}
	}
	if (pos != result->size) {
		printf("ERROR: result1=%f, result2=%f\n", result->master_data[pos], result->data[pos]);
		correct = 0;
	} else {
		// printf("INFO: verify_data_master_data_results: CORRECT\n");
	}

    return correct;
    
}

struct tensor_storage *
get_tensor(const struct tensor_storage * input_tensor, const size_t mode) {
	struct tensor_storage * tensor = calloc(1, sizeof(struct tensor_storage));
	tensor->dim = input_tensor->dim-1; // ensure we do not go below 1

	tensor->layout = copy_array_int_except_mode(input_tensor->layout, tensor->dim, mode);
	tensor->layout_perm = copy_array_int_except_mode(input_tensor->layout_perm, tensor->dim, mode);
	tensor->block_layout = copy_array_int_except_mode(input_tensor->block_layout, tensor->dim, mode);

	tensor->lin.size = 1;
	for (size_t i=0; i<(tensor->dim); ++i) {
		tensor->layout_perm[i] = i; // Assume zero-permutation
		tensor->lin.size = (tensor->lin.size) * tensor->layout[i];
	}

	// initialize empty array
	tensor->lin.data = calloc(tensor->lin.size, sizeof(DTYPE));

	return tensor;
}

// Operating function: returns an in_out permutation of the input_tensor
// Permutes: lin.data and layout
struct tensor_storage *
get_in_out_unfold(const struct tensor_storage * input_tensor, const int flag, const size_t mode) { 
	// flag: 0 - in_out, 1 - out_in
	struct tensor_storage * tensor = calloc(1, sizeof(struct tensor_storage));
	tensor->dim = input_tensor->dim;

	tensor->layout = copy_array_int(input_tensor->layout, input_tensor->dim);
	//print_to_console_sizet(input_tensor->layout_perm, tensor->dim);
	tensor->layout_perm = get_in_out_layout_perm(input_tensor->dim, input_tensor->layout_perm);
	//print_to_console_sizet(tensor->layout_perm, tensor->dim);

	tensor->lin.size = input_tensor->lin.size;
	// arrange the data (+ init the array)
	if (flag == 0) {
		in_out_array_int(&tensor->lin, input_tensor, mode);
	} else {
		out_in_array_int(&tensor->lin, input_tensor, mode);
	}

	return tensor;
}


// Initialize the memory as well (so main() cannot infer anything about the storage)
// Operating functions: should be callable
struct tensor_storage *
gen_tensor(const size_t num_dimensions, const size_t * tensor_layout) {
	struct tensor_storage * tensor = calloc(1, sizeof(struct tensor_storage));
	tensor->dim = num_dimensions;

	set_tensor_layout(tensor, tensor_layout);

	set_seed(TENSOR_SEED);
	// #if (TEST_ENV == 0)
	// 	gen_array_double_stochastic(&tensor->lin, tensor_layout[0]);
	// #else
		gen_array_double(&tensor->lin);
	// #endif
	
	return tensor;
}

// same as above but with additional parameter of block_layout
struct tensor_storage *
gen_block_tensor(const size_t num_dimensions, const size_t * tensor_layout, const size_t * block_layout) {
	struct tensor_storage * tensor = calloc(1, sizeof(struct tensor_storage));
	tensor->dim = num_dimensions;

	set_tensor_layout(tensor, tensor_layout);
	tensor->block_layout = copy_array_int(block_layout, tensor->dim);

	set_seed(TENSOR_SEED);
	// #if (TEST_ENV == 0)
	// 	gen_array_double_stochastic(&tensor->lin, tensor_layout[0]);
	// #else
		gen_array_double(&tensor->lin);
	// #endif

	return tensor;
}


// same as above but with additional parameter of block_layout
struct tensor_storage *
gen_block_tensor_clean(const size_t num_dimensions, const size_t * tensor_layout, const size_t * block_layout) {
	struct tensor_storage * tensor = calloc(1, sizeof(struct tensor_storage));
	tensor->dim = num_dimensions;

	tensor->layout = copy_array_int(tensor_layout, tensor->dim);
	tensor->block_layout = copy_array_int(block_layout, tensor->dim);
	// tensor->layout2 = copy_array_int(tensor_layout, tensor->dim);

	tensor->layout_perm = malloc(tensor->dim * sizeof(size_t));

	set_seed(TENSOR_SEED);

	// compute the tensor->lin.size (and assign layout_perm)
	tensor->lin.size = 1;
	for (size_t i=0; i<tensor->dim; ++i) {
		tensor->layout_perm[i] = i; // Assume zero-permutation
		tensor->lin.size = (tensor->lin.size) * tensor->layout[i];
		//printf("tensor.size: %ld\n", tensor->lin.size);
	}
	// tensor->lin.p_sizes = malloc(tensor->dim * sizeof(size_t));

	// #if (TEST_ENV == 0)
	// 	gen_array_double_stochastic(&tensor->lin, tensor_layout[0]);
	// #else
		gen_array_double(&tensor->lin);
	// #endif

	return tensor;
}

struct tensor_storage *
gen_block_tensor_clean_safe(const size_t num_dimensions, const size_t * tensor_layout, const size_t * block_layout, const size_t dim_max, const int local_allocation) {

	struct tensor_storage * tensor = calloc(1, sizeof(struct tensor_storage));

	tensor->dim = num_dimensions;

	// All this is done with respect to dim_max
	tensor->layout = calloc(dim_max, sizeof(size_t));
	tensor->block_layout = calloc(dim_max, sizeof(size_t));
	tensor->layout2 = calloc(dim_max, sizeof(size_t));
	tensor->layout_perm = malloc(dim_max * sizeof(size_t));
	for (size_t i=0; i<dim_max; ++i) {
		tensor->layout_perm[i] = i; // Assume zero-permutation
	}

	// This is done with respect to dim ACTUAL (most likely 1D)
	memcpy(tensor->layout, tensor_layout, tensor->dim * sizeof(size_t));
	memcpy(tensor->block_layout, block_layout, tensor->dim * sizeof(size_t));
	tensor->lin.size = 1;
	for (size_t i=0; i<tensor->dim; ++i) {
		tensor->lin.size = (tensor->lin.size) * tensor->layout[i];
	}
	
	set_seed(TENSOR_SEED);
	// #if (TEST_ENV == 0)
	// 	gen_array_double_stochastic(&tensor->lin, tensor_layout[0]);
	// #else
		// gen_array_double(&tensor->lin);
	// #endif
	 		
	#if (TEST_ENV == 1) 
		#ifdef SINGLESOCKET
			printf("Single socket setting; Use malloc (for master_data)\n");
			tensor->lin.master_data = get_aligned_memory(tensor->lin.size * sizeof(DTYPE), ALIGNMENT);
		#else
			printf("Many sockets setting; Use numa_alloc_interleaved (for master_data)\n");
			tensor->lin.master_data = numa_alloc_interleaved(tensor->lin.size * sizeof(DTYPE));
		#endif
	#endif
			
	switch (local_allocation) {
		case 0: // both
			printf("INFO: Interleaved allocation\n");
			gen_array_double(&tensor->lin);
			// tensor->lin.data = numa_alloc_interleaved(tensor->lin.size * sizeof(DTYPE));
			assert(tensor->lin.data);
		case 1: // local allocation
			printf("INFO: Local allocation\n");
 			gen_array_double_locally(&tensor->lin);
 			break;
 		case 2: // interleaved allocation
			printf("INFO: Interleaved allocation ONLY\n");
			gen_array_double(&tensor->lin);
			assert(tensor->lin.data);
 			break;
 		default:
 			printf("NO SUCH OPTION!\n");
 			exit(-1);
 			break;
 	}

	return tensor;
}

// modification to create a tensor to accommodate a result for the specific mode
// invariants NOT checked:
// input_tensor at least 2D
struct tensor_storage *
get_block_result_tensor(const struct tensor_storage * input_tensor, const size_t mode) {

	// printf("Preparing blocked result tensor\n");

	struct tensor_storage * tensor = calloc(1, sizeof(struct tensor_storage));
	tensor->dim = input_tensor->dim-1; // ensure we do not go below 1
	// printf("tensor dim %zu is input tensor dim %zu - 1\n", tensor->dim, input_tensor->dim);

	tensor->layout = copy_array_int_except_mode(input_tensor->layout, tensor->dim, mode);
	tensor->layout_perm = copy_array_int_except_mode(input_tensor->layout_perm, tensor->dim, mode);
	tensor->block_layout = copy_array_int_except_mode(input_tensor->block_layout, tensor->dim, mode);

	// printf("dim=%zu, \n", tensor->dim);
	// print_to_console_sizet(tensor->layout, tensor->dim);
	// print_to_console_sizet(input_tensor->layout, input_tensor->dim);
	
	tensor->lin.size = 1;
	for (size_t i=0; i<(tensor->dim); ++i) {
		tensor->layout_perm[i] = i; // Assume zero-permutation
		tensor->lin.size = (tensor->lin.size) * tensor->layout[i];
		printf("Multiply: new tensor size is %zu\n", tensor->lin.size);
	}

	printf("Allocating an output tensor of size %zu\n", tensor->lin.size);
	// For all modes, local allocation is a must
	// if (mode != 0) {
	// 	// initialize an empty array
	// 	tensor->lin.data = malloc(tensor->lin.size * sizeof(DTYPE));
	// 	#pragma omp parallel 
	// 	{
	// 		int nthreads = omp_get_num_threads();
	// 		int tid = omp_get_thread_num();
	// 		size_t partition_size = tensor->lin.size / nthreads;
	// 		assert(nthreads*partition_size == tensor->lin.size); // If this fails means we do not not divide the tensor equally
	// 		for (size_t el=(tid*partition_size); el<((tid+1)*partition_size); ++el) {
	// 			tensor->lin.data[el] = .0;
	// 		}
	// 	}
	// } else {
	
	#ifdef SINGLESOCKET
		printf("Single socket; Use a simple malloc!\n");
		tensor->lin.data = get_aligned_memory(tensor->lin.size * sizeof(DTYPE), ALIGNMENT);
		// test1 relies on the fact that the result tensor is zero'ed (!)
		reset_struct_int(&tensor->lin);
	#else
		printf("Using all sockets; Use interleaved allocation!\n");
		tensor->lin.data = numa_alloc_interleaved(tensor->lin.size * sizeof(DTYPE));
	#endif

	assert(tensor->lin.data);

	return tensor;
}

// this one takes an exisitng tensor, and creates a new one from it
struct tensor_storage *
get_block_tensor(const struct tensor_storage * input_tensor, const int unblock, const int morton) {

	// printf("Stats: tensor dim = %zu, size = %zu, \n", input_tensor->dim, input_tensor->lin.size);
	struct tensor_storage * tensor = calloc(1, sizeof(struct tensor_storage));
	tensor->dim = input_tensor->dim;

	tensor->layout = copy_array_int(input_tensor->layout, tensor->dim);

	// malloc block_layout and set to a value
	//tensor->block_layout = get_array_int(tensor->dim, block_size);
	tensor->block_layout = copy_array_int(input_tensor->block_layout, tensor->dim);

	// Problem: the memory rerranging functions work with input_tensor, hence it should have the layout2 with it(!)

	tensor->layout2 = calloc(tensor->dim, sizeof(size_t));
	#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		int nthreads;
		if (tid == 0) {
			for (int i=0; i<(int) tensor->dim; ++i) {
				size_t temp_value = tensor->layout[i]/tensor->block_layout[i];
				nthreads = omp_get_num_threads();
				while (temp_value % nthreads != 0) {
					--nthreads;
				}
				tensor->layout2[i] = tensor->block_layout[i] * (temp_value / nthreads);
			}
		}
	}
	// #pragma omp parallel
	// {
	// 	int tid = omp_get_thread_num();
	// 	int nthreads = omp_get_num_threads();
	// 	if (tid == 0) {
	// 		for (int i=0; i<tensor->dim; ++i) {
	// 			if (tensor->block_layout[i] > tensor->layout[i]/nthreads) {
	// 				tensor->layout2[i] = tensor->block_layout[i];
	// 			} else {
	// 				tensor->layout2[i] = tensor->layout[i]/nthreads;
	// 			}
	// 		}
	// 	}
	// }
	
	tensor->layout_perm = copy_array_int(input_tensor->layout_perm, tensor->dim);

	tensor->lin.size = input_tensor->lin.size;
	// arrange the data (+ init the array)
	switch (morton) {
		case 0:
			block_array_int(&tensor->lin, input_tensor, unblock, 0);
			break;
		case 1:
			morton_block_array_int(&tensor->lin, input_tensor, unblock);
			break;
		case 2:
			hilbert_block_array_int(&tensor->lin, input_tensor, unblock);
			break;
		case 3:
			// printf("input_tensor layout2 = %zu\n", input_tensor->layout2[0]);
			block_array_int(&tensor->lin, input_tensor, unblock, 1);
			break;
	}

	return tensor;
}

// this one takes an exisitng tensor, and creates a new one from it
struct tensor_storage *
get_blockmodemajor_tensor(const struct tensor_storage * input_tensor, const size_t mode) {
	struct tensor_storage * tensor = calloc(1, sizeof(struct tensor_storage));
	tensor->dim = input_tensor->dim;

	tensor->layout = copy_array_int(input_tensor->layout, tensor->dim);
	tensor->layout_perm = copy_array_int(input_tensor->layout_perm, tensor->dim);
	//print_to_console(input_tensor->layout, tensor->dim);

	// malloc block_layout and set to a value
	tensor->block_layout = copy_array_int(input_tensor->block_layout, tensor->dim);

	tensor->lin.size = input_tensor->lin.size;
	// arrange the data (+ init the array)
	blockmodemajor_array_int(&tensor->lin, input_tensor, mode);

	return tensor;
}

// this one takes an exisitng tensor, and creates a new one from it
struct tensor_storage *
get_blockmode_tensor(const struct tensor_storage * input_tensor, const size_t mode, const int unblock) {
//et_block_tensor(const struct tensor_storage * input_tensor, const int unblock, const int morton) {
	struct tensor_storage * tensor = calloc(1, sizeof(struct tensor_storage));
	tensor->dim = input_tensor->dim;

	tensor->layout = copy_array_int(input_tensor->layout, tensor->dim);

	tensor->layout_perm = copy_array_int(input_tensor->layout_perm, tensor->dim);

	// malloc block_layout and set to a value
	//tensor->block_layout = get_array_int(tensor->dim, block_size);
	tensor->block_layout = copy_array_int(input_tensor->block_layout, tensor->dim);

	tensor->lin.size = input_tensor->lin.size;
	// arrange the data (+ init the array)
	blockmode_array_int(&tensor->lin, input_tensor, mode, unblock);
	
	return tensor;
}

#if 0

// this one takes an exisitng tensor, and creates a new one from it
struct tensor_storage *
get_blockmode_tensor(const struct tensor_storage * input_tensor, const size_t mode) {
	struct tensor_storage * tensor = calloc(1, sizeof(struct tensor_storage));
	tensor->dim = input_tensor->dim;

	tensor->layout = copy_array_int(input_tensor->layout, tensor->dim);
	tensor->layout_perm = copy_array_int(input_tensor->layout_perm, tensor->dim);
	//print_to_console(input_tensor->layout, tensor->dim);

	// malloc block_layout and set to a value
	tensor->block_layout = copy_array_int(input_tensor->block_layout, tensor->dim);

	tensor->lin.size = input_tensor->lin.size;
	// arrange the data (+ init the array)
	blockmode_array_int(&tensor->lin, input_tensor, mode);

	return tensor;
}
#endif

struct tensor_storage *
get_unfold(const struct tensor_storage * input_tensor, const size_t mode) {

	struct tensor_storage * tensor = calloc(1, sizeof(struct tensor_storage));
	tensor->dim = input_tensor->dim;
	tensor->lin.size = input_tensor->lin.size;

	tensor->layout = copy_array_int(input_tensor->layout, input_tensor->dim);

	//print_to_console_sizet(input_tensor->layout_perm, tensor->dim);
	tensor->layout_perm = get_lapack_layout_perm(input_tensor->dim, input_tensor->layout_perm, mode);
	//print_to_console_sizet(tensor->layout_perm, tensor->dim);
	// then, all is left is to move the last to last-1

	// arrange the data (+ init the array)
	unfold_array_int(&tensor->lin, tensor->layout_perm, input_tensor);

	return tensor;	
}

struct tensor_storage *
get_unfold_row(const struct tensor_storage * input_tensor, const size_t mode) {
	struct tensor_storage * tensor = calloc(1, sizeof(struct tensor_storage));
	tensor->dim = input_tensor->dim;
	tensor->lin.size = input_tensor->lin.size;
	tensor->layout = copy_array_int(input_tensor->layout, input_tensor->dim);
	tensor->layout_perm = get_blas_layout_perm(input_tensor->dim, input_tensor->layout_perm, mode);
	unfold_array_int(&tensor->lin, tensor->layout_perm, input_tensor);
	return tensor;	
}

struct lin_storage *
gen_vector(const size_t vector_size) {
	struct lin_storage * vector = calloc(1, sizeof(struct lin_storage));
	vector->size = vector_size;
	set_seed(VECTOR_SEED);
	// #if (TEST_ENV == 0)
	// 	gen_array_double_stochastic(vector, 0);
	// #else
	gen_array_double(vector);
	// #endif
	return vector;
}

struct lin_storage *
gen_vector_interleaved(const size_t vector_size) {
	struct lin_storage * vector = calloc(1, sizeof(struct lin_storage));
	vector->size = vector_size;
	set_seed(VECTOR_SEED);
	vector->data = numa_alloc_interleaved(vector->size * sizeof(DTYPE));
	if (print_status("gen_array_double", vector->data)) {
		for (size_t i=0; i<vector->size; ++i) {
			#if (TEST_ENV == 1)
			vector->data[i] = round(rand_double()*10);
			#else
			vector->data[i] = rand_double()*10;
			#endif
		}
	}
	return vector;
}

double
normalize(const struct lin_storage * vector, const size_t norm_limit) {

	// printf("normalizing\n");

	// calculate its size
	double current_size = 0;
	for (size_t i=0; i<norm_limit; ++i) {
		current_size += vector->data[i]*vector->data[i];
	}
	current_size = sqrt(current_size);
	// printf("current_size =%f\n", current_size);
	
	double new_size = 0;
	for (size_t i=0; i<norm_limit; ++i) {
		vector->data[i] = vector->data[i]/current_size;
		new_size += vector->data[i]*vector->data[i];
	}
	// printf("new_size =%f\n", new_size);
	new_size = sqrt(new_size);
	// printf("new_size =%f\n", new_size);
	return new_size;
}

void
normalize_rows(const struct tensor_storage * tensor, const size_t mode) {
	const size_t column_size = tensor->layout[mode]; // Already sensitive
	const size_t row_size = tensor->lin.size / column_size;
	// here: possibly we need to simply multiply layouts to get the tensor "column"
	// GO over columns and rows (2 loops!)
	size_t el = 0;
	for (size_t row=0; row<row_size; ++row) {
		// printf("normalize_rows: before: (row %zu of size %zu): ", row, column_size);
		// print_to_console(tensor->lin.data+el, column_size);
		int ones_in_row = 0;
		for (size_t column=0; column<column_size; ++column) {
			if (tensor->lin.data[el++] == 1) {
				++ones_in_row;
			}
			// printf("el is increased, %zu\n", el);
		}
		el -= column_size;

		if (ones_in_row != 0) {
			// printf("el is %zu, ones in row are %d\n", el, ones_in_row);
			double new_entry = 1.0/(double) ones_in_row;
			// printf("new entry is %f\n", new_entry);
			for (size_t column=0; column<column_size; ++column) {
				if (tensor->lin.data[el] == 1) {
					tensor->lin.data[el] = new_entry;
				}
				++el;
				// printf("el is increased %zu\n", el);
			}
			// printf("el is %zu\n", el);
		}
	}
}

struct lin_storage *
gen_vector_seeded(const size_t vector_size, int seed) {
	struct lin_storage * vector = calloc(1, sizeof(struct lin_storage));
	vector->size = vector_size;
	set_seed(VECTOR_SEED + seed*10);
	gen_array_double(vector);
	return vector;
}

struct lin_storage *
gen_vector_seeded_safe(const size_t vector_size, int seed, const int local_allocation) {
	struct lin_storage * vector = calloc(1, sizeof(struct lin_storage));
	vector->size = vector_size;
	set_seed(VECTOR_SEED + seed*10);

	switch (local_allocation) {
		case 0: // both 
			printf("INFO: Interleaved allocation\n");
			gen_array_double(vector);
			// vector->data = numa_alloc_interleaved(vector->size * sizeof(DTYPE));
			assert(vector->data);
		case 1: // locally allocated
			printf("INFO: Local allocation\n");
 			gen_array_double_locally(vector);
 			break;
 		case 2: // interleaved only
			printf("INFO: Interleaved allocation\n");
			gen_array_double(vector);
			// vector->data = numa_alloc_interleaved(vector->size * sizeof(DTYPE));
			assert(vector->data);
			break;
 	}

	return vector;
}

struct lin_storage *
get_vector(const size_t vector_size) {
	struct lin_storage * vector = calloc(1, sizeof(struct lin_storage));
	vector->size = vector_size;
	set_array_int(vector);
	return vector;
}

// Helper functions: free storage
void
free_lin_storage(struct lin_storage * lin) {
	free(lin->data);
	free(lin);
}

void
free_lin_storage_safe(struct lin_storage * lin, const int local_allocation) {

	switch (local_allocation) {
		case 0: // both
			// printf("Numa free\n");
			#ifdef SINGLESOCKET
			free(lin->data);
			#else
			numa_free(lin->data, lin->size * sizeof(DTYPE));
			#endif
		case 1:
			if (lin->local_data) {
				#pragma omp parallel
				{
					int tid = omp_get_thread_num();
					// printf("Local free (thread %d)\n", tid);
					free(lin->local_data[tid]);
				}
				free((void *)lin->local_data);
			}
			break;
		case 2:
			// printf("Numa free\n");
			#ifdef SINGLESOCKET
			free(lin->data);
			#else
			numa_free(lin->data, lin->size * sizeof(DTYPE));
			#endif
			break;
	}
	free(lin);

}

void
free_tensor_storage(struct tensor_storage * tensor) {
	//free_lin_storage(&lin->data)
	free(tensor->lin.data);
	free(tensor->layout);
	free(tensor->layout_perm);
	if (tensor->block_layout) {
		free(tensor->block_layout);
	}
	if (tensor->layout2) {
		free(tensor->layout2);
	}
	free(tensor);
}

void
free_tensor_storage_safe(struct tensor_storage * tensor, const int local_allocation) {

	// Smarter free: simply check if pointers are allocated before free'ing them (!)

	#if (TEST_ENV == 1) 
	if (tensor->lin.master_data) {
		#ifdef SINGLESOCKET
		free(tensor->lin.master_data);
		#else
		numa_free(tensor->lin.master_data, tensor->lin.size * sizeof(DTYPE));
		#endif
	}
	#endif

	if (tensor->lin.data) {
		printf("Deallocating a global block of memory.\n");
		#ifdef SINGLESOCKET
		free(tensor->lin.data);
		#else
		numa_free(tensor->lin.data, tensor->lin.size * sizeof(DTYPE));
		#endif
	}
	if (tensor->lin.local_data) {
		printf("Deallocating local blocks of memory.\n");
		#pragma omp parallel
		{
			int tid = omp_get_thread_num();
			// printf("Deallocating part %d\n", tid);
			free(tensor->lin.local_data[tid]);
		}
		free((void *) tensor->lin.local_data);
	}

	// switch (local_allocation) {
	// 	case 0: // both
	// 		// printf("Free a numa pointer.\n");
	// 		numa_free(tensor->lin.data, tensor->lin.size * sizeof(DTYPE));
	// 	case 1: // local allocation
	// 		if (tensor->lin.local_data) {
	// 			#pragma omp parallel
	// 			{
	// 				int tid = omp_get_thread_num();
	// 				// printf("Free of a local pointer (thread %d)\n", tid);
	// 				free(tensor->lin.local_data[tid]);
	// 			}
	// 			// printf("Free of a local pointer array\n");
	// 			free((void *) tensor->lin.local_data);
	// 		}
	// 		break;
	// 	case 2: // interleaved allocation
	// 		// printf("Free a numa pointer.\n");
	// 		numa_free(tensor->lin.data, tensor->lin.size * sizeof(DTYPE));
	// 		break;
	// }

	// free_lin_storage_safe(&tensor->lin, local_allocation);
	free(tensor->layout);
	free(tensor->layout_perm);
	if (tensor->block_layout) {
		free(tensor->block_layout);
	}
	if (tensor->layout2) {
		free(tensor->layout2);
	}
	free(tensor);
}
