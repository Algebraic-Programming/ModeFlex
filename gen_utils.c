#include <structures.h>
#include <rand_utils.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <space_curves.h>
#include <hilbert.h>
#include <file_utils.h>
#include <assert.h>
#include <omp.h>
#include <numa.h>

void
round_numbers(DTYPE * const array, const size_t size) {
	for (size_t i=0; i<size; ++i) {
		array[i] = (int) array[i];
	}
}

// ALL *_tensor_layout functions: 
// Requires: dim
// Initializes fields: layout, layout_perm, lin.size
void
gen_tensor_layout(struct tensor_storage * tensor, size_t approximate_size) {
	// Dynamic allocated array to store size of each dim
	tensor->layout = malloc(tensor->dim * sizeof(size_t)); 
	tensor->layout_perm = malloc(tensor->dim * sizeof(size_t));

	size_t approximate_dim_size = (size_t) pow(approximate_size, 1.0/tensor->dim);
	printf("approximate_(tensor)_size: %ld\n", approximate_size);
	tensor->lin.size = 1;

	for (size_t i=0; i<tensor->dim; ++i) {
		size_t dim_size = rand_int(1, approximate_dim_size);
		tensor->layout[i] = dim_size;
		tensor->layout_perm[i] = i; // Assume zero-permutation
		tensor->lin.size = (tensor->lin.size) * dim_size;
		//printf("tensor.size: %ld\n", tensor->lin.size);
		if (approximate_size != 0) // Do not divide by zero 
			approximate_size /= dim_size;
	}
}

void
set_tensor_layout(struct tensor_storage * tensor, const size_t * layout) {

	tensor->layout = malloc(tensor->dim * sizeof(size_t));
	tensor->layout_perm = malloc(tensor->dim * sizeof(size_t));

	// copy the array of layout into tensor->layout
	memcpy(tensor->layout, layout, tensor->dim * sizeof(size_t));

	// compute the tensor->lin.size (and assign layout_perm)
	// printf("tensor dim is %zu\n", tensor->dim);
	tensor->lin.size = 1;
	for (size_t i=0; i<tensor->dim; ++i) {
		tensor->layout_perm[i] = i; // Assume zero-permutation
		tensor->lin.size = (tensor->lin.size) * tensor->layout[i];
		// printf("tensor.size: %ld\n", tensor->lin.size);
	}
}

// layout_perm functions
size_t *
get_in_out_layout_perm(const size_t dim, const size_t const * layout_perm) {
	size_t * temp_layout_perm = malloc(dim * sizeof(size_t));	
	// copy the array of layout into unfold->layout_perm (with permutation)
	memcpy(&(temp_layout_perm)[1], &layout_perm[0], (dim-1) * sizeof(size_t));
	// correct (outermost dim is the original innermost dim)
	temp_layout_perm[0] = layout_perm[dim-1];
	return temp_layout_perm;
}

// layout_perm functions
size_t *
get_lapack_layout_perm(const size_t dim, const size_t const * layout_perm, const size_t mode) {
	size_t * temp_layout_perm = malloc(dim * sizeof(size_t));	

	// copy all!
	memcpy(&(temp_layout_perm)[0], &layout_perm[0], (dim) * sizeof(size_t));

	for (size_t i=0; i<dim; ++i) {
		if (i == dim-2) {
			temp_layout_perm[i] = mode;
		} else if (i > dim-2) {
			// last element!!!
			if (i==mode) {
				temp_layout_perm[i] = layout_perm[i]-1;
			}
		} else if (i>=mode) {
			temp_layout_perm[i] = layout_perm[i+1];
		}
	}
	//print_to_console_sizet(temp_layout_perm, dim);
	return temp_layout_perm;
}

size_t *
get_blas_layout_perm(const size_t dim, const size_t const * layout_perm, const size_t mode) {
	size_t * temp_layout_perm = malloc(dim * sizeof(size_t));	
	memcpy(&(temp_layout_perm)[0], &layout_perm[0], (dim) * sizeof(size_t));
	for (size_t i=0; i<dim; ++i) {
		if (i == dim-1) {
			temp_layout_perm[i] = mode;
		} else if (i > dim-1) {
			// last element!!!
			if (i==mode) {
				temp_layout_perm[i] = layout_perm[i]-1;
			}
		} else if (i>=mode) {
			temp_layout_perm[i] = layout_perm[i+1];
		}
	}
	return temp_layout_perm;
}

// Prints malloc status
int
print_status(const char * fun, const void * ptr) {
	if (ptr) {
		printf("%s alloc successful\n", fun);
		return 1;
	} else {
		printf("%s alloc failed\n", fun);
		exit(1);
		return 0;
	}
}

// Helper functions: alloc, gemerate, reset


// set_array_int: gets a zero'ed array of some size
// get_array_int: sets an array (to a single value) of some size 
void
reset_array(DTYPE * array, const size_t size, const DTYPE new_value) {

	// printf("WE ARE HERE!!!!!!!!\n");
	
	for (size_t i=0; i<size; ++i) {
		array[i] = new_value;
	}
	// print_to_console(array, size);
	//print_status("set_array_int", (void*)storage->data);
}

void
reset_array_sizet(size_t * const array, const size_t size, const size_t new_value) {
	for (size_t i=0; i<size; ++i) {
		array[i] = new_value;
	}
	//print_status("set_array_int", (void*)storage->data);
}

void
set_array_int(struct lin_storage * storage) {
	storage->data = calloc(storage->size, sizeof(DTYPE));
	print_status("set_array_int", (void*)storage->data);
}

void
randomize_array_sizet(size_t * array, const size_t size, const size_t max_value) {
	for (size_t i=0; i<size; ++i) {
		array[i] = (size_t) rand_int(1, max_value);
	}
}

void
randomize_array_int(size_t * array, const size_t size, const size_t max_value) {
	for (size_t i=0; i<size; ++i) {
		array[i] = rand_int(1, max_value);
	}
}

void
randomize_array_int_from_array(size_t * array, const size_t size, const size_t * array_input) {
	for (size_t i=0; i<size; ++i) {
		array[i] = rand_int(1, array_input[i]);
	}
}

size_t *
get_array_int(const size_t size, const size_t val) {
	size_t * temp = malloc(size * sizeof(size_t));
	for (size_t i=0; i<size; ++i) {
		temp[i] = val;
	}
	return temp;
}

DTYPE *
get_aligned_memory(size_t size, size_t alignment) {
	DTYPE * mem;
	int r = posix_memalign((void **)&mem, alignment, size);
	if (r != 0) {
		printf("ERROR: cannot allocate memory of size %zu.\n", size);
		exit(-1);
	}
	return mem;
}

// FUNCTIONS BELOW ARE dtype sensitive

void
gen_array_int(struct lin_storage * storage) {
	printf("INFO: Allocating some storage using only memalloc!\n");
	storage->data = get_aligned_memory(storage->size * sizeof(DTYPE), ALIGNMENT);
	if (print_status("gen_array_int", storage->data)) {
		for (size_t i=0; i<storage->size; ++i) {
			// DTYPE sensitive line
			storage->data[i] = rand_int(2,10);
		}
	}
}

void
gen_array_double(struct lin_storage * storage) {
	
	#ifdef SINGLESOCKET
		printf("Single socket; Use a simple malloc!\n");
		storage->data = get_aligned_memory(storage->size * sizeof(DTYPE), ALIGNMENT);
	#else
		printf("Using all sockets; Use interleaved allocation!\n");
		storage->data = numa_alloc_interleaved(storage->size * sizeof(DTYPE));
	#endif

	if (print_status("gen_array_double", storage->data)) {
		for (size_t i=0; i<storage->size; ++i) {
		#if(TEST_ENV == 1)
			storage->data[i] = round(rand_double()*10);
		#else
			storage->data[i] = rand_double()*10;
		#endif
		}
	}
}

// Careful: This assumes perfect distirbution (can backfire at some point)
void
reset_array_double_locally(struct lin_storage * const storage) {
	// code mostly taken from https://stackoverflow.com/questions/27199255/openmp-ordering-critical-sections
	#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		int nthreads = omp_get_num_threads();
		size_t partition_size = storage->size / nthreads;
		// printf("partition_size = %zu / %d = %zu\n", storage->size, nthreads, storage->size / nthreads); 
		
		#pragma omp for ordered schedule(static,1)
	    for (int t=0; t<nthreads; ++t)
	    {
	        assert( t==tid );
	        #pragma omp ordered
	        {
	        	// printf("Thread %d resetting output partition of size %zu\n", tid, partition_size);
	        	reset_array(storage->local_data[tid], partition_size, 0.0);
	        }
        }
    }
    // printf("Code finished successfully\n");
}

void
gen_array_double_locally(struct lin_storage * const storage) {
	// code mostly taken from https://stackoverflow.com/questions/27199255/openmp-ordering-critical-sections
	#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		int nthreads = omp_get_num_threads();

		#pragma omp single
		{
			storage->local_data = malloc(nthreads * sizeof(storage->local_data));
			storage->p_size = nthreads;
			printf("INFO: storage->p_size = %d\n", nthreads);
		}

		#pragma omp for ordered schedule(static,1)
	    for (int t=0; t<omp_get_num_threads(); ++t)
	    {
	        assert( t==tid );
	        #pragma omp ordered
	        {
	        	size_t partition_size = storage->size / nthreads;
	        	storage->local_data[tid] = get_aligned_memory(partition_size * sizeof(DTYPE), ALIGNMENT);
				for (size_t el=0; el<partition_size; ++el) {
					if (storage->data) {
						storage->local_data[tid][el] = storage->data[(tid*partition_size) + el]; // round(rand_double()*10);
					} else {
						#if (TEST_ENV==1)
						storage->local_data[tid][el] = round(rand_double()*10);
						#else
						storage->local_data[tid][el] = rand_double()*10;
						#endif
					}
				}
				printf("INFO: Partition storage->local_data referencing storage->data[%zu : %zu] locally allocated by thread %d...\n", (tid*partition_size), ((tid+1)*partition_size)-1, tid);
	        }
	    }
	}

}

// Values of tensor A between 0 and 1
void
gen_array_double_stochastic_old(struct lin_storage * storage) {
	storage->data = get_aligned_memory(storage->size * sizeof(DTYPE), ALIGNMENT);
	const double alpha = rand_double(); // between 0 and 1
	const double value_e = 1-alpha; // (1-alpha)E but E is a matrix full of 1s
	if (alpha == 0.0 || alpha == 1.0) {
		printf("Alpha has a value which is not permissible!\n");
		exit(-1);
	}
	if (print_status("gen_array_double_stochastic", storage->data)) {
		for (size_t i=0; i<storage->size; ++i) {
			// DTYPE sensitive line
			storage->data[i] = (alpha * rand_double()) + value_e;
		}
	}
}

void
gen_array_double_stochastic(struct lin_storage * storage, const size_t row_size) {
	(void) row_size;
	storage->data = get_aligned_memory(storage->size * sizeof(DTYPE), ALIGNMENT);
	if (print_status("gen_array_double_stochastic", storage->data)) {
		for (size_t i=0; i<storage->size; ++i) {
			storage->data[i] = rand_binary();
		}
	}
}	
	// storage->data = get_aligned_memory(storage->size * sizeof(DTYPE), ALIGNMENT);
	// size_t el = 0;
	// if (print_status("gen_array_double_stochastic_PAGERANK", storage->data)) {
	// 	for (size_t rows=0; rows<(storage->size/row_size); ++rows) {
	// 		printf("First row:\n");
	// 		// Generate in advance the number of 1s in a row
	// 		int number_of_ones = 0;
	// 		for (size_t entry=0; entry<row_size; ++entry) {
	// 			// We generate the matrix element by element
	// 			const int gen_entry = rand_int_int(0,1);
	// 			// printf("Generated int: %d\n", gen_entry);
	// 			storage->data[++el] = gen_entry;
	// 			if (gen_entry == 1) {
	// 				++number_of_ones;
	// 				// printf("increasing number of ones to %d\n", number_of_ones);
	// 			} else {
	// 				// printf("nto touching number of ones\n");
	// 			}
	// 		}
	// 		printf("(number of ones is %d)\n", number_of_ones);
	// 		el -= row_size;
	// 		for (size_t entry=0; entry<row_size; ++entry) {
	// 			printf("what is wrong with this shit!@!!!, el=%zu (row_size=%zu)\n", el, row_size);
	// 			// Normalize the generated row(!)
	// 			storage->data[++el] = 1;
	// 		}
	// 		print_to_console(storage->data + (el-row_size), row_size);

	// 	}
	// }

size_t
get_size(const size_t * const array, const size_t dim) {
	size_t size = 1;
	for(size_t i=0; i<dim; i++) {
		size *= array[i];
	}
	return size;
}

size_t *
copy_array_int(const size_t * const array, const size_t sizing) {
	size_t * temp = malloc(sizing * sizeof(size_t));
	memcpy(temp, array, sizing * sizeof(size_t));
	return temp;
}

size_t *
copy_array_int_replace_mode(const size_t * const array, const size_t sizing, const size_t mode, const size_t new_dim) {
	// NOT as straightforward
	size_t * temp = malloc(sizing * sizeof(size_t));
	for (size_t p=0; p<sizing; ++p) {
		if (p>mode) {
			temp[p] = array[p];
		} else if (p==mode) {
			temp[p] = new_dim;
		} else {
			temp[p] = array[p];
		}
	}
	return temp;
}

void
fill_array_sizet_except_mode(const size_t * const array_in, size_t * const array_out, const size_t dim, const size_t mode) {
	// printf("dim = %zu, mode=%zu\n", dim, mode);
	for (size_t p = 0; p < dim-1; ++p) {
		if (p >= mode) {
			// printf("Accessing address p+1=%zu\n", p+1);
			array_out[p] = array_in[p+1];
		} else {
			array_out[p] = array_in[p];
		}
	}
}

size_t *
copy_array_int_except_mode(const size_t * const array, const size_t sizing, const size_t mode) {

	// printf("So we are copying (sizing:%zu)\n", sizing);
	// NOT as straightforward
	size_t * temp = malloc(sizing * sizeof(size_t));
	for (size_t p=0; p<sizing; ++p) {
		if (p>=mode) {
			temp[p] = array[p+1];
		} else {
			temp[p] = array[p];			
		}
	}

	// printf("Printing this array (withotu mode i suppose?)\n");
	// print_to_console_sizet(temp, sizing);
	return temp;
}

// UNFOLDING DATA FUNCTIONS (by copy)

// unfold: in_out
void
in_out_array_int(struct lin_storage * storage, const struct tensor_storage * restrict tensor, const size_t mode) {

	storage->data = malloc(storage->size * sizeof(DTYPE));
	size_t last_dim_index = tensor->dim-1;
	size_t * mul = malloc(tensor->dim * sizeof(size_t));
	size_t * diff = calloc(tensor->dim, sizeof(size_t));
	size_t next = 0;

	mul[last_dim_index] = 1;
	for (size_t i=last_dim_index; i!=0; --i) {
		mul[i-1] = mul[i] * tensor->layout[tensor->layout_perm[i]];
		diff[i] = mul[i-1] - mul[i];
		if (i==mode) break;
	}
	
	size_t vector_size = tensor->layout[mode];
	size_t right_size = mul[mode];
	size_t result_size = tensor->lin.size / vector_size;
	size_t stride = diff[mode];
	size_t tensor_diff = 0;
	size_t tensor_index = 0;

	for (size_t j=0; j<vector_size; ++j) {
		tensor_index = tensor_diff;
		for (size_t i=0; i<result_size; ++i) {
			if ((i != 0) & (i % right_size == 0)) {
				tensor_index += stride;
			}
			storage->data[next++] = tensor->lin.data[tensor_index+i];
		}
		tensor_diff += right_size;
	}
	free(diff);
	free(mul);
}

// unfold: out_in
void
out_in_array_int(struct lin_storage * storage, const struct tensor_storage * restrict tensor, const size_t mode) {

	storage->data = malloc(storage->size * sizeof(DTYPE));
	size_t last_dim_index = tensor->dim-1;
	size_t * mul = malloc(tensor->dim * sizeof(size_t));
	size_t * diff = calloc(tensor->dim, sizeof(size_t));
	size_t next = 0;

	mul[last_dim_index] = 1;
	for (size_t i=last_dim_index; i!=0; --i) {
		mul[i-1] = mul[i] * tensor->layout[tensor->layout_perm[i]];
		diff[i] = mul[i-1] - mul[i];
		if (i==mode) {
			break;
		}
	}
#if 0
	size_t test = 1;
	size_t d = 0;
	for (size_t i=last_dim_index; i>mode; --i) {
		test *= tensor->layout[tensor->layout_perm[i]];
	}
	if (mode != 0) {
		d = test * (tensor->layout[tensor->layout_perm[mode]] - 1);
	}
#endif

	size_t vector_size = tensor->layout[mode];
	size_t right_size = mul[mode];
	size_t result_size = tensor->lin.size / vector_size;
	size_t stride = diff[mode];

#if 0
	printf("their=%d, my=%d\n", diff[mode], d);
	printf("their=%d, my=%d\n", mul[mode], test);

	if (diff[mode] != d) {
		exit(-1);
	} else if (mul[mode] != test) {
		exit(-1);
	}
#endif

	size_t tensor_diff = 0;
	size_t tensor_index = 0;
	for (size_t i=0; i<result_size; ++i) {
		if ((i!=0) & (i % right_size == 0)) {
			tensor_diff += stride;
		}
		tensor_index = i + tensor_diff;
		for (size_t j=0; j<vector_size; ++j) {
			storage->data[next++] = tensor->lin.data[tensor_index];
			//result_tensor->data[i] += tensor->lin.data[tensor_index] * vector->data[j];
			tensor_index += mul[mode];
		}

	}
	free(diff);
	free(mul);
}

// unfold: arbitrary
void
unfold_array_int(struct lin_storage * storage, const size_t const * unfold_layout_perm, const struct tensor_storage * in_tensor) {
	storage->data = malloc(storage->size * sizeof(DTYPE));
	print_status("unfold_array_DTYPE", (void*) storage->data);
	
	// create a NORMAL multiplication table
	size_t last_dim_index = in_tensor->dim-1;	
	size_t * mul = malloc(in_tensor->dim * sizeof(size_t));
	size_t * counter = calloc(in_tensor->dim, sizeof(size_t));

	// recursively create mul table
	// init step
	mul[last_dim_index] = 1;
	// n-step	
	for (size_t i=last_dim_index; i!=0; --i) {
		mul[i-1] = mul[i] * in_tensor->layout[i];
	}
	
	size_t tensor_index = 0; // tensor_index: set to 0 in the beginning

	// increment counters like the dimensions change in the tensor
	for (size_t i=0; i<in_tensor->lin.size; ++i) {	
		
		//printf("tensor_index=%d\n", tensor_index);
		// COPY (unfold) OPERATION


		// DTYPE sensitive operation
		storage->data[i] = (DTYPE) in_tensor->lin.data[tensor_index];

		tensor_index = 0; // tensor_index: reset to 0 again
		// tick the smallest dimension (of weight 1)
		++counter[last_dim_index];
		// this loop carries the little tocks over
		for (size_t j=last_dim_index; j!=0; --j) {
			size_t original_index = unfold_layout_perm[j];
			// threshold reached on the lower dimension
			if (counter[j] == in_tensor->layout[original_index]) {
				//  increment the higher dimension
				++counter[j-1];
				// reset the lower dimension
				counter[j] = 0;
			}
			// compute the tensor_index on the go
			tensor_index += counter[j] * mul[original_index];
		}
		// compute the tensor_index on the go
		tensor_index += counter[0] * mul[unfold_layout_perm[0]];
	}

	free(counter);
	free(mul);

}

// This algorithm transforms a tensor into blocked storage
// No unfold: the inner and outer storage is the same as the tensor_layout
void
blockmode_array_int(struct lin_storage * storage, const struct tensor_storage const * tensor, const size_t mode, const size_t unblock) {

	storage->data = malloc(storage->size * sizeof(DTYPE));	
	print_status("block_array_int", (void*) storage->data);
	//printf("vector size =%d\n", tensor->lin.size);
	//printf("layout= ,mode=%d\n", mode);
	//print_to_console_sizet(tensor->layout, tensor->dim);
	//printf("vector size =%d\n", tensor->lin.size);

	size_t last_dim_index = tensor->dim-1;
	size_t * mul = malloc(tensor->dim * sizeof(size_t));
	mul[last_dim_index] = 1;

	size_t block_size = tensor->block_layout[0];
	size_t right_block_size = 1;
	for (size_t i=last_dim_index; i!=0; --i) {
		mul[i-1] = mul[i] * tensor->layout[tensor->layout_perm[i]];
		block_size *= tensor->block_layout[i];
		if (i>mode) {
			right_block_size *= tensor->block_layout[i];
			//printf("RBS=%d i=%d mode=%d \n", right_block_size, i, mode);
		}
	}
	//printf("block_size = %d\n", block_size);
	size_t vector_size = tensor->layout[mode];
	size_t right_size = mul[mode]; 
	//printf("SIEMA JESTESMY TUTAJ!!!1\n");
	//printf("v_s=%d, r_s=%d\n", vector_size, right_size);
	size_t left_size = tensor->lin.size / vector_size / right_size;
	size_t RBS = right_block_size;
	size_t vector_block_size = tensor->block_layout[mode];
	size_t VBS = vector_block_size;
	size_t left_block_size = block_size / vector_block_size / right_block_size;
	//printf("LBS=%d, VBS=%d, RBS=%d\n", left_block_size, VBS, RBS);
	size_t global_t = 0;
	size_t t = 0;
	size_t counter = 0;
	size_t out_offset = 0;
	size_t left_offset = 1;
	if (mode != 0) {
		left_offset = mul[mode-1];
	}

	size_t next = 0;

	//printf("Left_offset=%d\n", left_offset);
	// optimization variables
	size_t calc1 = vector_block_size * right_size;
	size_t last_ii = (left_size / left_block_size) * left_block_size;
	size_t last_vv = (vector_size / vector_block_size) * vector_block_size;
	size_t last_jj = (right_size / right_block_size) * right_block_size;

	//printf("calc1=%d, last_ii=%d, last_v=%d, last_j=%d\n", calc1, last_ii, last_vv, last_jj);
	for (size_t ii=0; ii<left_size; ii+=left_block_size) {
		//printf("ii=%d\n", ii);
		if (ii==last_ii) {
			//printf("here ii\n");
			left_block_size = left_size % ii;
		}
		global_t = 0;
		vector_block_size = VBS;
		for (size_t vv=0; vv<vector_size; vv+=vector_block_size) {
			//printf("vv=%d\n", vv);
			if (vv==last_vv) {
				//printf("vec_b_size=%d, vec_size=%d\n", vector_block_size, vector_size);
				//printf("here vv\n");
				vector_block_size = vector_size % vv;
			}
			right_block_size = RBS;
			for (size_t jj=0; jj<right_size; jj+=right_block_size) {
				//printf("jj=%d\n", jj);
				if (jj==last_jj) {
					//printf("here jj\n");
					right_block_size = right_size % jj;
				}
				out_offset = ii*right_size+jj;
				//printf("new block...\n");
				for (size_t i=0; i<left_block_size; ++i) {
					//printf("i=%d\n", i);
					t = global_t + (i+ii)*left_offset + jj;
					//printf("t = global_t + (i+ii)*left_offset + jj (%d = %d + (%d+%d)*%d + %d)\n",
							//t, global_t, i, ii, left_offset, jj);
					for (size_t v=0; v<vector_block_size; ++v) {
						//printf("v=%d\n", v);
						for (size_t j=0; j<right_block_size; ++j) {
							//if (counter:wq
							//printf("t=%d, j=%d\n", t,j);
							if (unblock == 0) {
								storage->data[next++] = tensor->lin.data[t+j];
							} else {

								storage->data[t+j] = tensor->lin.data[next++];
							}
							//printf("out=%d = tensor=%d * vec=%d\n", out_offset+j, t+j, v+vv);
							//;/result_tensor->data[out_offset+j] += 
								//tensor->lin.data[t+j] * vector->data[v+vv];
							counter++;	
						}
						t += right_size;
					}
					out_offset += right_size;
				}
			}
			global_t += calc1; // we won't enter the loop anyway (replaced global_t = vv*right_size (before for jj loop)
			//printf("GLOBAL T IS UPDATED\n");
		}
	}
	free(mul);











#if 0
	size_t last_dim_index = tensor->dim-1;
	size_t * mul = malloc(tensor->dim * sizeof(size_t));
	mul[last_dim_index] = 1;

	size_t block_size = tensor->block_layout[0];
	//printf("block_size=%d\n", block_size);
	size_t right_block_size = 1;
	for (size_t i=last_dim_index; i!=0; --i) {
		mul[i-1] = mul[i] * tensor->layout[tensor->layout_perm[i]];
		block_size *= tensor->block_layout[i];
		if (i>mode) {
			right_block_size *= tensor->block_layout[i];
			//printf("RBS=%d i=%d mode=%d \n", right_block_size, i, mode);
		}
		//if (i==mode) break; // we cannot break at i-1==mode... we need the other elements!
		//right_block_size *= block_layout[i];
	}

	size_t vector_size = tensor->layout[mode]; //vector->size;
	size_t right_size = mul[mode]; 
	size_t left_size = tensor->lin.size / vector_size / right_size;
	//printf("left=%d, mode=%d, right=%d\n", left_size, vector_size, right_size);
	
	// right_block_size is figured out above
	size_t RBS = right_block_size;
	size_t vector_block_size = tensor->block_layout[mode];
	size_t VBS = vector_block_size;
	size_t left_block_size = block_size / vector_block_size / right_block_size;
	
	size_t global_t = 0;
	size_t t = 0;
	size_t next = 0;

	size_t left_offset = 1;
	if (mode != 0) {
		left_offset = mul[mode-1];
	}

	// optimization variables
	size_t calc1 = vector_block_size * right_size;
	size_t last_ii = (left_size / left_block_size) * left_block_size;
	size_t last_vv = (vector_size / vector_block_size) * vector_block_size;
	size_t last_jj = (right_size / right_block_size) * right_block_size;

	for (size_t ii=0; ii<left_size; ii+=left_block_size) {
		if (ii==last_ii) {
			left_block_size = left_size % ii;
		}
		global_t = 0;
		vector_block_size = VBS;
		//printf("\nleft_block_size ====== %d\n", left_block_size);
		for (size_t vv=0; vv<vector_size; vv+=vector_block_size) {
			if (vv==last_vv) {
				vector_block_size = vector_size % vv;
			}
			right_block_size = RBS;
			//printf("\nvector_block_size ====== %d\n", vector_block_size);
			// this is the loop which will cover all contiguous blocks (currently in memory) in unfold storage
			for (size_t jj=0; jj<right_size; jj+=right_block_size) {
				if (jj==last_jj) {
					right_block_size = right_size % jj;
				}
				for (size_t i=0; i<left_block_size; ++i) {
					t = global_t + (i+ii)*left_offset + jj;
					for (size_t v=0; v<vector_block_size; ++v) {
						for (size_t j=0; j<right_block_size; ++j) {
							//printf("block_unfold(%d) tensor(%d)\n", next, t+j);
							storage->data[next++] = tensor->lin.data[t+j];
							// we don't need out_offset, v and vv
						}
						t += right_size;
					}
				}
			}
			global_t += calc1; // we won't enter the loop anyway (replaced global_t = vv*right_size (before for jj loop)
		}
	}	

	free(mul);
#endif
}


void
block_array_int(struct lin_storage * out_tensor, const struct tensor_storage const * tensor, const size_t unblock) {

	// printf("inside block array int\n"); 
	out_tensor->data = malloc(out_tensor->size * sizeof(DTYPE));	

	size_t dim = tensor->dim;
	size_t * block_counter_threshold = calloc(dim, sizeof(size_t));
	size_t * block_counter = calloc(dim, sizeof(size_t));
	size_t * remainder_index = calloc(dim, sizeof(size_t));
	size_t * limits = calloc(dim, sizeof(size_t));
	size_t * block_mul = calloc(dim, sizeof(size_t));

	size_t blocks = 0;
	size_t block_size = 1;
	if (dim-1 < dim) {
		blocks = 1;
		block_mul[dim-1] = 1;
	} else {
		// actually, this line should be true no matter what!!!
		out_tensor->data[0] = tensor->lin.data[0];
	}

	for (size_t i=dim-1; i<=dim-1; --i) {
		if (i!=0) {
			// printf("We are here?\n");
			// printf("first: %zu\n", block_mul[i]);
			// print_to_console_sizet(tensor->layout, dim);
			// printf("second: %zu\n", tensor->layout[tensor->layout_perm[i]]);
			block_mul[i-1] = block_mul[i] * tensor->layout[tensor->layout_perm[i]];
		}
		// integer division with rounding up formula :)
		block_counter_threshold[i] = (tensor->layout[i] + tensor->block_layout[i] -1)
			/ tensor->block_layout[i];
		blocks *= block_counter_threshold[i];
		block_size *= tensor->block_layout[i];
		if (tensor->layout[i] % tensor->block_layout[i] != 0) {
			remainder_index[i] = tensor->layout[i] % tensor->block_layout[i];
		} else {
			remainder_index[i] = tensor->block_layout[i];
			// when it's not an "edgy" block-dim then just assign regular size
		}
	}

	//size_t result_index = 0;
	size_t tensor_index = 0;
	size_t next = 0;
	//size_t result_offset = 0;
	size_t tensor_offset = 0;
	//size_t vector_index = 0;
	// BLOCK-LEVEL LOOP (using counter method)
	for (size_t b=0; b<blocks; ++b) {
		// printf("blocks=%zu\n", blocks);
		//new_mul[dim-1] = 1;
		for (size_t d=dim-1; d<=dim-1; --d) {
			if (block_counter_threshold[d]-1 == block_counter[d]) {
				limits[d] = remainder_index[d];
			} else {
				limits[d] = tensor->block_layout[d];
			}
		}
		// TENSOR-LEVEL LOOP
	 	//vector_index = 0;
		size_t * counter = calloc(dim, sizeof(size_t));
		for (size_t t=0; t<block_size; ++t) {
			// printf("blocksize=%zu\n", block_size);
			if (unblock == 0) {
				// means we want to block
				out_tensor->data[next++] = tensor->lin.data[tensor_index+tensor_offset];
			} else {
					// printf("so, ndex=%zu, equals index= %zu\n", tensor_index+tensor_offset, next);

				out_tensor->data[tensor_index+tensor_offset] = tensor->lin.data[next++];
				// printf("tensor_index=%zu, tensor_offset=%zu\n", tensor_index, tensor_offset);
			}
			tensor_offset = 0;
			// should be protected 
			++counter[dim-1];
			for (size_t d=dim-1; d!=0; --d) {
				// two conditions to tick the counter
				// 1) it simply reaches the threshold
				// 2) it reaches the limit for this part dimension
				if (counter[d] == limits[d]) {
					if (d!=0) {
						++counter[d-1];
					}
					counter[d] = 0;
				}
				tensor_offset += counter[d] * block_mul[d];
			}

			// handle the 0 case
			if (counter[0] == limits[0]) {
				break;
			}
			tensor_offset += counter[0] * block_mul[0];
		}

		tensor_index = 0;
		++block_counter[dim-1];
		for (size_t d=dim-1; d<=dim-1; --d) {
			if (block_counter[d] == block_counter_threshold[d]) {
				if (d!=0) {
					++block_counter[d-1];
				}
				block_counter[d] = 0;
			}
			tensor_index += block_counter[d] * block_mul[d] * tensor->block_layout[d];
		}

		free(counter);
	}

	free(limits);
	free(block_mul);
	free(remainder_index);
	free(block_counter);
	free(block_counter_threshold);
}

void
blockmodemajor_array_int(struct lin_storage * blockmode_tensor, const struct tensor_storage const * tensor, const size_t mode) {

	blockmode_tensor->data = malloc(blockmode_tensor->size * sizeof(DTYPE));	

	size_t last_dim_index = tensor->dim-1;
	size_t * mul = malloc(tensor->dim * sizeof(size_t));
	mul[last_dim_index] = 1;

	size_t block_size = tensor->block_layout[0];
	size_t right_block_size = 1;
	for (size_t i=last_dim_index; i!=0; --i) {
		mul[i-1] = mul[i] * tensor->layout[tensor->layout_perm[i]];
		block_size *= tensor->block_layout[i];
		if (i>mode) {
			right_block_size *= tensor->block_layout[i];
			//printf("RBS=%d i=%d mode=%d \n", right_block_size, i, mode);
		}
	}
	//printf("block_size = %d\n", block_size);
	size_t vector_size = tensor->layout[mode];
	size_t right_size = mul[mode]; 
	size_t left_size = tensor->lin.size / vector_size / right_size;
	
	size_t RBS = right_block_size;
	size_t vector_block_size = tensor->block_layout[mode];
	size_t VBS = vector_block_size;
	size_t left_block_size = block_size / vector_block_size / right_block_size;
	//printf("LBS=%d, VBS=%d, RBS=%d\n", left_block_size, VBS, RBS);

	size_t global_t = 0;
	size_t t = 0;
	size_t next = 0;
	size_t out_offset = 0;
	size_t left_offset = 1;
	if (mode != 0) {
		left_offset = mul[mode-1];
	}
	// optimization variables
	size_t calc1 = vector_block_size * right_size;
	size_t last_ii = (left_size / left_block_size) * left_block_size;
	size_t last_vv = (vector_size / vector_block_size) * vector_block_size;
	size_t last_jj = (right_size / right_block_size) * right_block_size;
	//printf("calc1=%d, last_ii=%d, last_v=%d, last_j=%d\n", calc1, last_ii, last_vv, last_jj);

	for (size_t ii=0; ii<left_size; ii+=left_block_size) {
		//printf("ii=%d\n", ii);
		if (ii==last_ii) {
			left_block_size = left_size % ii;
		}
		right_block_size = RBS;
		for (size_t jj=0; jj<right_size; jj+=right_block_size) {
			//printf("jj=%d\n", jj);
			if (jj==last_jj) {
				right_block_size = right_size % jj;
			}

			global_t = 0;
			vector_block_size = VBS;
			for (size_t vv=0; vv<vector_size; vv+=vector_block_size) {
				//printf("vv=%d\n", vv);
				if (vv==last_vv) {
					vector_block_size = vector_size % vv;
				}	
				out_offset = ii*right_size+jj;
				for (size_t i=0; i<left_block_size; ++i) {
					t = global_t + (i+ii)*left_offset + jj;
					for (size_t v=0; v<vector_block_size; ++v) {
						for (size_t j=0; j<right_block_size; ++j) {
							//printf("block: vec=%d, out=%d\n", v+vv, j+out_offset);
							blockmode_tensor->data[next++] = tensor->lin.data[t+j];
							//result_tensor->data[out_offset+j] += 
								//tensor->lin.data[t+j] * vector->data[v+vv];
						}
						t += right_size;
					}
					out_offset += right_size;
				}
				global_t += calc1; // we won't enter the loop anyway (replaced global_t = vv*right_size (before for jj loop)
			}
		}
	}
	free(mul);
}

void
morton_block_array_int(struct lin_storage * out_tensor, const struct tensor_storage const * tensor, const size_t unblock) {

	out_tensor->data = malloc(out_tensor->size * sizeof(DTYPE));

	size_t dim = tensor->dim;
	// counters
	size_t * block_counter = calloc(dim, sizeof(size_t));
	// limits
	size_t * block_counter_threshold = calloc(dim, sizeof(size_t));
	size_t * remainder_index = calloc(dim, sizeof(size_t));
	size_t * limits = calloc(dim, sizeof(size_t));
	// calcs
	size_t * block_mul = calloc(dim, sizeof(size_t));
	//size_t * new_mul = calloc(dim, sizeof(size_t));

	size_t blocks = 0;
	size_t block_size = 1;
	if (dim-1 < dim) {
		blocks = 1;
		block_mul[dim-1] = 1;
	} else {
		// actually, this line should be true no matter what!!!
		out_tensor->data[0] = tensor->lin.data[0];
	}
	for (size_t i=dim; i!=0; --i) {
		if (i-1 != 0) {
			block_mul[i-2] = block_mul[i-1] * tensor->layout[tensor->layout_perm[i-1]];
		}
		// integer division with rounding up formula :)
		block_counter_threshold[i-1] = (tensor->layout[i-1] + tensor->block_layout[i-1] -1)
			/ tensor->block_layout[i-1];
		blocks *= block_counter_threshold[i-1];
		block_size *= tensor->block_layout[i-1];
		if (tensor->layout[i-1] % tensor->block_layout[i-1] != 0) {
			remainder_index[i-1] = tensor->layout[i-1] % tensor->block_layout[i-1];
		} else {
			remainder_index[i-1] = tensor->block_layout[i-1];
			// when it's not an "edgy" block-dim then just assign regular size
		}
	}
	
	size_t next = 0;
	size_t tensor_index = 0;
	size_t tensor_offset = 0;
	//print_to_console_sizet(block_counter_threshold, dim);
	// BLOCK-LEVEL LOOP (using counter method)
	for (size_t b=0; b<blocks; ++b) {

 		for (size_t d=dim-1; d<=dim-1; --d) {
			if (block_counter_threshold[d]-1 == block_counter[d]) {
				limits[d] = remainder_index[d];
			} else {
				limits[d] = tensor->block_layout[d];
			}
		}
		// TENSOR-LEVEL LOOP
		size_t * counter = calloc(dim, sizeof(size_t));
		for (size_t t=0; t<block_size; ++t) {

			if (unblock == 0) {
				out_tensor->data[next++] = tensor->lin.data[tensor_index+tensor_offset];
			} else {
				out_tensor->data[tensor_index+tensor_offset] = tensor->lin.data[next++];
			}

			tensor_offset = 0;
			++counter[dim-1];
			for (size_t d=dim-1; d!=0; --d) {
				// two conditions to tick the counter
				// 1) it simply reaches the threshold
				// 2) it reaches the limit for this part dimension
				if (counter[d] == limits[d]) {
					if (d!=0) {
						++counter[d-1];
					}
					counter[d] = 0;
				}
				tensor_offset += counter[d] * block_mul[d];
			}
			// handle the 0 case
			if (counter[0] == limits[0]) {
				break;
			}
			tensor_offset += counter[0] * block_mul[0];
		}

		tensor_index = 0;

		// this line affects all dimensions at once
		morton_inc(block_counter, block_counter_threshold, dim-1);	
		//print_to_console_sizet(block_counter, dim);
		//printf("\n");
		// morton_inc in fact you can increment at most only one dimension
	
		// if you reset a dimension which is not a mode
		// i.e. we go back but in fact we do NOT compute completely different part of result because of reset
		// but we compute the same part of answer!

		// (but the same!) -> we must reset the result_index
		// if dimension is mul of 2: reset is undetected
		// if dimension is not mul of 2: we know exactly when
		
		// brute force: recalculate each time (I think necessary!)
		for (size_t d=dim-1; d<=dim-1; --d) {
			tensor_index += block_counter[d] * block_mul[d] * tensor->block_layout[d];
		}
		free(counter);

	}

	free(limits);
	free(block_mul);
	free(remainder_index);
	free(block_counter);
	free(block_counter_threshold);

}

void
hilbert_block_array_int(struct lin_storage * out_tensor, const struct tensor_storage const * tensor, const size_t unblock) {

	// printf("Welcome to Hilbert curve TV algorithm because why not.\n");
	
	out_tensor->data = malloc(out_tensor->size * sizeof(DTYPE));

	size_t dim = tensor->dim;
	// counters
	size_t * block_counter = calloc(dim, sizeof(size_t));
	// limits
	size_t * block_counter_threshold = calloc(dim, sizeof(size_t));
	size_t * remainder_index = calloc(dim, sizeof(size_t));
	size_t * limits = calloc(dim, sizeof(size_t));
	// calcs
	size_t * block_mul = calloc(dim, sizeof(size_t));
	//size_t * new_mul = calloc(dim, sizeof(size_t));

	size_t blocks = 0;
	size_t block_size = 1;
	if (dim-1 < dim) {
		blocks = 1;
		block_mul[dim-1] = 1;
	} else {
		// actually, this line should be true no matter what!!!
		out_tensor->data[0] = tensor->lin.data[0];
	}
	for (size_t i=dim; i!=0; --i) {
		if (i-1 != 0) {
			block_mul[i-2] = block_mul[i-1] * tensor->layout[tensor->layout_perm[i-1]];
		}
		// integer division with rounding up formula :)
		block_counter_threshold[i-1] = (tensor->layout[i-1] + tensor->block_layout[i-1] -1)
			/ tensor->block_layout[i-1];
		blocks *= block_counter_threshold[i-1];
		block_size *= tensor->block_layout[i-1];
		if (tensor->layout[i-1] % tensor->block_layout[i-1] != 0) {
			remainder_index[i-1] = tensor->layout[i-1] % tensor->block_layout[i-1];
		} else {
			remainder_index[i-1] = tensor->block_layout[i-1];
			// when it's not an "edgy" block-dim then just assign regular size
		}
	}

	size_t next = 0;
	size_t tensor_index = 0;
	size_t tensor_offset = 0;
	//print_to_console_sizet(block_counter_threshold, dim);
	// BLOCK-LEVEL LOOP (using counter method)

  	double nbits_double = pow(block_counter_threshold[0],0.5);
    size_t nbits = nbits_double;
    if (nbits_double - nbits > 0.5) {
    	nbits++;
    }
    // printf("risky operation, pow(block_counter_threshold[0]=%zu, 1/dim=%f) = %f, nbits=%zu\n", block_counter_threshold[0], 0.5, pow(block_counter_threshold[0],0.5), nbits);

	for (size_t b=0; b<blocks; ++b) {

 		for (size_t d=dim-1; d<=dim-1; --d) {
			if (block_counter_threshold[d]-1 == block_counter[d]) {
				limits[d] = remainder_index[d];
			} else {
				limits[d] = tensor->block_layout[d];
			}
		}
		// TENSOR-LEVEL LOOP
		size_t * counter = calloc(dim, sizeof(size_t));
		for (size_t t=0; t<block_size; ++t) {

			if (unblock == 0) {
				out_tensor->data[next++] = tensor->lin.data[tensor_index+tensor_offset];
			} else {
				out_tensor->data[tensor_index+tensor_offset] = tensor->lin.data[next++];
			}

			tensor_offset = 0;
			++counter[dim-1];
			for (size_t d=dim-1; d!=0; --d) {
				// two conditions to tick the counter
				// 1) it simply reaches the threshold
				// 2) it reaches the limit for this part dimension
				if (counter[d] == limits[d]) {
					if (d!=0) {
						++counter[d-1];
					}
					counter[d] = 0;
				}
				tensor_offset += counter[d] * block_mul[d];
			}
			// handle the 0 case
			if (counter[0] == limits[0]) {
				break;
			}
			tensor_offset += counter[0] * block_mul[0];
		}

		tensor_index = 0;

		// global_result += result_size;
		// tensor_ptr += block_size;

		// block_counters are calculated from the index (el)
		hilbert_incr(dim, nbits, (unsigned long long*) block_counter);

		// potential optimization: if coord[mode] moved, then we simply do not change the result_ptr(!)
		
		// print_to_console_sizet(block_counter, dim);
		// size_t result_coord = 0;
		// result_ptr = base_result_ptr + (hilbert_c2i(dim, nbits, block_counter) * result_size);

		// printf("result_coord at this point its %zu\n", result_coord);
		// wow, job done :D

		// result_ptr = base_result_ptr + global_result;

		// VECTOR HAS TO CHANGE???
		// global_vector = block_counter[mode] * tensor->block_layout[mode];
		// vector_ptr = base_vector_ptr + global_vector;

		// this line affects all dimensions at once
		// morton_inc(block_counter, block_counter_threshold, dim-1);	
		//print_to_console_sizet(block_counter, dim);
		//printf("\n");
		// morton_inc in fact you can increment at most only one dimension
	
		// if you reset a dimension which is not a mode
		// i.e. we go back but in fact we do NOT compute completely different part of result because of reset
		// but we compute the same part of answer!

		// (but the same!) -> we must reset the result_index
		// if dimension is mul of 2: reset is undetected
		// if dimension is not mul of 2: we know exactly when
		
		// brute force: recalculate each time (I think necessary!)
		for (size_t d=dim-1; d<=dim-1; --d) {
			tensor_index += block_counter[d] * block_mul[d] * tensor->block_layout[d];
		}
		free(counter);

	}

	free(limits);
	free(block_mul);
	free(remainder_index);
	free(block_counter);
	free(block_counter_threshold);

}
























void
//block_array_int_test2(struct lin_storage * out_tensor, const struct tensor_storage const * tensor, const size_t unblock) {
block_array_int_test2(struct lin_storage * out_tensor, const struct tensor_storage * restrict tensor) {


	size_t dim = tensor->dim;
	size_t * block_counter_threshold = calloc(dim, sizeof(size_t));
	size_t * block_counter = calloc(dim, sizeof(size_t));
	size_t * remainder_index = calloc(dim, sizeof(size_t));
	//size_t * new_mul = calloc(dim, sizeof(size_t));
	size_t * mul = calloc(dim, sizeof(size_t));
	size_t * limits = calloc(dim, sizeof(size_t));
	size_t * block_mul = calloc(dim, sizeof(size_t));
	size_t block_size = 1;
	size_t blocks = 1;
	mul[dim-1] = 1;
	block_mul[dim-1] = 1;
	for (size_t i=dim; i!=0; --i) {
		if (i-1 != 0) {
			mul[i-2] = mul[i-1] * tensor->layout[tensor->layout_perm[i-1]];
			block_mul[i-2] = block_mul[i-1] * tensor->layout[tensor->layout_perm[i-1]];
		}
		// integer division with rounding up formula :)
		block_counter_threshold[i-1] = (tensor->layout[i-1] + tensor->block_layout[i-1] -1)
			/ tensor->block_layout[i-1];
		blocks *= block_counter_threshold[i-1];
		block_size *= tensor->block_layout[i-1];
		if (tensor->layout[i-1] % tensor->block_layout[i-1] != 0) {
			remainder_index[i-1] = tensor->layout[i-1] % tensor->block_layout[i-1];
		} else {
			remainder_index[i-1] = tensor->block_layout[i-1];
			// when it's not an "edgy" block-dim then just assign regular size
		}
	}
	size_t result_index = 0;
	size_t tensor_index = 0;
	size_t result_offset = 0;
	// BLOCK-LEVEL LOOP (using counter method)
	for (size_t b=0; b<blocks; ++b) {
		for (size_t d=dim-1; d<=dim-1; --d) {
			if (block_counter_threshold[d]-1 == block_counter[d]) {
				limits[d] = remainder_index[d];
			} else {
				limits[d] = tensor->block_layout[d];
			}
		}
		// TENSOR-LEVEL LOOP
		size_t * counter = calloc(dim, sizeof(size_t));
		for (size_t t=0; t<block_size; ++t) {
			out_tensor->data[result_index+result_offset] = tensor->lin.data[tensor_index++];
			++counter[dim-1];

			for (size_t d=dim-1; d!=0; --d) {

				// two conditions to tick the counter
				// 1) it simply reaches the threshold
				// 2) it reaches the limit for this part dimension
				if (counter[d] == limits[d]) {
					if (d>0) {
						++counter[d-1];
					}
					counter[d] = 0;
				}
			}
			// handle the 0 case
			if (counter[0] == limits[0]) {
				break;
			}


		}
		++block_counter[dim-1];
		for (size_t d=dim-1; d<=dim-1; --d) {
			if (block_counter[d] == block_counter_threshold[d]) {
				if (d>0) {
					++block_counter[d-1];
				}
				// alt: put below if into the above if (???)
				block_counter[d] = 0;
				//if (d == mode) {
				//} else if (d < mode) {
					//result_index += block_counter[d-1] * mul[d-1] * tensor->block_layout[d-1];
				//}
				//break;
			}
		}
		free(counter);
	}
	free(mul);
	free(limits);
	//free(new_mul);
	free(block_mul);
	free(remainder_index);
	free(block_counter);
	free(block_counter_threshold);

}






