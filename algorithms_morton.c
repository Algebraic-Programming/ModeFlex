#include<algorithms.h>
#include<rand_utils.h>
#include<file_utils.h>
#include<string.h>
#include<stdlib.h>
#include<math.h>
#include<space_curves.h>
#include<mkl.h>

/////////////////////////////////////// SPIS TRESCI
////////// BLAS_POWERS		working on it
////////// BLAS_POWERS_2	working on it

////////// BLAS_POWERS_bench	WORKS
////////// BLAS_POWERS_bench_2	WORKS
////////// BLAS_POWERS_bench_3 	WORKS (finally, optimized?)

////////// POWERS		little bug(s)
////////// _op1 		WORKS

/////////////////////// VERSION TESTED : BLAS -> no unfold

void
tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_BLAS(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {
	
	const size_t dim = tensor->dim;

	// Morton stuff
	size_t * const block_counter = calloc(dim, sizeof(size_t));
	size_t * const block_counter_threshold = calloc(dim, sizeof(size_t));

	// Required for unfold: right_size, result_size, vector_size, block_size
	const size_t vector_size = tensor->block_layout[mode];
	size_t block_size = 1;
	// Required for termination
	size_t blocks = 1;
	// Required to know the number of levels needed for morton curve
	size_t max_block = 0;
	for (size_t i=dim-1; i<dim; --i) {

		block_size *= tensor->block_layout[tensor->layout_perm[i]];

		block_counter_threshold[i] = (tensor->layout[i] + tensor->block_layout[i] -1) / tensor->block_layout[i];
		if (block_counter_threshold[i] > max_block) {
			max_block = block_counter_threshold[i];
		}
		blocks *= block_counter_threshold[i];

	}
	const size_t result_size = block_size / vector_size;

	const size_t morton_block_levels = log2(max_block)+1; // round-up (take front size_teger and add 1)
	size_t * const morton_block_indices = calloc(morton_block_levels, sizeof(size_t));

	// Elements for traversing T-V-R
	size_t global_result = 0;
	size_t global_vector = 0;
	size_t next = 0;

	// Things for MKL
	const double alpha = 1;
	const double beta = 1;
	const MKL_INT incx = 1;
	const MKL_INT incy = 1;
	const MKL_INT lda = vector_size;

	for (size_t b=0;;) {

		// BLAS call: tensor (increment next)
		cblas_dgemv(
			CblasRowMajor, CblasNoTrans,
			result_size, lda, // vector_size, const MKL_INT*2
			alpha, // const double
			tensor->lin.data + next, lda, // const double*, const MKL_INT
			(vector->data + global_vector), incx, // const double*, const MKL_INT
			beta, // const double
			(result_tensor->data + global_result), incy); // const double*, const MKL_INT
		next += block_size;

		if (++b == blocks) {
			break;
		}

		// In both cases, increment global_result
		global_result += result_size;

		size_t mask = 1;
		size_t level = 0;
		size_t inc_game = 1;
		size_t offset = dim-1;

		while (inc_game) {
			if (block_counter[offset] & mask) {
				block_counter[offset] &= ~mask;
				if (offset == 0) {
					mask <<= 1;
					level += 1;
					offset = dim-1;
				} else {
					offset -= 1;
				}
			} else {
				if ((block_counter[offset] | mask) >= block_counter_threshold[offset]) {
					if (offset == 0) {
						mask <<= 1;
						level += 1;
						offset = dim-1;
					} else {
						offset -= 1;
					}
				} else {
					inc_game = 0;
				}
			}
		}
		block_counter[offset] |= mask;
		if (offset == mode) {
			// swap values
			size_t temp = global_result;
			global_result = morton_block_indices[level];
			morton_block_indices[level] = temp;
			// propagate if we are finished with the mode dimension
			if (level > 0 && block_counter[mode] != block_counter_threshold[mode]-1) {
				for (size_t i=level-1; i<=level-1; --i) {
					morton_block_indices[i] = global_result;
				} 
			}
		}

		// calculate global vector
		global_vector = block_counter[mode] * tensor->block_layout[mode];

	}

	free(morton_block_indices);
	free(block_counter);
	free(block_counter_threshold);
}

/////////////////////// VERSION : UNFOLD

void
tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_UNFOLD(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {
	
	const size_t dim = tensor->dim;

	// Morton stuff
	size_t * const block_counter = calloc(dim, sizeof(size_t));
	size_t * const block_counter_threshold = calloc(dim, sizeof(size_t));

	// Required for unfold: right_size, result_size, vector_size, block_size
	const size_t vector_size = tensor->block_layout[mode];
	size_t right_size = 1;
	size_t block_size = 1;
	// Required for termination
	size_t blocks = 1;
	// Required to know the number of levels needed for morton curve
	size_t max_block = 0;
	for (size_t i=dim-1; i<dim; --i) {

		if (i > mode) {
			right_size *= tensor->block_layout[tensor->layout_perm[i]];
		}
		block_size *= tensor->block_layout[tensor->layout_perm[i]];

		block_counter_threshold[i] = (tensor->layout[i] + tensor->block_layout[i] -1) / tensor->block_layout[i];
		if (block_counter_threshold[i] > max_block) {
			max_block = block_counter_threshold[i];
		}
		blocks *= block_counter_threshold[i];

	}
	const size_t result_size = block_size / vector_size;

	const size_t morton_block_levels = log2(max_block)+1; // round-up (take front size_teger and add 1)
	size_t * const morton_block_indices = calloc(morton_block_levels, sizeof(size_t));

	// Elements for traversing T-V-R
	size_t global_result = 0;
	size_t next = 0;

	// Decide whether or not UNFOLD_REQ
	size_t UNFOLD_REQ;
	DTYPE * unfold;
	if (mode != dim-1) {
		UNFOLD_REQ = 1;
		unfold = malloc(block_size * sizeof(DTYPE));
		if (unfold == NULL) {
			printf("Malloc failure\n");
			exit(-1);
		}
	} else {
		UNFOLD_REQ = 0;
	}

	// Compute stride for UNFOLD
	size_t stride;
	if (mode != 0) {
		stride = right_size * (tensor->block_layout[tensor->layout_perm[mode]] - 1);
	} else {
		stride = 0;
	}

	// For the unfold
	size_t global_tensor = 0;
	size_t tensor_index, tensor_diff;

	for (size_t b=0;;) {

		if (UNFOLD_REQ) {
			// requires: result_size, right_size, vector_size
			next = 0;
			tensor_index = global_tensor;
			tensor_diff = 0;
			for (size_t i=0; i<result_size; ++i) {
				if ((i!=0) & (i % right_size == 0)) {
					tensor_diff += stride;
				}
				tensor_index = global_tensor + i + tensor_diff;
				for (size_t j=0; j<vector_size; ++j) {
					unfold[next++] = tensor->lin.data[tensor_index];
					tensor_index += right_size;
				}
			}
			// BLAS call: unfold (must reset next)
			global_tensor += block_size;
		} else {
			// BLAS call: tensor (increment next)
			next += block_size;
		}

		if (++b == blocks) {
			break;
		}

		// In both cases, increment global_result
		global_result += result_size;

		size_t mask = 1;
		size_t level = 0;
		size_t inc_game = 1;
		size_t offset = dim-1;

		while (inc_game) {
			if (block_counter[offset] & mask) {
				block_counter[offset] &= ~mask;
				if (offset == 0) {
					mask <<= 1;
					level += 1;
					offset = dim-1;
				} else {
					offset -= 1;
				}
			} else {
				if ((block_counter[offset] | mask) >= block_counter_threshold[offset]) {
					if (offset == 0) {
						mask <<= 1;
						level += 1;
						offset = dim-1;
					} else {
						offset -= 1;
					}
				} else {
					inc_game = 0;
				}
			}
		}
		block_counter[offset] |= mask;
		if (offset == mode) {
			// swap values
			size_t temp = global_result;
			global_result = morton_block_indices[level];
			morton_block_indices[level] = temp;
			// propagate if we are finished with the mode dimension
			if (level > 0 && block_counter[mode] != block_counter_threshold[mode]-1) {
				for (size_t i=level-1; i<=level-1; --i) {
					morton_block_indices[i] = global_result;
				} 
			}
		}

	}

	if (UNFOLD_REQ) {
		result_tensor->data[0] = unfold[0];
		free(unfold);
	}

	free(morton_block_indices);
	free(block_counter);
	free(block_counter_threshold);
}

/////////////////////// VERSION TESTED : UNFOLD + BLAS
#if 0
void
tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {
	
	const size_t dim = tensor->dim;

	// Morton stuff
	size_t * const block_counter = calloc(dim, sizeof(size_t));
	size_t * const block_counter_threshold = calloc(dim, sizeof(size_t));

	// Required for unfold: right_size, result_size, vector_size, block_size
	const size_t vector_size = tensor->block_layout[mode];
	size_t right_size = 1;
	size_t block_size = 1;
	// Required for termination
	size_t blocks = 1;
	// Required to know the number of levels needed for morton curve
	size_t max_block = 0;
	for (size_t i=dim-1; i<dim; --i) {

		if (i > mode) {
			right_size *= tensor->block_layout[tensor->layout_perm[i]];
		}
		block_size *= tensor->block_layout[tensor->layout_perm[i]];

		block_counter_threshold[i] = (tensor->layout[i] + tensor->block_layout[i] -1) / tensor->block_layout[i];
		if (block_counter_threshold[i] > max_block) {
			max_block = block_counter_threshold[i];
		}
		blocks *= block_counter_threshold[i];

	}
	const size_t result_size = block_size / vector_size;

	const size_t morton_block_levels = log2(max_block)+1; // round-up (take front size_teger and add 1)
	size_t * const morton_block_indices = calloc(morton_block_levels, sizeof(size_t));

	// Elements for traversing T-V-R
	size_t global_result = 0;
	size_t global_vector = 0;
	size_t next = 0;

	// Things for MKL
	const double alpha = 1;
	const double beta = 1;
	const MKL_INT incx = 1;
	const MKL_INT incy = 1;
	const MKL_INT lda = vector_size;

	// Decide whether or not UNFOLD_REQ
	size_t UNFOLD_REQ;
	DTYPE * unfold;
	if (mode != dim-1) {
		UNFOLD_REQ = 1;
		unfold = malloc(block_size * sizeof(DTYPE));
		if (unfold == NULL) {
			printf("Malloc failure\n");
			exit(-1);
		}
	} else {
		UNFOLD_REQ = 0;
	}

	// Compute stride for UNFOLD
	size_t stride;
	if (mode != 0) {
		stride = right_size * (tensor->block_layout[tensor->layout_perm[mode]] - 1);
	} else {
		stride = 0;
	}

	// For the unfold
	size_t global_tensor = 0;
	size_t tensor_index, tensor_diff;

	for (size_t b=0;;) {

		if (UNFOLD_REQ) {
			// requires: result_size, right_size, vector_size
			next = 0;
			tensor_index = global_tensor;
			tensor_diff = 0;
			for (size_t i=0; i<result_size; ++i) {
				if ((i!=0) & (i % right_size == 0)) {
					tensor_diff += stride;
				}
				tensor_index = global_tensor + i + tensor_diff;
				for (size_t j=0; j<vector_size; ++j) {
					unfold[next++] = tensor->lin.data[tensor_index];
					tensor_index += right_size;
				}
			}
			// BLAS call: unfold (must reset next)
			global_tensor += block_size;
			cblas_dgemv(
				CblasRowMajor, CblasNoTrans,
				result_size, lda, // vector_size, const MKL_INT*2
				alpha, // const double
				unfold, lda, // const double*, const MKL_INT
				(vector->data + global_vector), incx, // const double*, const MKL_INT
				beta, // const double
				(result_tensor->data + global_result), incy); // const double*, const MKL_INT
		} else {
			// BLAS call: tensor (increment next)
			cblas_dgemv(
				CblasRowMajor, CblasNoTrans,
				result_size, lda, // vector_size, const MKL_INT*2
				alpha, // const double
				tensor->lin.data + next, lda, // const double*, const MKL_INT
				(vector->data + global_vector), incx, // const double*, const MKL_INT
				beta, // const double
				(result_tensor->data + global_result), incy); // const double*, const MKL_INT
			next += block_size;
		}

		if (++b == blocks) {
			break;
		}

		// In both cases, increment global_result
		global_result += result_size;

		size_t mask = 1;
		size_t level = 0;
		size_t inc_game = 1;
		size_t offset = dim-1;

		while (inc_game) {
			if (block_counter[offset] & mask) {
				block_counter[offset] &= ~mask;
				if (offset == 0) {
					mask <<= 1;
					level += 1;
					offset = dim-1;
				} else {
					offset -= 1;
				}
			} else {
				if ((block_counter[offset] | mask) >= block_counter_threshold[offset]) {
					if (offset == 0) {
						mask <<= 1;
						level += 1;
						offset = dim-1;
					} else {
						offset -= 1;
					}
				} else {
					inc_game = 0;
				}
			}
		}
		block_counter[offset] |= mask;
		if (offset == mode) {
			// swap values
			size_t temp = global_result;
			global_result = morton_block_indices[level];
			morton_block_indices[level] = temp;
			// propagate if we are finished with the mode dimension
			if (level > 0 && block_counter[mode] != block_counter_threshold[mode]-1) {
				for (size_t i=level-1; i<=level-1; --i) {
					morton_block_indices[i] = global_result;
				} 
			}
		}

		// calculate global vector
		global_vector = block_counter[mode] * tensor->block_layout[mode];

	}

	if (UNFOLD_REQ) {
		free(unfold);
	}

	free(morton_block_indices);
	free(block_counter);
	free(block_counter_threshold);
}
#endif

/////////////////////////// AUX ONLY
void
tvm_morton_block_major_input_aligned_output_aligned_POWERS_aux(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {
	const size_t dim = tensor->dim;
	//const size_t vector_size = tensor->block_layout[mode];
	// Morton stuff
	size_t * const block_counter_threshold = calloc(dim, sizeof(size_t));
	size_t * const block_counter = calloc(dim, sizeof(size_t));
	size_t max_block = 0;
	size_t blocks = 1;
	for (size_t i=dim-1; i<dim; --i) {
		block_counter_threshold[i] = (tensor->layout[i] + tensor->block_layout[i] -1) / tensor->block_layout[i];
		if (block_counter_threshold[i] > max_block) {
			max_block = block_counter_threshold[i];
		}
		blocks *= block_counter_threshold[i];
	}
	const size_t morton_block_levels = log2(max_block)+1;
	size_t * const morton_block_indices = calloc(morton_block_levels, sizeof(size_t));
	size_t global_result = 0;
	//size_t global_vector = 0;
	//size_t next = 0;
	for (size_t b=0;;) {
		if (++b == blocks) {
			break;
		}
		size_t mask = 1;
		size_t level = 0;
		size_t inc_game = 1;
		size_t offset = dim-1;
		while (inc_game) {
			if (block_counter[offset] & mask) {
				block_counter[offset] &= ~mask;
				if (offset == 0) {
					mask <<= 1;
					level += 1;
					offset = dim-1;
				} else {
					offset -= 1;
				}
			} else {
				if ((block_counter[offset] | mask) >= block_counter_threshold[offset]) {
					if (offset == 0) {
						mask <<= 1;
						level += 1;
						offset = dim-1;
					} else {
						offset -= 1;
					}
				} else {
					inc_game = 0;
				}
			}
		}
		block_counter[offset] |= mask;
		if (offset == mode) {
			// swap values
			size_t temp = global_result;
			global_result = morton_block_indices[level];
			morton_block_indices[level] = temp;
			// propagate if we are finished with the mode dimension
			if (level > 0 && block_counter[mode] != block_counter_threshold[mode]-1) {
				for (size_t i=level-1; i<=level-1; --i) {
					morton_block_indices[i] = global_result;
				} 
			}
		}
	}
	free(morton_block_indices);
	free(block_counter);
	free(block_counter_threshold);
}

/////////////////////////////////////////////////////////////////////////////

void
tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_bench(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

	// WORKING CODE OF BLOCK VERSION
	const size_t dim = tensor->dim;
	const size_t vector_size = tensor->block_layout[mode];

	// Morton stuff
	size_t * const block_counter_threshold = calloc(dim, sizeof(size_t));
	size_t * const block_counter = calloc(dim, sizeof(size_t));
	size_t max_block = 0;

	for (size_t i=dim-1; i<dim; --i) {
		block_counter_threshold[i] = (tensor->layout[i] + tensor->block_layout[i] -1) / tensor->block_layout[i];
		if (block_counter_threshold[i] > max_block) {
			max_block = block_counter_threshold[i];
		}
	}

	struct morton_data * const morton_metadata = calloc(1, sizeof(struct morton_data));
	const size_t morton_block_levels = log2(max_block)+1; // round-up (take front size_teger and add 1)
	size_t * const morton_block_indices = calloc(morton_block_levels, sizeof(size_t));

	size_t right_size = 1;
	size_t block_size = 1;
	for (size_t d=dim-1; d<dim; --d) {
		if (d > mode) {
			right_size *= tensor->block_layout[tensor->layout_perm[d]];
		}
		block_size *= tensor->block_layout[tensor->layout_perm[d]];
	}
	const size_t result_size = block_size / vector_size;

	//size_t really_global_result = 0;
	size_t global_result = 0;
	size_t global_vector = 0;
	size_t next = 0;
	const double alpha = 1;
	const double beta = 1;
	const MKL_INT incx = 1;
	const MKL_INT incy = 1;
	const MKL_INT lda = vector_size;
	//MKL_INT n;

	// required for the unfold!
	//size_t stride;
	//if (mode != 0) {
	//	stride = right_size * (tensor->block_layout[tensor->layout_perm[mode]] - 1);
	//} else {
	//	stride = 0;
	//}
	//size_t global_tensor = 0;
	//size_t tensor_index, tensor_diff;

	size_t UNFOLD_REQ;
	DTYPE * unfold;
	if (mode != dim-1) {
		UNFOLD_REQ = 1;
		unfold = malloc(block_size * sizeof(DTYPE));
		if (unfold == NULL) {
			printf("Malloc failure\n");
		}
		//n = tensor->block_layout[dim-1]; // layout_perm not needed (?) we are referring to original
	} else {
		UNFOLD_REQ = 0;
		//n = tensor->block_layout[dim-2]; // layout_perm not needed (?) we are referring to original
	}

	//const size_t left_mat_size = result_size / n;
	//const size_t mat_size = tensor->block_layout[mode] * n;
	//size_t next_result;

	//for(size_t el=0;;) {
	for(;;) {

		cblas_dgemv(
			CblasRowMajor, CblasNoTrans,
			result_size, lda, // vector_size, const MKL_INT*2
			alpha, // const double
			tensor->lin.data + next, lda, // const double*, const MKL_INT
			(vector->data + global_vector), incx, // const double*, const MKL_INT
			beta, // const double
			(result_tensor->data + global_result), incy); // const double*, const MKL_INT

		next += block_size;
		global_result += result_size;

		// morton increment
		morton_inc_struct(block_counter, block_counter_threshold, dim-1, morton_metadata);

		// termination condition
		if (morton_metadata->level >= morton_block_levels) {
			break;
		}

		// calculate global_result
		if (morton_metadata->dim==mode) {
			// swap values
			size_t init_level = morton_metadata->level;
			size_t temp = global_result;
			global_result = morton_block_indices[init_level];
			morton_block_indices[init_level] = temp;

			// propagate if we are finished with the mode dimension
			if (init_level>0 && block_counter[mode] != block_counter_threshold[mode]-1) {
				for (size_t i=--init_level; i<=init_level; --i) {
					morton_block_indices[i] = global_result;
				} 
			}
		}

		// calculate global vector
		global_vector = block_counter[mode] * tensor->block_layout[mode];
	}

	if (UNFOLD_REQ) {
		free(unfold);
	}

	free(morton_metadata);
	free(morton_block_indices);
	free(block_counter);
	free(block_counter_threshold);
}

// morton BLAS BENCH
void
tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_n(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

	// WORKING CODE OF BLOCK VERSION
	const size_t dim = tensor->dim;
	const size_t vector_size = tensor->block_layout[mode];

	// Morton stuff
	size_t * const block_counter_threshold = calloc(dim, sizeof(size_t));
	size_t * const block_counter = calloc(dim, sizeof(size_t));
	size_t * const block_counter_2 = calloc(dim, sizeof(size_t));
	size_t blocks = 1;
	size_t max_block = 0;
	for (size_t i=dim-1; i<dim; --i) {
		block_counter_threshold[i] = (tensor->layout[i] + tensor->block_layout[i] -1) / tensor->block_layout[i];
		if (block_counter_threshold[i] > max_block) {
			max_block = block_counter_threshold[i];
		}
		blocks *= block_counter_threshold[i];
	}
	struct morton_data * const morton_metadata = calloc(1, sizeof(struct morton_data));
	const size_t morton_block_levels = log2(max_block)+1; // round-up (take front size_teger and add 1)
	size_t * const morton_block_indices = calloc(morton_block_levels, sizeof(size_t));

	size_t right_size = 1;
	size_t block_size = 1;
	for (size_t d=dim-1; d<dim; --d) {
		if (d > mode) {
			right_size *= tensor->block_layout[tensor->layout_perm[d]];
		}
		block_size *= tensor->block_layout[tensor->layout_perm[d]];
	}

	const size_t result_size = block_size / vector_size;

	//size_t really_global_result = 0;
	size_t global_result = 0;
	size_t global_vector = 0;
	size_t next = 0;
	const double alpha = 1;
	const double beta = 1;
	const MKL_INT incx = 1;
	const MKL_INT incy = 1;
	const MKL_INT lda = vector_size;
	//MKL_INT n;

	// required for the unfold!
	size_t stride;
	if (mode != 0) {
		stride = right_size * (tensor->block_layout[tensor->layout_perm[mode]] - 1);
	} else {
		stride = 0;
	}
	size_t global_tensor = 0;
	size_t tensor_index, tensor_diff;

	//prsize_t_to_console_size_t(tensor->block_layout, dim);

	size_t UNFOLD_REQ;
	DTYPE * unfold;
	if (mode != dim-1) {
		UNFOLD_REQ = 1;
		unfold = malloc(block_size * sizeof(DTYPE));
		if (unfold == NULL) {
			printf("Malloc failure\n");
		}
		//n = tensor->block_layout[dim-1]; // layout_perm not needed (?) we are referring to original
	} else {
		UNFOLD_REQ = 0;
		//n = tensor->block_layout[dim-2]; // layout_perm not needed (?) we are referring to original
	}

	// BLAS matrix-style processing
	//for (;;) {
	for(size_t b=0;;) {

		if (UNFOLD_REQ) {
			next = 0;
			tensor_index = global_tensor;
			tensor_diff = 0;
			for (size_t i=0; i<result_size; ++i) {
				if ((i!=0) & (i % right_size == 0)) {
					tensor_diff += stride;
				}
				tensor_index = global_tensor + i + tensor_diff;
				for (size_t j=0; j<vector_size; ++j) {
					unfold[next++] = tensor->lin.data[tensor_index];
					tensor_index += right_size;
				}
			}
			global_tensor += block_size;
			cblas_dgemv(
				CblasRowMajor, CblasNoTrans,
				result_size, lda, // vector_size, const MKL_INT*2
				alpha, // const double
				unfold, lda, // const double*, const MKL_INT
				(vector->data + global_vector), incx, // const double*, const MKL_INT
				beta, // const double
				(result_tensor->data + global_result), incy); // const double*, const MKL_INT
		} else {
			cblas_dgemv(
				CblasRowMajor, CblasNoTrans,
				result_size, lda, // vector_size, const MKL_INT*2
				alpha, // const double
				tensor->lin.data + next, lda, // const double*, const MKL_INT
				(vector->data + global_vector), incx, // const double*, const MKL_INT
				beta, // const double
				(result_tensor->data + global_result), incy); // const double*, const MKL_INT
		}

		// STOP MORTON CURVE
		if (++b == blocks) {
			break;
		}

		global_result += result_size;
		next += block_size;

		// get the next morton block
		size_t mask = 1;
		size_t level = 0;
		//size_t quit = 0;
		size_t inc_game = 1;
		size_t offset = dim-1;

		while (inc_game) {
			if (block_counter[offset] & mask) {
				block_counter[offset] &= ~mask;
				if (offset == 0) {
					mask <<= 1;
					level += 1;
					offset = dim-1;
				} else {
					offset -= 1;
				}
			} else {
				if ((block_counter[offset] | mask) >= block_counter_threshold[offset]) {
					if (offset == 0) {
						mask <<= 1;
						level += 1;
						offset = dim-1;
					} else {
						offset -= 1;
					}
				} else {
					inc_game = 0;
				}
			}
		}

		block_counter[offset] |= mask;

		if (offset == mode) {
			// swap values
			size_t temp = global_result;
			global_result = morton_block_indices[level];
			morton_block_indices[level] = temp;
			// propagate if we are finished with the mode dimension
			if (block_counter[mode] != block_counter_threshold[mode]-1) {
				for (size_t i=--level; i<=level; --i) {
					morton_block_indices[i] = global_result;
				} 
			}
		}

		// calculate global vector
		global_vector = block_counter[mode] * tensor->block_layout[mode];

	}

	if (UNFOLD_REQ) {
		free(unfold);
	}

	free(morton_metadata);
	free(morton_block_indices);
	free(block_counter);
	free(block_counter_2);
	free(block_counter_threshold);
}

void
tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_bench_3(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

	const size_t dim = tensor->dim;

	size_t * const block_counter_threshold = calloc(dim, sizeof(size_t));
	size_t * const remainder_index = calloc(dim, sizeof(size_t));
	//size_t * const block_mul = calloc(dim, sizeof(size_t));
	size_t * const block_counter = calloc(dim, sizeof(size_t));

	//block_mul[dim-1] = 1;
	size_t max_block = 0;
	size_t blocks = 1;

	// We must compute the block_counter_thresholds (num of blocks in each dimension)
	for (size_t i=dim-1; i<=dim-1; --i) {
		// size_teger division with rounding up formula
		block_counter_threshold[i] = (tensor->layout[i] + tensor->block_layout[i] -1)
			/ tensor->block_layout[i];
		if (block_counter_threshold[i] > max_block) {
			max_block = block_counter_threshold[i];
		}
		blocks *= block_counter_threshold[i];
		if (tensor->layout[i] % tensor->block_layout[i] != 0) {
			remainder_index[i] = tensor->layout[i] % tensor->block_layout[i];
		} else {
			remainder_index[i] = tensor->block_layout[i];
			// when it's not an "edgy" block-dim then just assign regular size
		}
		//if (i!=0) {
			//block_mul[i-1] = block_mul[i] * tensor->layout[i];
		//}
	}

	struct morton_data * const morton_metadata = calloc(1, sizeof(struct morton_data));
	// round-up (take front size_teger and add 1)
	const size_t morton_block_levels = log2(max_block)+1;

	size_t * const morton_block_indices = calloc(morton_block_levels, sizeof(size_t));

	size_t global_result = 0;
	size_t vector_index = 0;
	//size_t global_vector = 0;

	size_t t = 0;
	//size_t vector_block_size = tensor->block_layout[mode];

	// PRECALCULATE THE SIZE OF BLOCK TO SEE ANY OBSERVED SPEEDUP

	size_t block_size = 1;
	for (size_t d=dim-1; d<dim; --d) {
		block_size *= tensor->block_layout[tensor->layout_perm[d]];
	}
	const size_t result_size = block_size / tensor->block_layout[mode];

	//size_t really_global_result = 0;
	//size_t global_result = 0;
	//size_t global_vector = 0;
	//size_t next = 0;
	const double alpha = 1;
	const double beta = 1;
	const MKL_INT incx = 1;
	const MKL_INT incy = 1;
	const MKL_INT lda = tensor->block_layout[mode];
	MKL_INT n;

	if (mode != dim-1) {
		n = tensor->block_layout[dim-1]; // layout_perm not needed (?) we are referring to original
	} else {
		n = tensor->block_layout[dim-2]; // layout_perm not needed (?) we are referring to original
	}
	size_t next_result = 0;
	const size_t left_mat_size = result_size / n;
	const size_t mat_size = tensor->block_layout[mode] * n;
	
	for (size_t b=0; b<blocks; ++b) {

		////////// FIND DIMENSIONS OF THE BLOCK WE ARE IN
		size_t right_size = 1;
		size_t left_size = 1;
		//size_t vector_size = 0;
		for (size_t d=dim-1; d<=dim-1; --d) {
			if (d > mode) {
				if (block_counter_threshold[d]-1 == block_counter[d]) {
					right_size *= remainder_index[d];
				} else {
					right_size *= tensor->block_layout[d];
				}
			} else if (d == mode) {
				if (block_counter_threshold[d]-1 == block_counter[d]) {
					//vector_size = remainder_index[d];
				} else {
					//vector_size = tensor->block_layout[d];
				}
			} else {
				if (block_counter_threshold[d]-1 == block_counter[d]) {
					left_size *= remainder_index[d];
				} else {
					left_size *= tensor->block_layout[d];
				}
			}
		}

		////////// GO THROUGH ELEMENTS IN THAT BLOCK
		next_result = 0;
		for (size_t i=0; i<left_mat_size; ++i) {
			cblas_dgemv(
				CblasRowMajor, CblasNoTrans,
				n, lda, // vector_size, const MKL_INT*2
				alpha, // const double
				(tensor->lin.data + t), lda, // const double*, const MKL_INT
				(vector->data + vector_index), incx, // const double*, const MKL_INT
				beta, // const double
				(result_tensor->data + global_result + next_result), incy); // const double*, const MKL_INT
			t += mat_size;
			next_result += n;
		}
		global_result += result_size;

#if 0
		size_t temp_result = 0;
		for (size_t i=0; i<left_size; ++i) {
			for (size_t v=0; v<vector_size; ++v) {
				for (size_t j=0; j<right_size; ++j) {
					result_tensor->data[global_result+temp_result+j] +=
						tensor->lin.data[t++] * vector->data[vector_index+v];
				}
			}
			temp_result += right_size;
		}
		global_result += temp_result;
#endif

		////////// INCREMENT BLOCK COUNTERS (according to the morton curve)
		morton_inc_struct(block_counter, block_counter_threshold, dim-1, morton_metadata);

		////////// CODE TO CALCULATE THE RESULT INDEX
		if (morton_metadata->level >= morton_block_levels) {
			break;
		}

		if (morton_metadata->dim==mode) {
			size_t init_level = morton_metadata->level; 
			// swap global_result with a result index stored at morton_block_indices[init_level]
			if (global_result < result_tensor->size) {
				// swap it!
				size_t temp = global_result;
				global_result = morton_block_indices[init_level];
				morton_block_indices[init_level] = temp;
			} else {
				global_result = morton_block_indices[init_level];
			}
			// consider how much space left for the z-curve
			// diff belongs to <0,~)
			if (init_level>0 && block_counter_threshold[mode]-1 != block_counter[mode]) {
				const size_t diff_level = log2(block_counter_threshold[mode]-1-block_counter[mode])+1;
				if (diff_level < init_level) {
					init_level = diff_level-1;
				} else {
					init_level = init_level-1;
				}
				for (size_t i=init_level; i<=init_level; --i) {
					morton_block_indices[i] = global_result;
				} 
			}
		}
		// block_counter[mode] may have been decreased (above condition checks only if it has been increased)
		vector_index = block_counter[mode] * tensor->block_layout[mode];
		//tensor_index = 0;
		//for (size_t d=dim-1; d<=dim-1; --d) {
			//tensor_index += block_counter[d] * block_mul[d] * tensor->block_layout[d];
		//}
	}
	free(morton_metadata);
	free(morton_block_indices);
	free(block_counter);
	//free(block_mul);
	free(remainder_index);
	free(block_counter_threshold);
}

// morton BLAS BENCH
void
tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_bench_2(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

	// WORKING CODE OF BLOCK VERSION
	const size_t dim = tensor->dim;
	const size_t vector_size = tensor->block_layout[mode];

#if 0
	size_t * const mul = malloc(dim * sizeof(size_t));

	size_t blocks = 1;
	mul[dim-1] = 1;
	for (size_t d=dim-1; d<dim; --d) {
		size_t temp = (tensor->layout[d] + tensor->block_layout[d] - 1) / tensor->block_layout[d];
		blocks *= temp;
		if (d!=0) {
			mul[d-1] = mul[d] * temp;
		}
	}
#endif

	// Morton stuff
	size_t * const block_counter_threshold = calloc(dim, sizeof(size_t));
	size_t * const block_counter = calloc(dim, sizeof(size_t));
	size_t max_block = 0;
	for (size_t i=dim-1; i<dim; --i) {
		block_counter_threshold[i] = (tensor->layout[i] + tensor->block_layout[i] -1) / tensor->block_layout[i];
		if (block_counter_threshold[i] > max_block) {
			max_block = block_counter_threshold[i];
		}
	}
	struct morton_data * const morton_metadata = calloc(1, sizeof(struct morton_data));
	const size_t morton_block_levels = log2(max_block)+1; // round-up (take front size_teger and add 1)
	size_t * const morton_block_indices = calloc(morton_block_levels, sizeof(size_t));

	size_t right_size = 1;
	size_t block_size = 1;
	for (size_t d=dim-1; d<dim; --d) {
		if (d > mode) {
			right_size *= tensor->block_layout[tensor->layout_perm[d]];
		}
		block_size *= tensor->block_layout[tensor->layout_perm[d]];
	}
	const size_t result_size = block_size / vector_size;

	//size_t really_global_result = 0;
	size_t global_result = 0;
	size_t global_vector = 0;
	size_t next = 0;
	const double alpha = 1;
	const double beta = 1;
	const MKL_INT incx = 1;
	const MKL_INT incy = 1;
	const MKL_INT lda = vector_size;
	MKL_INT n;

	if (mode != dim-1) {
		n = tensor->block_layout[dim-1]; // layout_perm not needed (?) we are referring to original
	} else {
		n = tensor->block_layout[dim-2]; // layout_perm not needed (?) we are referring to original
	}

	const size_t left_mat_size = result_size / n;
	const size_t mat_size = tensor->block_layout[mode] * n;
	size_t next_result;
	
	//for(size_t el=0;;) {
	for(;;) {

		next_result = 0;
		for (size_t i=0; i<left_mat_size; ++i) {
			cblas_dgemv(
				CblasRowMajor, CblasNoTrans,
				n, lda, // vector_size, const MKL_INT*2
				alpha, // const double
				(tensor->lin.data + next), lda, // const double*, const MKL_INT
				(vector->data + global_vector), incx, // const double*, const MKL_INT
				beta, // const double
				(result_tensor->data + global_result + next_result), incy); // const double*, const MKL_INT
			next += mat_size;
			next_result += n;
		}
		global_result += result_size;

		// morton increment
		morton_inc_struct(block_counter, block_counter_threshold, dim-1, morton_metadata);

		// termination condition
		if (morton_metadata->level >= morton_block_levels) {
			break;
		}

		// calculate global_result
		if (morton_metadata->dim==mode) {
			size_t init_level = morton_metadata->level; 
			if (global_result < result_tensor->size) {
				size_t temp = global_result;
				global_result = morton_block_indices[init_level];
				morton_block_indices[init_level] = temp;
			} else {
				global_result = morton_block_indices[init_level];
			}
			if (init_level>0 && block_counter_threshold[mode]-1 != block_counter[mode]) {
				const size_t diff_level = log2(block_counter_threshold[mode]-1-block_counter[mode])+1;
				if (diff_level < init_level) {
					init_level = diff_level-1;
				} else {
					init_level = init_level-1;
				}
				for (size_t i=init_level; i<=init_level; --i) {
					morton_block_indices[i] = global_result;
				} 
			}
		}

		// calculate global vector
		global_vector = block_counter[mode] * tensor->block_layout[mode];
	}

	free(morton_metadata);
	free(morton_block_indices);
	free(block_counter);
	free(block_counter_threshold);
}

void tvm_morton_block_major_input_aligned_output_aligned_op1(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

	const size_t dim = tensor->dim;

	size_t * const block_counter_threshold = calloc(dim, sizeof(size_t));
	size_t * const remainder_index = calloc(dim, sizeof(size_t));
	size_t * const block_counter = calloc(dim, sizeof(size_t));
	size_t max_block = 0;
	size_t blocks = 1;

	// We must compute the block_counter_thresholds (num of blocks in each dimension)
	for (size_t i=dim-1; i<=dim-1; --i) {
		// size_teger division with rounding up formula
		block_counter_threshold[i] = (tensor->layout[i] + tensor->block_layout[i] -1)
			/ tensor->block_layout[i];
		if (block_counter_threshold[i] > max_block) {
			max_block = block_counter_threshold[i];
		}
		blocks *= block_counter_threshold[i];
		if (tensor->layout[i] % tensor->block_layout[i] != 0) {
			remainder_index[i] = tensor->layout[i] % tensor->block_layout[i];
		} else {
			remainder_index[i] = tensor->block_layout[i];
			// when it's not an "edgy" block-dim then just assign regular size
		}
	}

	struct morton_data * const morton_metadata = calloc(1, sizeof(struct morton_data));
	// round-up (take front size_teger and add 1)
	const size_t morton_block_levels = log2(max_block)+1;
	size_t * const morton_block_indices = calloc(morton_block_levels, sizeof(size_t));
	size_t global_result = 0;
	//size_t global_vector = 0;
	size_t t = 0;
	size_t vector_index = 0;

	// PRECALCULATE THE SIZE OF BLOCK TO SEE ANY OBSERVED SPEEDUP
	size_t right_size = 1;
	size_t left_size = 1;
	size_t vector_size = 0;
	for (size_t d=dim-1; d<=dim-1; --d) {
		if (d > mode) {
			right_size *= tensor->block_layout[d];
		} else if (d == mode) {
			vector_size = tensor->block_layout[d];
		} else {
			left_size *= tensor->block_layout[d];
		}
	}

	for (size_t b=0; b<blocks; ++b) {
		////////// GO THROUGH ELEMENTS IN THAT BLOCK
		size_t temp_result = 0;
		for (size_t i=0; i<left_size; ++i) {
			for (size_t v=0; v<vector_size; ++v) {
				for (size_t j=0; j<right_size; ++j) {
					result_tensor->data[global_result+temp_result+j] +=
						tensor->lin.data[t++] * vector->data[vector_index+v];
				}
			}
			temp_result += right_size;
		}
		global_result += temp_result;

		////////// INCREMENT BLOCK COUNTERS (according to the morton curve)
		morton_inc_struct(block_counter, block_counter_threshold, dim-1, morton_metadata);

		////////// CODE TO CALCULATE THE RESULT INDEX
		if (morton_metadata->level >= morton_block_levels) {
			break;
		}
		if (morton_metadata->dim==mode) {
			size_t init_level = morton_metadata->level; 
			// swap global_result with a result index stored at morton_block_indices[init_level]
			if (global_result < result_tensor->size) {
				// swap it!
				size_t temp = global_result;
				global_result = morton_block_indices[init_level];
				morton_block_indices[init_level] = temp;
			} else {
				global_result = morton_block_indices[init_level];
			}
			// consider how much space left for the z-curve
			// diff belongs to <0,~)
			if (init_level>0 && block_counter_threshold[mode]-1 != block_counter[mode]) {
				const size_t diff_level = log2(block_counter_threshold[mode]-1-block_counter[mode])+1;
				if (diff_level < init_level) {
					init_level = diff_level-1;
				} else {
					init_level = init_level-1;
				}
				for (size_t i=init_level; i<=init_level; --i) {
					morton_block_indices[i] = global_result;
				} 
			}
		}
		vector_index = block_counter[mode] * tensor->block_layout[mode];
	}
	free(morton_metadata);
	free(morton_block_indices);
	free(block_counter);
	free(remainder_index);
	free(block_counter_threshold);
}

void tvm_morton_block_major_input_aligned_output_aligned(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

	const size_t dim = tensor->dim;

	size_t * const block_counter_threshold = calloc(dim, sizeof(size_t));
	size_t * const remainder_index = calloc(dim, sizeof(size_t));
	//size_t * const block_mul = calloc(dim, sizeof(size_t));
	size_t * const block_counter = calloc(dim, sizeof(size_t));

	//block_mul[dim-1] = 1;
	size_t max_block = 0;
	size_t blocks = 1;

	// We must compute the block_counter_thresholds (num of blocks in each dimension)
	for (size_t i=dim-1; i<=dim-1; --i) {
		// size_teger division with rounding up formula
		block_counter_threshold[i] = (tensor->layout[i] + tensor->block_layout[i] -1)
			/ tensor->block_layout[i];
		if (block_counter_threshold[i] > max_block) {
			max_block = block_counter_threshold[i];
		}
		blocks *= block_counter_threshold[i];
		if (tensor->layout[i] % tensor->block_layout[i] != 0) {
			remainder_index[i] = tensor->layout[i] % tensor->block_layout[i];
		} else {
			remainder_index[i] = tensor->block_layout[i];
			// when it's not an "edgy" block-dim then just assign regular size
		}
		//if (i!=0) {
			//block_mul[i-1] = block_mul[i] * tensor->layout[i];
		//}
	}

	struct morton_data * const morton_metadata = calloc(1, sizeof(struct morton_data));

	// round-up (take front size_teger and add 1)
	const size_t morton_block_levels = log2(max_block) + 1;
	size_t * const morton_block_indices = calloc(morton_block_levels, sizeof(size_t));

	size_t global_result = 0;
	size_t vector_index = 0;
	size_t t = 0;

	//size_t vector_block_size = tensor->block_layout[mode];
	// PRECALCULATE THE SIZE OF BLOCK TO SEE ANY OBSERVED SPEEDUP

	for (size_t b=0; b<blocks; ++b) {

		////////// FIND DIMENSIONS OF THE BLOCK WE ARE IN
		size_t right_size = 1;
		size_t left_size = 1;
		size_t vector_size = 0;
		for (size_t d=dim-1; d<=dim-1; --d) {
			if (d > mode) {
				if (block_counter_threshold[d]-1 == block_counter[d]) {
					right_size *= remainder_index[d];
				} else {
					right_size *= tensor->block_layout[d];
				}
			} else if (d == mode) {
				if (block_counter_threshold[d]-1 == block_counter[d]) {
					vector_size = remainder_index[d];
				} else {
					vector_size = tensor->block_layout[d];
				}
			} else {
				if (block_counter_threshold[d]-1 == block_counter[d]) {
					left_size *= remainder_index[d];
				} else {
					left_size *= tensor->block_layout[d];
				}
			}
		}

		////////// GO THROUGH ELEMENTS IN THAT BLOCK
		size_t temp_result = 0;
		for (size_t i=0; i<left_size; ++i) {
			for (size_t v=0; v<vector_size; ++v) {
				for (size_t j=0; j<right_size; ++j) {
					result_tensor->data[global_result+temp_result+j] +=
						tensor->lin.data[t++] * vector->data[vector_index+v];
				}
			}
			temp_result += right_size;
		}
		global_result += temp_result;

		////////// INCREMENT BLOCK COUNTERS (according to the morton curve)
		morton_inc_struct(block_counter, block_counter_threshold, dim-1, morton_metadata);

		////////// CODE TO CALCULATE THE RESULT INDEX
		if (morton_metadata->level >= morton_block_levels) {
			break;
		}

		if (morton_metadata->dim==mode) {
			size_t init_level = morton_metadata->level; 
			// swap global_result with a result index stored at morton_block_indices[init_level]
			if (global_result < result_tensor->size) {
				// swap it!
				size_t temp = global_result;
				global_result = morton_block_indices[init_level];
				morton_block_indices[init_level] = temp;
			} else {
				global_result = morton_block_indices[init_level];
			}

			// consider how much space left for the z-curve
			// diff belongs to <0,~)

			if (init_level>0 && block_counter_threshold[mode]-1 != block_counter[mode]) {
				const size_t diff_level = log2(block_counter_threshold[mode]-1 - block_counter[mode]) + 1;
				if (diff_level < init_level) {
					init_level = diff_level-1;
				} else {
					init_level = init_level-1;
				}
				for (size_t i=init_level; i<=init_level; --i) {
					morton_block_indices[i] = global_result;
				} 
			}
		}

		// block_counter[mode] may have been decreased (above condition checks only if it has been increased)
		vector_index = block_counter[mode] * tensor->block_layout[mode];
		//tensor_index = 0;
		//for (size_t d=dim-1; d<=dim-1; --d) {
			//tensor_index += block_counter[d] * block_mul[d] * tensor->block_layout[d];
		//}

	}

	free(morton_metadata);
	free(morton_block_indices);
	free(block_counter);
	free(remainder_index);
	free(block_counter_threshold);
}

void tvm_morton_block_major_input_aligned_output_aligned_POWERS(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

	struct morton_data * const morton_metadata = calloc(1, sizeof(struct morton_data));

	size_t dim = tensor->dim;
	size_t * block_counter_threshold = calloc(dim, sizeof(size_t));
	size_t * block_counter = calloc(dim, sizeof(size_t));
	size_t * remainder_index = calloc(dim, sizeof(size_t));
	size_t blocks = 1;
	size_t max_block = 0;
	// We must compute the block_counter_thresholds (num of blocks in each dimension)
	for (size_t i=dim-1; i<=dim-1; --i) {
		// size_teger division with rounding up formula
		block_counter_threshold[i] = (tensor->layout[i] + tensor->block_layout[i] -1)
			/ tensor->block_layout[i];

		blocks *= block_counter_threshold[i];
	
		// find the smallest size of blocks along one of the dimensions
		if (block_counter_threshold[i] > max_block) {
			max_block = block_counter_threshold[i];
		}

		if (tensor->layout[i] % tensor->block_layout[i] != 0) {
			remainder_index[i] = tensor->layout[i] % tensor->block_layout[i];
		} else {
			remainder_index[i] = tensor->block_layout[i];
			// when it's not an "edgy" block-dim then just assign regular size
		}
	}

	size_t morton_block_levels = log2(max_block) + 1; // take only the front size_teger (round-down at all times)
	size_t * morton_block_indices = calloc(morton_block_levels, sizeof(size_t));

	size_t vector_block_size = tensor->block_layout[mode];

	size_t global_result = 0;
	size_t global_vector = 0;

	size_t t = 0;

	for (size_t b=0; b<blocks; ++b) {
		////////// FIND DIMENSIONS OF THE BLOCK WE ARE IN
		size_t right_size = 1;
		size_t left_size = 1;
		size_t vector_size = 0;
		for (size_t d=dim-1; d<=dim-1; --d) {
			if (d > mode) {
				if (block_counter_threshold[d]-1 == block_counter[d]) {
					right_size *= remainder_index[d];
				} else {
					right_size *= tensor->block_layout[d];
				}
			} else if (d == mode) {
				if (block_counter_threshold[d]-1 == block_counter[d]) {
					vector_size = remainder_index[d];
				} else {
					vector_size = tensor->block_layout[d];
				}
			} else {
				if (block_counter_threshold[d]-1 == block_counter[d]) {
					left_size *= remainder_index[d];
				} else {
					left_size *= tensor->block_layout[d];
				}
			}
		}
		////////// GO THROUGH ELEMENTS IN THAT BLOCK
		size_t temp_result = 0;
		for (size_t i=0; i<left_size; ++i) {
			for (size_t v=0; v<vector_size; ++v) {
				for (size_t j=0; j<right_size; ++j) {
					result_tensor->data[global_result+temp_result+j] += 
						tensor->lin.data[t++] * vector->data[global_vector+v];
				}
			}
			temp_result += right_size;
		}

		////////// INC BLOCK COUNTERS
		// increment block counters according to morton curve
		// this line affects all dimensions at once
		morton_inc_struct(block_counter, block_counter_threshold, dim-1, morton_metadata);

		// necessary for stopping at tricky positions
		if (morton_metadata->level >= morton_block_levels) {
			// ????????????
			break;
		}


		// if we need to reset the result
		// otherwise: the result is not reset (it's +1 of the last element of the last block)
		if (morton_metadata->dim==mode) {
			// perform a swap
			size_t temp = global_result;
			global_result = morton_block_indices[morton_metadata->level];
			morton_block_indices[morton_metadata->level] = temp;
			// this loop runs only if we ticked forward in a higher level (>0)
			for (size_t i=morton_metadata->level-1; i<=morton_metadata->level-1; --i) {
				// then we must update all lower levels to current position
				morton_block_indices[i] = global_result;
			}

			global_vector = block_counter[mode] * vector_block_size;

		}
	}

	free(remainder_index);
	free(block_counter);
	free(block_counter_threshold);

#if 0

	size_t dim = tensor->dim;
	// counters
	size_t * block_counter = calloc(dim, sizeof(size_t));
	// limits
	size_t * block_counter_threshold = calloc(dim, sizeof(size_t));
	size_t * new_mul = calloc(dim, sizeof(size_t));
	size_t * remainder_index = calloc(dim, sizeof(size_t));
	size_t * limits = calloc(dim, sizeof(size_t));
	// calcs
	size_t * mul = calloc(dim, sizeof(size_t));
	size_t * block_mul = calloc(dim, sizeof(size_t));
	//size_t * new_mul = calloc(dim, sizeof(size_t));
	size_t block_size = 1;
	size_t blocks = 1;
	mul[dim-1] = 1;
	block_mul[dim-1] = 1;
	size_t min_block = 0;

	for (size_t i=dim; i!=0; --i) {
		if (i-1>0) {
			if (i-1 == mode) {
				mul[i-2] = mul[i-1];
			} else {
				mul[i-2] = mul[i-1] * tensor->layout[tensor->layout_perm[i-1]];
			}
			block_mul[i-2] = block_mul[i-1] * tensor->layout[tensor->layout_perm[i-1]];
		}

		// size_teger division with rounding up formula :)
		block_counter_threshold[i-1] = (tensor->layout[i-1] + tensor->block_layout[i-1] -1)
			/ tensor->block_layout[i-1];
		// threshold: the max index of a block in this dimension

		// find the smallest size of blocks along one of the dimensions
		if (block_counter_threshold[i-1] > min_block) {
			min_block = block_counter_threshold[i-1];
		}

		blocks *= block_counter_threshold[i-1];
		block_size *= tensor->block_layout[i-1];
		if (tensor->layout[i-1] % tensor->block_layout[i-1] != 0) {
			remainder_index[i-1] = tensor->layout[i-1] % tensor->block_layout[i-1];
		} else {
			remainder_index[i-1] = tensor->block_layout[i-1];
			// when it's not an "edgy" block-dim then just assign regular size
		}
	}
	size_t global_result = 0;
	size_t last_offset = 0;

	size_t morton_block_levels = log2(min_block)+1; // take only the front size_teger (round-down at all times)
	size_t * morton_block_indices = calloc(morton_block_levels, sizeof(size_t));
	//printf("level=%d", morton_block_levels);

	size_t tensor_index = 0;
	size_t tensor_offset = 0;

	size_t next = 0;
	size_t result_offset = 0;
	size_t vector_index = 0;
	//prsize_t_to_console_size_t(block_counter_threshold, dim);
	// BLOCK-LEVEL LOOP (using counter method)
	for (size_t b=0; b<blocks; ++b) {

		// TAKE A NEW BLOCK
		new_mul[dim-1] = 1;
		for (size_t d=dim-2; d<=dim-2; --d) {
			if (d+1 == mode) {
				// just carry over without multiplying
				new_mul[d] = new_mul[d+1];
				new_mul[d+1] = 0;
			} else {
				if (block_counter_threshold[d+1]-1 == block_counter[d+1]) {
					new_mul[d] = new_mul[d+1] * remainder_index[d+1];
				} else {
					new_mul[d] = new_mul[d+1] * tensor->block_layout[d+1];
				}
			}
		}

 		for (size_t d=dim-1; d<=dim-1; --d) {
			if (block_counter_threshold[d]-1 == block_counter[d]) {
				// assign the limit of this dimension!!!
				limits[d] = remainder_index[d];
			} else {
				// assign the actual max for this dimension
				limits[d] = tensor->block_layout[d];
			}
		}

		// TENSOR-LEVEL LOOP
	 	vector_index = 0;
		size_t * counter = calloc(dim, sizeof(size_t));

		for (size_t t=0; t<block_size; ++t) {

			debug_printf("result=%ld, ten=%ld:\n", global_result, tensor_index);

			vector_index = block_counter[mode] * tensor->block_layout[mode] + counter[mode];

			result_tensor->data[global_result+result_offset] +=
				tensor->lin.data[next++] * vector->data[vector_index];

			last_offset = result_offset;
			result_offset = 0;
			tensor_offset = 0;
			++counter[dim-1];
			for(size_t d=dim-1; d!=0; --d) {
				// two conditions to tick the counter
				// 1) it simply reaches the threshold
				// 2) it reaches the limit for this part dimension
				if (counter[d] == limits[d]) {
					if (d!=0) {
						++counter[d-1];
					}
					counter[d] = 0;
				}
				if (d != mode) {
					result_offset += counter[d] * new_mul[d];
				}
				tensor_offset += counter[d] * block_mul[d];
			}
			// handle the 0 case
			if (counter[0] == limits[0]) {
				global_result += last_offset + 1;
				break;
			}
			if (0 != mode) {
				result_offset += counter[0] * new_mul[0];
			}
			tensor_offset += counter[0] * block_mul[0];
		}
		
		// SETUP
		// this line affects all dimensions at once
		morton_inc_struct(block_counter, block_counter_threshold, dim-1, morton_metadata);

		// ALGORITHM 1
		// works for multiplies of 2
		//printf("lvl=%d\n",morton_metadata->level);
		// necessary for stopping at tricky positions
		if (morton_metadata->level >= morton_block_levels) {
			free(counter);
			break;
		}

		// if we need to reset the result
		// otherwise: the result is not reset (it's +1 of the last element of the last block)
		if (morton_metadata->dim==mode) {
			// perform a swap
			size_t temp = global_result;
			global_result = morton_block_indices[morton_metadata->level];
			morton_block_indices[morton_metadata->level] = temp;
			// this loop runs only if we ticked forward in a higher level (>0)
			for (size_t i=morton_metadata->level-1; i>=0; --i) {
				// then we must update all lower levels to current position
				morton_block_indices[i] = global_result;
			}
		}

		tensor_index = 0;
		for (size_t d=dim-1; d<=dim-1; --d) {
			tensor_index += block_counter[d] * block_mul[d] * tensor->block_layout[d];
		}

		free(counter);
	}

	free(morton_block_indices);
	free(morton_metadata);
	free(mul);
	free(new_mul);
	free(limits);
	free(block_mul);
	free(remainder_index);
	free(block_counter);
	free(block_counter_threshold);
#endif
}

#if 0
void tvm_morton_block_major_input_aligned_output_aligned_ME(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

	struct morton_data * const morton_metadata = calloc(1, sizeof(struct morton_data));

	size_t dim = tensor->dim;
	// counters
	size_t * block_counter = calloc(dim, sizeof(size_t));
	// limits
	size_t * block_counter_threshold = calloc(dim, sizeof(size_t));
	size_t * new_mul = calloc(dim, sizeof(size_t));
	size_t * remainder_index = calloc(dim, sizeof(size_t));
	size_t * limits = calloc(dim, sizeof(size_t));
	// calcs
	size_t * mul = calloc(dim, sizeof(size_t));
	size_t * block_mul = calloc(dim, sizeof(size_t));
	//size_t * new_mul = calloc(dim, sizeof(size_t));
	size_t block_size = 1;
	size_t blocks = 1;
	mul[dim-1] = 1;
	block_mul[dim-1] = 1;
	size_t min_block = 0;

	for (size_t i=dim; i!=0; --i) {
		if (i-1>0) {
			if (i-1 == mode) {
				mul[i-2] = mul[i-1];
			} else {
				mul[i-2] = mul[i-1] * tensor->layout[tensor->layout_perm[i-1]];
			}
			block_mul[i-2] = block_mul[i-1] * tensor->layout[tensor->layout_perm[i-1]];
		}

		// size_teger division with rounding up formula :)
		block_counter_threshold[i-1] = (tensor->layout[i-1] + tensor->block_layout[i-1] -1)
			/ tensor->block_layout[i-1];
		// threshold: the max index of a block in this dimension

		// find the smallest size of blocks along one of the dimensions
		if (block_counter_threshold[i-1] > min_block) {
			min_block = block_counter_threshold[i-1];
		}

		blocks *= block_counter_threshold[i-1];
		block_size *= tensor->block_layout[i-1];
		if (tensor->layout[i-1] % tensor->block_layout[i-1] != 0) {
			remainder_index[i-1] = tensor->layout[i-1] % tensor->block_layout[i-1];
		} else {
			remainder_index[i-1] = tensor->block_layout[i-1];
			// when it's not an "edgy" block-dim then just assign regular size
		}
	}
	size_t global_result = 0;
	size_t last_offset = 0;

	size_t morton_block_levels = log2(min_block)+1; // take only the front size_teger (round-down at all times)
	size_t * morton_block_indices = calloc(morton_block_levels, sizeof(size_t));
	//printf("level=%d", morton_block_levels);

	size_t tensor_index = 0;
	size_t tensor_offset = 0;

	size_t next = 0;
	size_t result_offset = 0;
	size_t vector_index = 0;
	//prsize_t_to_console_size_t(block_counter_threshold, dim);
	// BLOCK-LEVEL LOOP (using counter method)
	for (size_t b=0; b<blocks; ++b) {

		debug_printf("started block %d\n", b);

		// TAKE A NEW BLOCK
		new_mul[dim-1] = 1;
		for (size_t d=dim-2; d<=dim-2; --d) {
			if (d+1 == mode) {
				// just carry over without multiplying
				new_mul[d] = new_mul[d+1];
				new_mul[d+1] = 0;
			} else {
				if (block_counter_threshold[d+1]-1 == block_counter[d+1]) {
					new_mul[d] = new_mul[d+1] * remainder_index[d+1];
				} else {
					new_mul[d] = new_mul[d+1] * tensor->block_layout[d+1];
				}
			}
		}

 		for (size_t d=dim-1; d<=dim-1; --d) {
			if (block_counter_threshold[d]-1 == block_counter[d]) {
				// assign the limit of this dimension!!!
				limits[d] = remainder_index[d];
			} else {
				// assign the actual max for this dimension
				limits[d] = tensor->block_layout[d];
			}
		}

		// TENSOR-LEVEL LOOP
	 	vector_index = 0;
		size_t * counter = calloc(dim, sizeof(size_t));

		for (size_t t=0; t<block_size; ++t) {

			debug_printf("result=%ld, ten=%ld:\n", global_result, tensor_index);

			vector_index = block_counter[mode] * tensor->block_layout[mode] + counter[mode];


			debug_printf("vector_index = block_counter[mode](%d) * tensor->block_layout[mode](%d) + counter[mode](%d)",
					block_counter[mode],
					tensor->block_layout[mode],
					counter[mode]);

			result_tensor->data[global_result+result_offset] +=
				tensor->lin.data[next++] * vector->data[vector_index];

			last_offset = result_offset;
			result_offset = 0;
			tensor_offset = 0;
			++(counter[dim-1]);
			for(size_t d=dim-1; d!=0; --d) {
				// two conditions to tick the counter
				// 1) it simply reaches the threshold
				// 2) it reaches the limit for this part dimension
				if (counter[d] == limits[d]) {
					if (d!=0) {
						++counter[d-1];
					}
					counter[d] = 0;
				}
				if (d != mode) {
					result_offset += counter[d] * new_mul[d];
				}
				tensor_offset += counter[d] * block_mul[d];
			}
			// handle the 0 case
			if (counter[0] == limits[0]) {
				global_result += last_offset + 1;
				break;
			}
			if (0 != mode) {
				result_offset += counter[0] * new_mul[0];
			}
			tensor_offset += counter[0] * block_mul[0];
		}
		
		// SETUP
		// this line affects all dimensions at once
		morton_inc_struct(block_counter, block_counter_threshold, dim-1, morton_metadata);

#if 0
		// ALGORITHM 1
		// works for multiplies of 2
		printf("lvl=%d\n",morton_metadata->level);
		// necessary for stopping at tricky positions
		if (morton_metadata->level >= morton_block_levels) {
			break;
		}
		// if we need to reset the result
		// otherwise: the result is not reset (it's +1 of the last element of the last block)
		if (morton_metadata->dim==mode) {
			// perform a swap
			size_t temp = global_result;
			global_result = morton_block_indices[morton_metadata->level];
			morton_block_indices[morton_metadata->level] = temp;
			// this loop runs only if we ticked forward in a higher level (>0)
			for (size_t i=morton_metadata->level-1; i>=0; --i) {
				// then we must update all lower levels to current position
				morton_block_indices[i] = global_result;
			}
		}
#endif 

		// ALGORITHM 2
		//printf("lvl=%d\n",morton_metadata->level);
		// necessary for stopping at tricky positions
		if (morton_metadata->level >= morton_block_levels) {
			free(counter);
			break;
		}
		// if we need to reset the result
		// otherwise: the result is not reset (it's +1 of the last element of the last block)
		if (morton_metadata->dim==mode) {
			//printf("==:\n");
			// perform a partial swap
			size_t temp = global_result;
			debug_printf("global_result = %d = morton[%d]\n", morton_block_indices[morton_metadata->level], morton_metadata->level);
			global_result = morton_block_indices[morton_metadata->level];

			// GUARD CODE: find largest level which fits the mask
			size_t diff = block_counter_threshold[mode] - 1 - block_counter[mode];
			size_t current_level = morton_metadata->level;
			debug_printf("current_level=%d\n", current_level);
			// Find the max level it fits for the lowest diff
			// start with the given mask
			// run only if mask does NOT fit the diff (go level lower!)
			// in fact, even if it fits the mask of next dim, it should be fine?
			// on each iteration, shift mask to the right

			// we must really let here the number to go down!!!
			debug_printf("->level=%d, current_level=%d \n", morton_metadata->level, current_level);
			size_t max_level = current_level;
			for(size_t mask = morton_metadata->mask; mask > diff && max_level>-1; mask>>=1) {
				debug_printf("decrease...: mask=%d, level=%d\n", mask, morton_metadata->level);
				// decrease the level until we find one we can update
				max_level-=1;
			}

			debug_printf("max_level: %d\n", max_level);

			size_t swapped_highest = 0;

			debug_printf("->level=%d, current_level=%d \n", morton_metadata->level, current_level);

			// perform finishing swap (from the level it allows!)
			// NO: perform it anyway (from current_level)
			//prsize_t_to_console_size_t(morton_block_indices, morton_block_levels);

			if (temp < result_tensor->size) {
				morton_block_indices[morton_metadata->level] = temp;
				swapped_highest = 1;

				if (max_level == morton_metadata->level) {
					max_level -= 1;
				}

				debug_printf("we finished the swap, now, morton[%d]=%d\n", morton_metadata->level, temp);
			} else {
				//printf("we iterated to global_result which is out of bounds: not touching it!\n");
			}
			//prsize_t_to_console_size_t(morton_block_indices, morton_block_levels);

			//if (swapped_highest == 0) {
				//max_level+=1;
			//}
			//printf("max_level=%d\n", max_level);

			// perform a trickle down from max_level-1 (i.e. for remaining dimensions)
			debug_printf("temp=%d, result_tensor->size=%ld\n", temp, result_tensor->size);
			for(size_t i=max_level; i>=0; --i) {
				// ADDITIONAL GUARD again
				if (swapped_highest) {
					// because we swapped the highest dimension...
					// that means it's also okay to reset lowest to that new position
					//printf("we swapped highest dimension: as in algo1, assign global_result to lower\n");
					morton_block_indices[i] = global_result;
				} else {
					if (temp < result_tensor->size) {
						//printf("we didn't swap highest dimension: unlike algo1, assign next global to lower\n");
						morton_block_indices[i] = temp;
					} else {
						// do nothing?
						//printf("do nothing: we cannot assign any number\n");
						morton_block_indices[i] = global_result;
					}
				}
			}

			//prsize_t_to_console_size_t(morton_block_indices, morton_block_levels);
			//printf("\n");
		} else {
			//printf("!!, ");
		}

		tensor_index = 0;
		for (size_t d=dim-1; d<=dim-1; --d) {
			tensor_index += block_counter[d] * block_mul[d] * tensor->block_layout[d];
		}

		free(counter);
	}

	free(morton_block_indices);
	free(morton_metadata);
	free(mul);
	free(new_mul);
	free(limits);
	free(block_mul);
	free(remainder_index);
	free(block_counter);
	free(block_counter_threshold);
} 
#endif

#if 0
void tvm_morton_block_major_input_aligned_output_aligned_copy(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

	struct morton_data * const morton_metadata = calloc(1, sizeof(struct morton_data));

	size_t dim = tensor->dim;
	// counters
	size_t * block_counter = calloc(dim, sizeof(size_t));
	// limits
	size_t * block_counter_threshold = calloc(dim, sizeof(size_t));
	size_t * new_mul = calloc(dim, sizeof(size_t));
	size_t * remainder_index = calloc(dim, sizeof(size_t));
	size_t * limits = calloc(dim, sizeof(size_t));
	// calcs
	size_t * mul = calloc(dim, sizeof(size_t));
	size_t * block_mul = calloc(dim, sizeof(size_t));
	//size_t * new_mul = calloc(dim, sizeof(size_t));
	size_t block_size = 1;
	size_t blocks = 1;
	mul[dim-1] = 1;
	block_mul[dim-1] = 1;
	size_t min_block = 0;

	for (size_t i=dim; i!=0; --i) {
		if (i-1>0) {
			if (i-1 == mode) {
				mul[i-2] = mul[i-1];
			} else {
				mul[i-2] = mul[i-1] * tensor->layout[tensor->layout_perm[i-1]];
			}
			block_mul[i-2] = block_mul[i-1] * tensor->layout[tensor->layout_perm[i-1]];
		}

		// size_teger division with rounding up formula :)
		block_counter_threshold[i-1] = (tensor->layout[i-1] + tensor->block_layout[i-1] -1)
			/ tensor->block_layout[i-1];
		// threshold: the max index of a block in this dimension

		// find the smallest size of blocks along one of the dimensions
		if (block_counter_threshold[i-1] > min_block) {
			min_block = block_counter_threshold[i-1];
		}

		blocks *= block_counter_threshold[i-1];
		block_size *= tensor->block_layout[i-1];
		if (tensor->layout[i-1] % tensor->block_layout[i-1] != 0) {
			remainder_index[i-1] = tensor->layout[i-1] % tensor->block_layout[i-1];
		} else {
			remainder_index[i-1] = tensor->block_layout[i-1];
			// when it's not an "edgy" block-dim then just assign regular size
		}
	}
	size_t global_result = 0;
	size_t last_offset = 0;

	size_t morton_block_levels = log2(min_block)+1; // take only the front size_teger (round-down at all times)
	size_t * morton_block_indices = calloc(morton_block_levels, sizeof(size_t));
	//printf("level=%d", morton_block_levels);

	size_t tensor_index = 0;
	size_t tensor_offset = 0;

	size_t next = 0;
	size_t result_offset = 0;
	size_t vector_index = 0;
	//prsize_t_to_console_size_t(block_counter_threshold, dim);
	// BLOCK-LEVEL LOOP (using counter method)
	for (size_t b=0; b<blocks; ++b) {

		// TAKE A NEW BLOCK
		new_mul[dim-1] = 1;
		for (size_t d=dim-2; d<=dim-2; --d) {
			if (d+1 == mode) {
				// just carry over without multiplying
				new_mul[d] = new_mul[d+1];
				new_mul[d+1] = 0;
			} else {
				if (block_counter_threshold[d+1]-1 == block_counter[d+1]) {
					new_mul[d] = new_mul[d+1] * remainder_index[d+1];
				} else {
					new_mul[d] = new_mul[d+1] * tensor->block_layout[d+1];
				}
			}
		}

 		for (size_t d=dim-1; d<=dim-1; --d) {
			if (block_counter_threshold[d]-1 == block_counter[d]) {
				// assign the limit of this dimension!!!
				limits[d] = remainder_index[d];
			} else {
				// assign the actual max for this dimension
				limits[d] = tensor->block_layout[d];
			}
		}

		// TENSOR-LEVEL LOOP
	 	vector_index = 0;
		size_t * counter = calloc(dim, sizeof(size_t));

		for (size_t t=0; t<block_size; ++t) {

			debug_printf("result=%ld, ten=%ld:\n", global_result, tensor_index);

			vector_index = block_counter[mode] * tensor->block_layout[mode] + counter[mode];

			result_tensor->data[global_result+result_offset] +=
				tensor->lin.data[next++] * vector->data[vector_index];

			last_offset = result_offset;
			result_offset = 0;
			tensor_offset = 0;
			++counter[dim-1];
			for(size_t d=dim-1; d!=0; --d) {
				// two conditions to tick the counter
				// 1) it simply reaches the threshold
				// 2) it reaches the limit for this part dimension
				if (counter[d] == limits[d]) {
					if (d!=0) {
						++counter[d-1];
					}
					counter[d] = 0;
				}
				if (d != mode) {
					result_offset += counter[d] * new_mul[d];
				}
				tensor_offset += counter[d] * block_mul[d];
			}
			// handle the 0 case
			if (counter[0] == limits[0]) {
				global_result += last_offset + 1;
				break;
			}
			if (0 != mode) {
				result_offset += counter[0] * new_mul[0];
			}
			tensor_offset += counter[0] * block_mul[0];
		}
		
		// SETUP
		// this line affects all dimensions at once
		morton_inc_struct(block_counter, block_counter_threshold, dim-1, morton_metadata);

#if 0
		// ALGORITHM 1
		// works for multiplies of 2
		printf("lvl=%d\n",morton_metadata->level);
		// necessary for stopping at tricky positions
		if (morton_metadata->level >= morton_block_levels) {
			break;
		}
		// if we need to reset the result
		// otherwise: the result is not reset (it's +1 of the last element of the last block)
		if (morton_metadata->dim==mode) {
			// perform a swap
			size_t temp = global_result;
			global_result = morton_block_indices[morton_metadata->level];
			morton_block_indices[morton_metadata->level] = temp;
			// this loop runs only if we ticked forward in a higher level (>0)
			for (size_t i=morton_metadata->level-1; i>=0; --i) {
				// then we must update all lower levels to current position
				morton_block_indices[i] = global_result;
			}
		}
#endif 

		// ALGORITHM 2
		//printf("lvl=%d\n",morton_metadata->level);
		// necessary for stopping at tricky positions
		if (morton_metadata->level >= morton_block_levels) {
			free(counter);
			break;
		}
		// if we need to reset the result
		// otherwise: the result is not reset (it's +1 of the last element of the last block)
		if (morton_metadata->dim==mode) {
			//printf("==:\n");
			// perform a partial swap
			size_t temp = global_result;
			debug_printf("global_result = %d = morton[%d]\n", morton_block_indices[morton_metadata->level], morton_metadata->level);
			global_result = morton_block_indices[morton_metadata->level];

			// GUARD CODE: find largest level which fits the mask
			size_t diff = block_counter_threshold[mode] - 1 - block_counter[mode];
			size_t current_level = morton_metadata->level;
			debug_printf("current_level=%d\n", current_level);
			// Find the max level it fits for the lowest diff
			// start with the given mask
			// run only if mask does NOT fit the diff (go level lower!)
			// in fact, even if it fits the mask of next dim, it should be fine?
			// on each iteration, shift mask to the right

			// we must really let here the number to go down!!!
			debug_printf("->level=%d, current_level=%d \n", morton_metadata->level, current_level);
			size_t max_level = current_level;
			for(size_t mask = morton_metadata->mask; mask > diff && max_level>-1; mask>>=1) {
				debug_printf("decrease...: mask=%d, level=%d\n", mask, morton_metadata->level);
				// decrease the level until we find one we can update
				max_level-=1;
			}

			debug_printf("max_level: %d\n", max_level);

			size_t swapped_highest = 0;

			debug_printf("->level=%d, current_level=%d \n", morton_metadata->level, current_level);

			// perform finishing swap (from the level it allows!)
			// NO: perform it anyway (from current_level)
			//prsize_t_to_console_size_t(morton_block_indices, morton_block_levels);

			if (temp < result_tensor->size) {
				morton_block_indices[morton_metadata->level] = temp;
				swapped_highest = 1;

				if (max_level == morton_metadata->level) {
					max_level -= 1;
				}

				debug_printf("we finished the swap, now, morton[%d]=%d\n", morton_metadata->level, temp);
			} else {
				//printf("we iterated to global_result which is out of bounds: not touching it!\n");
			}
			//prsize_t_to_console_size_t(morton_block_indices, morton_block_levels);

			//if (swapped_highest == 0) {
				//max_level+=1;
			//}
			//printf("max_level=%d\n", max_level);

			// perform a trickle down from max_level-1 (i.e. for remaining dimensions)
			debug_printf("temp=%d, result_tensor->size=%ld\n", temp, result_tensor->size);
			for(size_t i=max_level; i>=0; --i) {
				// ADDITIONAL GUARD again
				if (swapped_highest) {
					// because we swapped the highest dimension...
					// that means it's also okay to reset lowest to that new position
					//printf("we swapped highest dimension: as in algo1, assign global_result to lower\n");
					morton_block_indices[i] = global_result;
				} else {
					if (temp < result_tensor->size) {
						//printf("we didn't swap highest dimension: unlike algo1, assign next global to lower\n");
						morton_block_indices[i] = temp;
					} else {
						// do nothing?
						//printf("do nothing: we cannot assign any number\n");
						morton_block_indices[i] = global_result;
					}
				}
			}

			//prsize_t_to_console_size_t(morton_block_indices, morton_block_levels);
			//printf("\n");
		} else {
			//printf("!!, ");
		}

		tensor_index = 0;
		for (size_t d=dim-1; d<=dim-1; --d) {
			tensor_index += block_counter[d] * block_mul[d] * tensor->block_layout[d];
		}

		free(counter);
	}

	free(morton_block_indices);
	free(morton_metadata);
	free(mul);
	free(new_mul);
	free(limits);
	free(block_mul);
	free(remainder_index);
	free(block_counter);
	free(block_counter_threshold);
}
#endif

void
tvm_morton_block_major_input_aligned(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

	size_t dim = tensor->dim;
	// counters
	size_t * block_counter = calloc(dim, sizeof(size_t));
	// limits
	size_t * block_counter_threshold = calloc(dim, sizeof(size_t));
	size_t * remainder_index = calloc(dim, sizeof(size_t));
	size_t * limits = calloc(dim, sizeof(size_t));
	// calcs
	size_t * mul = calloc(dim, sizeof(size_t));
	size_t * block_mul = calloc(dim, sizeof(size_t));
	//size_t * new_mul = calloc(dim, sizeof(size_t));
	size_t block_size = 1;
	size_t blocks = 1;
	mul[dim-1] = 1;
	block_mul[dim-1] = 1;
	for (size_t i=dim; i!=0; --i) {
		if (i-1>0) {
			if (i-1 == mode) {
				mul[i-2] = mul[i-1];
			} else {
				mul[i-2] = mul[i-1] * tensor->layout[tensor->layout_perm[i-1]];
			}
			block_mul[i-2] = block_mul[i-1] * tensor->layout[tensor->layout_perm[i-1]];
		}
		// size_teger division with rounding up formula :)
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
	size_t next = 0;
	size_t result_offset = 0;
	size_t vector_index = 0;
	//prsize_t_to_console_size_t(block_counter_threshold, dim);
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
	 	vector_index = 0;
		size_t * counter = calloc(dim, sizeof(size_t));
		for (size_t t=0; t<block_size; ++t) {
			vector_index = block_counter[mode] * tensor->block_layout[mode] + counter[mode];
			result_tensor->data[result_index+result_offset] +=
				tensor->lin.data[next++] * vector->data[vector_index];
			result_offset = 0;
			++counter[dim-1];
			for(size_t d=dim-1; d!=0; --d) {
				// two conditions to tick the counter
				// 1) it simply reaches the threshold
				// 2) it reaches the limit for this part dimension
				if (counter[d] == limits[d]) {
					if (d!=0) {
						++counter[d-1];
					}
					if (d != mode) {
						result_offset = 0;
					}
					counter[d] = 0;
				}
				if (d != mode) {
					result_offset += counter[d] * mul[d];
				}
			}
			// handle the 0 case
			if (counter[0] == limits[0]) {
				break;
			}
			if (0 != mode) {
				result_offset += counter[0] * mul[0];
			}
		}

		// LAST CHANGE HERE!
		//result_index = 0;

		// this line affects all dimensions at once
		morton_inc(block_counter, block_counter_threshold, dim-1);
		//prsize_t_to_console_size_t(block_counter, dim);
		//printf("\n");
		// morton_inc in fact you can increment at most only one dimension
	
		// if you reset a dimension which is not a mode
		// i.e. we go back but in fact we do NOT compute completely different part of result because of reset
		// but we compute the same part of answer!

		// (but the same!) -> we must reset the result_index
		// if dimension is mul of 2: reset is undetected
		// if dimension is not mul of 2: we know exactly when
		result_index = 0;
		
		// brute force: recalculate each time (I think necessary!)
		for (size_t d=dim-1; d<=dim-1; --d) {
			if (d != mode) {
				result_index += block_counter[d] * mul[d] * tensor->block_layout[d];
			}
		}
		free(counter);

	}
	free(mul);
	free(limits);
	free(block_mul);
	free(remainder_index);
	free(block_counter);
	free(block_counter_threshold);
}

void
tvm_morton_block_major(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {
	size_t dim = tensor->dim;
	// counters
	size_t * block_counter = calloc(dim, sizeof(size_t));
	// limits
	size_t * block_counter_threshold = calloc(dim, sizeof(size_t));
	size_t * remainder_index = calloc(dim, sizeof(size_t));
	size_t * limits = calloc(dim, sizeof(size_t));
	// calcs
	size_t * mul = calloc(dim, sizeof(size_t));
	size_t * block_mul = calloc(dim, sizeof(size_t));
	//size_t * new_mul = calloc(dim, sizeof(size_t));
	size_t block_size = 1;
	size_t blocks = 1;
	mul[dim-1] = 1;
	block_mul[dim-1] = 1;
	for (size_t i=dim; i!=0; --i) {
		if (i-1>0) {
			if (i-1 == mode) {
				mul[i-2] = mul[i-1];
			} else {
				mul[i-2] = mul[i-1] * tensor->layout[tensor->layout_perm[i-1]];
			}
			block_mul[i-2] = block_mul[i-1] * tensor->layout[tensor->layout_perm[i-1]];
		}
		// size_teger division with rounding up formula :)
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
	size_t tensor_offset = 0;
	size_t vector_index = 0;
	//prsize_t_to_console_size_t(block_counter_threshold, dim);
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
	 	vector_index = 0;
		size_t * counter = calloc(dim, sizeof(size_t));
		for (size_t t=0; t<block_size; ++t) {
			vector_index = block_counter[mode] * tensor->block_layout[mode] + counter[mode];
			//printf("vector_ind=%d\n", vector_index);
			result_tensor->data[result_index+result_offset] +=
				tensor->lin.data[tensor_index+tensor_offset] * vector->data[vector_index];
			result_offset = 0;
			tensor_offset = 0;
			++counter[dim-1];
			for(size_t d=dim-1; d!=0; --d) {
				// two conditions to tick the counter
				// 1) it simply reaches the threshold
				// 2) it reaches the limit for this part dimension
				if (counter[d] == limits[d]) {
					if (d!=0) {
						++counter[d-1];
					}
					if (d != mode) {
						result_offset = 0;
					}
					counter[d] = 0;
				}
				if (d != mode) {
					result_offset += counter[d] * mul[d];
				}
				tensor_offset += counter[d] * block_mul[d];
			}
			// handle the 0 case
			if (counter[0] == limits[0]) {
				break;
			}
			if (0 != mode) {
				result_offset += counter[0] * mul[0];
			}
			tensor_offset += counter[0] * block_mul[0];
		}

		result_index = 0;
		tensor_index = 0;

		// this line affects all dimensions at once
		morton_inc(block_counter, block_counter_threshold, dim-1);	
		//prsize_t_to_console_size_t(block_counter, dim);
		//printf("\n");
		// morton_inc in fact you can increment at most only one dimension
	
		// if you reset a dimension which is not a mode
		// i.e. we go back but in fact we do NOT compute completely different part of result because of reset
		// but we compute the same part of answer!

		// (but the same!) -> we must reset the result_index
		// if dimension is mul of 2: reset is undetected
		// if dimension is not mul of 2: we know exactly when
		result_index = 0;
		
		// brute force: recalculate each time (I think necessary!)
		for (size_t d=dim-1; d<=dim-1; --d) {
			if (d != mode) {
				result_index += block_counter[d] * mul[d] * tensor->block_layout[d];
			}
			tensor_index += block_counter[d] * block_mul[d] * tensor->block_layout[d];
		}
		free(counter);

	}
	free(mul);
	free(limits);
	free(block_mul);
	free(remainder_index);
	free(block_counter);
	free(block_counter_threshold);
}

