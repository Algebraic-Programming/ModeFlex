#include <algorithms.h>
#include <string.h>
#include <stdlib.h>
#include <mkl.h>
#include <immintrin.h>
#include <smmintrin.h>
#include <stdint.h>
#include <tensorlibthreads.h>
#include <math.h>

// All functions here use pthread instead of mythread (no assert statemnets)

void
tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_consumer_prodonly(
	const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode, DTYPE * const restrict unfold, 
	buffer_t * const buffer) {

	const size_t dim = tensor->dim;
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
	size_t el = 0;

	pthread_cond_wait(&buffer->preface, &buffer->monitor_on_main);

	while (1) {
		
		++el;
		
		asm volatile ("nop" ::);

		pthread_mutex_lock(&buffer->monitor_begin);
		pthread_cond_signal(&buffer->steady_state);
		pthread_mutex_unlock(&buffer->monitor_begin);
		if (el == blocks-1) break;

	
		++el;
		
		asm volatile ("nop" ::);

		pthread_mutex_lock(&buffer->monitor_end);
		pthread_cond_signal(&buffer->steady_state);
		pthread_mutex_unlock(&buffer->monitor_end);
		if (el == blocks-1) break;

	}

	free(mul);
}

// void
// tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_consumer(
// 	const struct tensor_storage * __restrict__ const  tensor, const struct lin_storage * __restrict__ const vector, struct lin_storage * __restrict__ const  result_tensor, const size_t mode, DTYPE * const restrict unfold, 
// 	buffer_t * __restrict__ const  buffer) {

// 	// const int prefetch = LIBXSMM_PREFETCH_AUTO;
// 	libxsmm_dmmfunction kernel_2;

// 	// We use unfold from buffer NOT from argument (not the same???)
// 	// But actually it should point to the same object (so it makes buffer unnecessary???)
// 	DTYPE const * __restrict__ const unfold_1 = buffer->unfold_1;
// 	DTYPE const * __restrict__ const unfold_2 = buffer->unfold_2;

// 	/////////////////////////////////////////////////////////////////////////////////////////////////////////
// 	// 1. Calc all init variables
// 	const size_t dim = tensor->dim;
// 	size_t blocks = 1;
// 	size_t mul_mode = 1;
// 	size_t mul_left = 1;
// 	for (size_t d=dim-1; d<dim; --d) {
// 		size_t temp = (tensor->layout[d] + tensor->block_layout[d] - 1) / tensor->block_layout[d];
// 		blocks *= temp;
// 		if (d > mode) {
// 			mul_mode *= temp;
// 		} else if (d == mode) {
// 			mul_left = mul_mode * temp;
// 		}
// 	}

// 	// size_t * const mul = malloc(dim * sizeof(size_t));
// 	// mul[dim-1] = 1;
// 	// for (size_t d=dim-1; d<dim; --d) {
// 	// 	size_t temp = (tensor->layout[d] + tensor->block_layout[d] - 1) / tensor->block_layout[d];
// 	// 	blocks *= temp;
// 	// 	if (d!=0) {
// 	// 		mul[d-1] = mul[d] * temp;
// 	// 	}
// 	// }
// 	// compute: right, left, block, result sizes
// 	size_t right_size = 1;
// 	size_t block_size = 1;
// 	for (size_t d=dim-1; d<dim; --d) {
// 		if (d > mode) {
// 			right_size *= tensor->block_layout[tensor->layout_perm[d]];
// 		}
// 		block_size *= tensor->block_layout[tensor->layout_perm[d]];
// 	}
// 	const size_t vector_size = tensor->block_layout[mode];
// 	const size_t result_size = block_size / vector_size;
// 	size_t really_global_result = 0;
// 	size_t global_tensor = 0;
// 	size_t global_result = 0;
// 	size_t global_vector = 0;
// 	// BLAS call constants
// 	const double alpha = 1;
// 	const double beta = 1;
// 	const MKL_INT incx = 1;
// 	const MKL_INT incy = 1;
// 	const MKL_INT lda = vector_size;
// 	const MKL_INT n = result_size;
// 	size_t el = 0;
// 	size_t size;
// 	// __m128d memory_tmp;
// 	size_t diff_new = result_size * mul_mode;

// 	/////////////////////////////////////////////////////////////////////////////////////////////////////////
// 	// 2. The wait/init vars for the HT

// 	pthread_cond_wait(&buffer->preface, &buffer->monitor_on_main);

//     if (mode != dim-1) {
//   		kernel_2 = libxsmm_dmmdispatch(result_size, 1, vector_size, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
//   	} else {
// 		kernel_2 = libxsmm_dmmdispatch(1, result_size, vector_size, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
//   	}

//   	// if (!kernel_2) {
//   	// 	printf("Crash; Kernel not available\n");
//   	// 	exit(-1);
//   	// } else {
//   	// 	printf("All good!\n");
//   	// }

//     const double *const tensor_ptr;
// 	// const double *const vector_ptr = vector->data;
// 	// double *const result_ptr = result_tensor->data;
//     double * result_ptr = result_tensor->data;
//     const double * vector_ptr = vector->data;

// 	while (1) {

		
// 		/////////////////////////////////////////////////////////////////////////////////////////////////////////
// 		// 3. Loop (1)

// 		// printf("CONSUMER (1): block %zu (%zu)\n", el, blocks);
// 		// print_to_console(unfold_1, block_size);
// 		// print_to_console(vector->data + global_vector, vector_size);

// 		if (mode != dim-1) {
// 			kernel_2(unfold_1, vector_ptr, result_ptr, NULL, NULL, NULL);
// 	    } else {
// 	    	kernel_2(vector_ptr, unfold_1, result_ptr, NULL, NULL, NULL);
// 	    }


// 		// cblas_dgemv(
// 			// CblasRowMajor, // const CBLAS_LAYOUT
// 		// 	CblasNoTrans, // const CBLAS_TRANSPOSE
// 		// 	n, lda, // const MKL_size_t (s)
// 		// 	alpha, // const double
// 		// 	unfold_1, lda, // const double*, const MKL_size_t
// 		// 	(vector->data + global_vector), incx, // const double*, const MKL_size_t
// 		// 	beta, // const float
// 		// 	(result_tensor->data + global_result), incy); // const double*, const MKL_size_t
// 		// kernel_2((vector->data + global_vector), unfold_1, (result_tensor->data + global_result), NULL, NULL, NULL);

// 		// global_result += result_size;
// 		// global_tensor += block_size;
// 		++el;
// 		// for (size_t i=0; i<=mode; ++i) {
// 		// 	if (el % mul[i] == 0) {
// 		// 		if (i == mode) {
// 		// 			global_result = really_global_result;
// 		// 			global_vector += vector_size;
// 		// 		} else {
// 		// 			really_global_result = global_result;
// 		// 			global_vector = 0;
// 		// 		}
// 		// 		break; // quit - no need go further in the loop
// 		// 	}
// 		// }
// 		// asm volatile ("nop" ::);

// 		result_ptr += result_size;
// 		if (mode>0 && el % mul_left == 0) {
// 				vector_ptr = vector->data;
// 		} else if (el % mul_mode == 0) {
// 				result_ptr = result_ptr - diff_new;
// 				vector_ptr += vector_size;			
// 		}

// 		pthread_mutex_lock(&buffer->monitor_begin);
// 		pthread_cond_signal(&buffer->steady_state);
// 		pthread_mutex_unlock(&buffer->monitor_begin);
// 		if (el == blocks-1) break;

// 		/////////////////////////////////////////////////////////////////////////////////////////////////////////
// 		// 4. Loop (2)

// 		// printf("CONSUMER (2): block %zu (%zu)\n", el, blocks);
// 		// print_to_console(unfold_2, block_size);
// 		// print_to_console(vector->data + global_vector, vector_size);

// 		// reset it back to normal
// 		// size = block_size;
// 		// printf("unfold2\n");

// 		if (mode != dim-1) {
// 			kernel_2(unfold_2, vector_ptr, result_ptr, NULL, NULL, NULL);
// 	    } else {
// 	    	kernel_2(vector_ptr, unfold_2, result_ptr, NULL, NULL, NULL);
// 	    }
// 		// result_ptr += result_size;
// 		// if (mode>0 && el % mul_left == 0) {
// 		// 		vector_ptr = vector->data;
// 		// } else if (el % mul_mode == 0) {
// 		// 		result_ptr = result_ptr - diff_new;
// 		// 		vector_ptr += vector_size;			
// 		// }

// 		// cblas_dgemv(
// 		// 	CblasRowMajor, // const CBLAS_LAYOUT
// 		// 	CblasNoTrans, // const CBLAS_TRANSPOSE
// 		// 	n, lda, // const MKL_size_t (s)
// 		// 	alpha, // const double
// 		// 	unfold_2, lda, // const double*, const MKL_size_t
// 		// 	(vector->data + global_vector), incx, // const double*, const MKL_size_t
// 		// 	beta, // const float
// 		// 	(result_tensor->data + global_result), incy); // const double*, const MKL_size_t

// 		// kernel_2((vector->data + global_vector), unfold_2, (result_tensor->data + global_result), NULL, NULL, NULL);

// 		// global_result += result_size;
// 		// global_tensor += block_size;
// 		++el;
// 		// for (size_t i=0; i<=mode; ++i) {
// 		// 	if (el % mul[i] == 0) {
// 		// 		if (i == mode) {
// 		// 			global_result = really_global_result;
// 		// 			global_vector += vector_size;
// 		// 		} else {
// 		// 			really_global_result = global_result;
// 		// 			global_vector = 0;
// 		// 		}
// 		// 		break; // quit - no need go further in the loop
// 		// 	}
// 		// }
// 		// asm volatile ("nop" ::);

// 		result_ptr += result_size;
// 		if (mode>0 && el % mul_left == 0) {
// 				vector_ptr = vector->data;
// 		} else if (el % mul_mode == 0) {
// 				result_ptr = result_ptr - diff_new;
// 				vector_ptr += vector_size;			
// 		}

// 		pthread_mutex_lock(&buffer->monitor_end);
// 		pthread_cond_signal(&buffer->steady_state);
// 		pthread_mutex_unlock(&buffer->monitor_end);
// 		if (el == blocks-1) break;

// 	}

// 	if (el % 2 == 0) {
// 		// printf("CONSUMER (final | 1): %zu (%zu)\n", el, blocks);
// 		// print_to_console(unfold_1, block_size);		print_to_console(vector->data + global_vector, vector_size);
// 		// kernel_2((vector->data + global_vector), unfold_1, (result_tensor->data + global_result), NULL, NULL, NULL);

// 		if (mode != dim-1) {
// 			kernel_2(unfold_1, vector_ptr, result_ptr, NULL, NULL, NULL);
// 	    } else {
// 	    	kernel_2(vector_ptr, unfold_1, result_ptr, NULL, NULL, NULL);
// 	    }

// 		// cblas_dgemv(
// 		// 	CblasRowMajor, // const CBLAS_LAYOUT
// 		// 	CblasNoTrans, // const CBLAS_TRANSPOSE
// 		// 	n, lda, // const MKL_size_t (s)
// 		// 	alpha, // const double
// 		// 	unfold_1, lda, // const double*, const MKL_size_t
// 		// 	(vector->data + global_vector), incx, // const double*, const MKL_size_t
// 		// 	beta, // const float
// 		// 	(result_tensor->data + global_result), incy); // const double*, const MKL_size_t

// 	} else {
// 		// printf("CONSUMER (final | 2): %zu (%zu)\n", el, blocks);
// 		// print_to_console(unfold_2, block_size);		print_to_console(vector->data + global_vector, vector_size);
// 		// kernel_2((vector->data + global_vector), unfold_2, (result_tensor->data + global_result), NULL, NULL, NULL);

// 		if (mode != dim-1) {
// 			kernel_2(unfold_2, vector_ptr, result_ptr, NULL, NULL, NULL);
// 	    } else {
// 	    	kernel_2(vector_ptr, unfold_2, result_ptr, NULL, NULL, NULL);
// 	    }

// 		// cblas_dgemv(
// 		// 	CblasRowMajor, // const CBLAS_LAYOUT
// 		// 	CblasNoTrans, // const CBLAS_TRANSPOSE
// 		// 	n, lda, // const MKL_size_t (s)
// 		// 	alpha, // const double
// 		// 	unfold_2, lda, // const double*, const MKL_size_t
// 		// 	(vector->data + global_vector), incx, // const double*, const MKL_size_t
// 		// 	beta, // const float
// 		// 	(result_tensor->data + global_result), incy); // const double*, const MKL_size_t
// 	}

// 	// free(mul);
// }

void
tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_consumer(
	const struct tensor_storage * __restrict__ const  tensor, const struct lin_storage * __restrict__ const vector, struct lin_storage * __restrict__ const  result_tensor, const size_t mode, DTYPE * const restrict unfold, 
	buffer_t * __restrict__ const  buffer) {

	// const int prefetch = LIBXSMM_PREFETCH_AUTO;
	libxsmm_dmmfunction kernel_2;

	// We use unfold from buffer NOT from argument (not the same???)
	// But actually it should point to the same object (so it makes buffer unnecessary???)
	DTYPE const * __restrict__ const unfold_1 = buffer->unfold_1;
	DTYPE const * __restrict__ const unfold_2 = buffer->unfold_2;

	/////////////////////////////////////////////////////////////////////////////////////////////////////////
	// 1. Calc all init variables
	const size_t dim = tensor->dim;
	// size_t blocks = 1;
	// size_t mul_mode = 1;
	// size_t mul_left = 1;
	// for (size_t d=dim-1; d<dim; --d) {
	// 	size_t temp = (tensor->layout[d] + tensor->block_layout[d] - 1) / tensor->block_layout[d];
	// 	blocks *= temp;
	// 	if (d > mode) {
	// 		mul_mode *= temp;
	// 	} else if (d == mode) {
	// 		mul_left = mul_mode * temp;
	// 	}
	// }

	// MORTON SHIIIIIIIIIIIIIIIIIIIIIIIT

	// Morton stuff (1)
	size_t * const block_counter = calloc(dim, sizeof(size_t));
	size_t * const block_counter_threshold = calloc(dim, sizeof(size_t));


	size_t mul_mode = 1;
	size_t mul_left = 1;
	// compute: right, block, result sizes
	// + blocks, block_counter_thresholds, max_block
	size_t right_size = 1;
	size_t block_size = 1;
	size_t blocks = 1;
	size_t max_block = 0;

	size_t * const mul = malloc(dim * sizeof(size_t));

	mul[dim-1] = 1;


	for (size_t i=dim-1; i<dim; --i) {
		/// BASICS
		size_t temp = tensor->block_layout[tensor->layout_perm[i]];
		if (i > mode) {
			right_size *= temp;
		}
		block_size *= temp;
		/// +
		block_counter_threshold[i] = (tensor->layout[i] + temp -1) / temp;
		blocks *= block_counter_threshold[i];
		if (block_counter_threshold[i] > max_block) {
			max_block = block_counter_threshold[i];
		}


		if (i > mode) {
			mul_mode *= block_counter_threshold[i];
		} else if (i == mode) {
			mul_left = mul_mode * block_counter_threshold[i];
		}

		if (i!=0) {
			mul[i-1] = mul[i] * block_counter_threshold[i];
		}
	}

	const size_t vector_size = tensor->block_layout[mode];
	const size_t result_size = block_size / vector_size;
	const size_t mat_size = right_size * vector_size;
	const size_t left_mat_size = block_size / mat_size;

	// Morton stuff (2)
	const size_t morton_block_levels = log2(max_block)+1; // round-up (take front size_teger and add 1)
	size_t * const morton_block_indices = calloc(morton_block_levels, sizeof(size_t));

	// MORTON SHIIIIIIIIIIIIIIIIIIIIIIIT

	// size_t * const mul = malloc(dim * sizeof(size_t));
	// mul[dim-1] = 1;
	// for (size_t d=dim-1; d<dim; --d) {
	// 	size_t temp = (tensor->layout[d] + tensor->block_layout[d] - 1) / tensor->block_layout[d];
	// 	blocks *= temp;
	// 	if (d!=0) {
	// 		mul[d-1] = mul[d] * temp;
	// 	}
	// }
	// compute: right, left, block, result sizes
	// size_t right_size = 1;
	// size_t block_size = 1;
	// for (size_t d=dim-1; d<dim; --d) {
	// 	if (d > mode) {
	// 		right_size *= tensor->block_layout[tensor->layout_perm[d]];
	// 	}
	// 	block_size *= tensor->block_layout[tensor->layout_perm[d]];
	// }
	// const size_t vector_size = tensor->block_layout[mode];
	// const size_t result_size = block_size / vector_size;
	// size_t really_global_result = 0;
	// size_t global_tensor = 0;
	// size_t global_result = 0;
	// size_t global_vector = 0;
	// // BLAS call constants
	// const double alpha = 1;
	// const double beta = 1;
	// const MKL_INT incx = 1;
	// const MKL_INT incy = 1;
	// const MKL_INT lda = vector_size;
	// const MKL_INT n = result_size;
	// size_t el = 0;
	// size_t size;
	// // __m128d memory_tmp;
	// size_t diff_new = result_size * mul_mode;

	/////////////////////////////////////////////////////////////////////////////////////////////////////////
	// 2. The wait/init vars for the HT

	size_t global_tensor = 0;
	size_t global_result = 0;
	size_t global_vector = 0;
	size_t old_global_vector = 0;

	size_t next;
	size_t next_result;

	// BLAS call constants
	const double alpha = 1;
	const double beta = 1;
	const MKL_INT incx = 1;
	const MKL_INT incy = 1;
	const MKL_INT lda = right_size;
	const MKL_INT lda2 = result_size;
	const MKL_INT n = vector_size;

	// MORTON-CURVE ONLY (3)
	size_t mask;
	size_t level;
	size_t inc_game;
	size_t offset;

	int block_diff;
	double block_diff_log;

    const int nn = 1;
    const int kk = vector_size;
    const double * tensor_ptr = tensor->lin.data;
    const double * vector_ptr = vector->data;
    double * result_ptr = result_tensor->data;
    double * base_result_ptr = result_tensor->data;
    const double * base_vector_ptr = vector->data;
    /* JIT Kernel */
    if (mode != dim-1) {
  		kernel_2 = libxsmm_dmmdispatch(result_size, nn, kk, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
  	} else {
  		kernel_2 = libxsmm_dmmdispatch(nn, result_size, kk, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
  	}
	pthread_cond_wait(&buffer->preface, &buffer->monitor_on_main);

	size_t el = 0;
  	// if (!kernel_2) {
  	// 	printf("Crash; Kernel not available\n");
  	// 	exit(-1);
  	// } else {
  	// 	printf("All good!\n");
  	// }

    // const double *const tensor_ptr;
	// const double *const vector_ptr = vector->data;
	// double *const result_ptr = result_tensor->data;
    // double * result_ptr = result_tensor->data;
    // const double * vector_ptr = vector->data;

	while (1) {

		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		// 3. Loop (1)

		// printf("CONSUMER (1): block %zu (%zu)\n", el, blocks);
		// print_to_console(unfold_1, block_size);
		// print_to_console(vector->data + global_vector, vector_size);

		if (mode != dim-1) {
			kernel_2(unfold_1, vector_ptr, result_ptr);//, NULL, NULL, NULL);
	    } else {
	    	kernel_2(vector_ptr, unfold_1, result_ptr);//, NULL, NULL, NULL);
	    }

		++el;
		// if (++el == blocks) {
			// break;
		// }

		old_global_vector = block_counter[mode];
		global_result += result_size;
		tensor_ptr += block_size;

		mask = 1;
		level = 0;
		inc_game = 1;
		offset = dim-1;
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

		// RESULT HAS TO CHANGE(!)
		// Perhaps we can use this as an indication of the change...
		if (offset == mode) {

			size_t temp = global_result;
			global_result = morton_block_indices[level];
			morton_block_indices[level] = temp;
	
			 block_diff = block_counter_threshold[mode] - block_counter[mode];

			if (block_diff != 0) {
				 block_diff_log = log2(block_diff);
				if (block_diff_log == (int) block_diff_log) {
					block_diff = block_diff_log;
				} else {
					block_diff = block_diff_log+1;
				}
			} else {
				block_diff = 0;
			}

			if (block_diff < level) {
				if (block_diff > 0) {
				for (size_t i=0; i<=block_diff-1; ++i) {
					morton_block_indices[i] = global_result;
				}
				}
			} else {
				if (level > 0) {
				for (size_t i=0; i<=level-1; ++i) {
					morton_block_indices[i] = global_result;
				}
				}
			}
		}
		result_ptr = base_result_ptr + global_result;

		// VECTOR HAS TO CHANGE???
		global_vector = block_counter[mode] * tensor->block_layout[mode];
		vector_ptr = base_vector_ptr + global_vector;

		// cblas_dgemv(
			// CblasRowMajor, // const CBLAS_LAYOUT
		// 	CblasNoTrans, // const CBLAS_TRANSPOSE
		// 	n, lda, // const MKL_size_t (s)
		// 	alpha, // const double
		// 	unfold_1, lda, // const double*, const MKL_size_t
		// 	(vector->data + global_vector), incx, // const double*, const MKL_size_t
		// 	beta, // const float
		// 	(result_tensor->data + global_result), incy); // const double*, const MKL_size_t
		// kernel_2((vector->data + global_vector), unfold_1, (result_tensor->data + global_result), NULL, NULL, NULL);

		// global_result += result_size;
		// global_tensor += block_size;
		// ++el;
		// for (size_t i=0; i<=mode; ++i) {
		// 	if (el % mul[i] == 0) {
		// 		if (i == mode) {
		// 			global_result = really_global_result;
		// 			global_vector += vector_size;
		// 		} else {
		// 			really_global_result = global_result;
		// 			global_vector = 0;
		// 		}
		// 		break; // quit - no need go further in the loop
		// 	}
		// }
		// asm volatile ("nop" ::);

		// result_ptr += result_size;
		// if (mode>0 && el % mul_left == 0) {
		// 		vector_ptr = vector->data;
		// } else if (el % mul_mode == 0) {
		// 		result_ptr = result_ptr - diff_new;
		// 		vector_ptr += vector_size;			
		// }

		pthread_mutex_lock(&buffer->monitor_begin);
		pthread_cond_signal(&buffer->steady_state);
		pthread_mutex_unlock(&buffer->monitor_begin);
		if (el == blocks-1) break;

		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		// 4. Loop (2)

		// printf("CONSUMER (2): block %zu (%zu)\n", el, blocks);
		// print_to_console(unfold_2, block_size);
		// print_to_console(vector->data + global_vector, vector_size);

		// reset it back to normal
		// size = block_size;
		// printf("unfold2\n");

		if (mode != dim-1) {
			kernel_2(unfold_2, vector_ptr, result_ptr);//, NULL, NULL, NULL);
	    } else {
	    	kernel_2(vector_ptr, unfold_2, result_ptr);//, NULL, NULL, NULL);
	    }

		// result_ptr += result_size;
		// if (mode>0 && el % mul_left == 0) {
		// 		vector_ptr = vector->data;
		// } else if (el % mul_mode == 0) {
		// 		result_ptr = result_ptr - diff_new;
		// 		vector_ptr += vector_size;			
		// }

		// cblas_dgemv(
		// 	CblasRowMajor, // const CBLAS_LAYOUT
		// 	CblasNoTrans, // const CBLAS_TRANSPOSE
		// 	n, lda, // const MKL_size_t (s)
		// 	alpha, // const double
		// 	unfold_2, lda, // const double*, const MKL_size_t
		// 	(vector->data + global_vector), incx, // const double*, const MKL_size_t
		// 	beta, // const float
		// 	(result_tensor->data + global_result), incy); // const double*, const MKL_size_t

		// kernel_2((vector->data + global_vector), unfold_2, (result_tensor->data + global_result), NULL, NULL, NULL);

		// global_result += result_size;
		// global_tensor += block_size;
		// ++el;
		// for (size_t i=0; i<=mode; ++i) {
		// 	if (el % mul[i] == 0) {
		// 		if (i == mode) {
		// 			global_result = really_global_result;
		// 			global_vector += vector_size;
		// 		} else {
		// 			really_global_result = global_result;
		// 			global_vector = 0;
		// 		}
		// 		break; // quit - no need go further in the loop
		// 	}
		// }
		// asm volatile ("nop" ::);

		// result_ptr += result_size;
		// if (mode>0 && el % mul_left == 0) {
		// 		vector_ptr = vector->data;
		// } else if (el % mul_mode == 0) {
		// 		result_ptr = result_ptr - diff_new;
		// 		vector_ptr += vector_size;			
		// }

		++el;
		// if (++el == blocks) {
		// 	break;
		// }

		old_global_vector = block_counter[mode];
		global_result += result_size;
		tensor_ptr += block_size;

		mask = 1;
		level = 0;
		inc_game = 1;
		offset = dim-1;
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

		// RESULT HAS TO CHANGE(!)
		// Perhaps we can use this as an indication of the change...
		if (offset == mode) {

			size_t temp = global_result;
			global_result = morton_block_indices[level];
			morton_block_indices[level] = temp;
	
			 block_diff = block_counter_threshold[mode] - block_counter[mode];

			if (block_diff != 0) {
				 block_diff_log = log2(block_diff);
				if (block_diff_log == (int) block_diff_log) {
					block_diff = block_diff_log;
				} else {
					block_diff = block_diff_log+1;
				}
			} else {
				block_diff = 0;
			}

			if (block_diff < level) {
				if (block_diff > 0) {
				for (size_t i=0; i<=block_diff-1; ++i) {
					morton_block_indices[i] = global_result;
				}
				}
			} else {
				if (level > 0) {
				for (size_t i=0; i<=level-1; ++i) {
					morton_block_indices[i] = global_result;
				}
				}
			}
		}
		result_ptr = base_result_ptr + global_result;

		// VECTOR HAS TO CHANGE???
		global_vector = block_counter[mode] * tensor->block_layout[mode];
		vector_ptr = base_vector_ptr + global_vector;

		pthread_mutex_lock(&buffer->monitor_end);
		pthread_cond_signal(&buffer->steady_state);
		pthread_mutex_unlock(&buffer->monitor_end);
		if (el == blocks-1) break;

	}

	if (el % 2 == 0) {
		// printf("CONSUMER (final | 1): %zu (%zu)\n", el, blocks);
		// print_to_console(unfold_1, block_size);		print_to_console(vector->data + global_vector, vector_size);
		// kernel_2((vector->data + global_vector), unfold_1, (result_tensor->data + global_result), NULL, NULL, NULL);

		if (mode != dim-1) {
			kernel_2(unfold_1, vector_ptr, result_ptr);//, NULL, NULL, NULL);
	    } else {
	    	kernel_2(vector_ptr, unfold_1, result_ptr);//, NULL, NULL, NULL);
	    }

		// cblas_dgemv(
		// 	CblasRowMajor, // const CBLAS_LAYOUT
		// 	CblasNoTrans, // const CBLAS_TRANSPOSE
		// 	n, lda, // const MKL_size_t (s)
		// 	alpha, // const double
		// 	unfold_1, lda, // const double*, const MKL_size_t
		// 	(vector->data + global_vector), incx, // const double*, const MKL_size_t
		// 	beta, // const float
		// 	(result_tensor->data + global_result), incy); // const double*, const MKL_size_t

	} else {
		// printf("CONSUMER (final | 2): %zu (%zu)\n", el, blocks);
		// print_to_console(unfold_2, block_size);		print_to_console(vector->data + global_vector, vector_size);
		// kernel_2((vector->data + global_vector), unfold_2, (result_tensor->data + global_result), NULL, NULL, NULL);

		if (mode != dim-1) {
			kernel_2(unfold_2, vector_ptr, result_ptr);//, NULL, NULL, NULL);
	    } else {
	    	kernel_2(vector_ptr, unfold_2, result_ptr);//, NULL, NULL, NULL);
	    }

		// cblas_dgemv(
		// 	CblasRowMajor, // const CBLAS_LAYOUT
		// 	CblasNoTrans, // const CBLAS_TRANSPOSE
		// 	n, lda, // const MKL_size_t (s)
		// 	alpha, // const double
		// 	unfold_2, lda, // const double*, const MKL_size_t
		// 	(vector->data + global_vector), incx, // const double*, const MKL_size_t
		// 	beta, // const float
		// 	(result_tensor->data + global_result), incy); // const double*, const MKL_size_t
	}

	free(morton_block_indices);
	free(block_counter);
	free(block_counter_threshold);
	// free(mul);
}

void
tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_consumer_multicore(
	const struct tensor_storage * __restrict__ const  tensor, const struct lin_storage * __restrict__ const  vector, struct lin_storage * __restrict__ const result_tensor, const size_t mode, DTYPE * const restrict unfold, 
	buffer_t * __restrict__ const  buffer) {

	// We use unfold from buffer NOT from argument (not the same???)
	// But actually it should point to the same object (so it makes buffer unnecessary???)
	DTYPE const * __restrict__ const  unfold_1 = buffer->unfold_1;
	DTYPE const * __restrict__ const  unfold_2 = buffer->unfold_2;
	
	/////////////////////////////////////////////////////////////////////////////////////////////////////////
	// 1. Calc all init variables

	const size_t dim = tensor->dim;
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
	// compute: right, left, block, result sizes
	size_t right_size = 1;
	size_t block_size = 1;
	for (size_t d=dim-1; d<dim; --d) {
		if (d > mode) {
			right_size *= tensor->block_layout[tensor->layout_perm[d]];
		}
		block_size *= tensor->block_layout[tensor->layout_perm[d]];
	}
	const size_t vector_size = tensor->block_layout[mode];
	const size_t result_size = block_size / vector_size;
	size_t really_global_result = 0;
	size_t global_tensor = 0;
	size_t global_result = 0;
	size_t global_vector = 0;
	// BLAS call constants
	const double alpha = 1;
	const double beta = 1;
	const MKL_INT incx = 1;
	const MKL_INT incy = 1;
	const MKL_INT lda = vector_size;
	const MKL_INT n = result_size;
	size_t el = 0;
	size_t size;
	// __m128d memory_tmp;

	/////////////////////////////////////////////////////////////////////////////////////////////////////////
	// 2. The wait/init vars for the HT

	pthread_cond_wait(&buffer->preface, &buffer->monitor_on_main);

	while (1) {
		
		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		// 3. Loop (1)

		// printf("CONSUMER (1): block %zu (%zu)\n", el, blocks);
		// print_to_console(unfold_1, block_size);
		// print_to_console(vector->data + global_vector, vector_size);

		size = block_size;
		cblas_dgemv(
			CblasRowMajor, // const CBLAS_LAYOUT
			CblasNoTrans, // const CBLAS_TRANSPOSE
			n, lda, // const MKL_size_t (s)
			alpha, // const double
			unfold_1, lda, // const double*, const MKL_size_t
			(vector->data + global_vector), incx, // const double*, const MKL_size_t
			beta, // const float
			(result_tensor->data + global_result), incy); // const double*, const MKL_size_t
		global_result += result_size;
		global_tensor += block_size;
		++el;
		for (size_t i=0; i<=mode; ++i) {
			if (el % mul[i] == 0) {
				if (i == mode) {
					global_result = really_global_result;
					global_vector += vector_size;
				} else {
					really_global_result = global_result;
					global_vector = 0;
				}
				break; // quit - no need go further in the loop
			}
		}
		// asm volatile ("nop" ::);

		pthread_mutex_lock(&buffer->monitor_begin);
		pthread_cond_signal(&buffer->steady_state);
		pthread_mutex_unlock(&buffer->monitor_begin);
		if (el == blocks-1) break;

		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		// 4. Loop (2)

		// printf("CONSUMER (2): block %zu (%zu)\n", el, blocks);
		// print_to_console(unfold_2, block_size);
		// print_to_console(vector->data + global_vector, vector_size);

		// reset it back to normal
		size = block_size;

		cblas_dgemv(
			CblasRowMajor, // const CBLAS_LAYOUT
			CblasNoTrans, // const CBLAS_TRANSPOSE
			n, lda, // const MKL_size_t (s)
			alpha, // const double
			unfold_2, lda, // const double*, const MKL_size_t
			(vector->data + global_vector), incx, // const double*, const MKL_size_t
			beta, // const float
			(result_tensor->data + global_result), incy); // const double*, const MKL_size_t
		global_result += result_size;
		global_tensor += block_size;
		++el;
		for (size_t i=0; i<=mode; ++i) {
			if (el % mul[i] == 0) {
				if (i == mode) {
					global_result = really_global_result;
					global_vector += vector_size;
				} else {
					really_global_result = global_result;
					global_vector = 0;
				}
				break; // quit - no need go further in the loop
			}
		}
		// asm volatile ("nop" ::);

		pthread_mutex_lock(&buffer->monitor_end);
		pthread_cond_signal(&buffer->steady_state);
		pthread_mutex_unlock(&buffer->monitor_end);
		if (el == blocks-1) break;

	}

	if (el % 2 == 0) {
		// printf("CONSUMER (final | 1): %zu (%zu)\n", el, blocks);
		// print_to_console(unfold_1, block_size);		print_to_console(vector->data + global_vector, vector_size);
		cblas_dgemv(
			CblasRowMajor, // const CBLAS_LAYOUT
			CblasNoTrans, // const CBLAS_TRANSPOSE
			n, lda, // const MKL_size_t (s)
			alpha, // const double
			unfold_1, lda, // const double*, const MKL_size_t
			(vector->data + global_vector), incx, // const double*, const MKL_size_t
			beta, // const float
			(result_tensor->data + global_result), incy); // const double*, const MKL_size_t

	} else {
		// printf("CONSUMER (final | 2): %zu (%zu)\n", el, blocks);
		// print_to_console(unfold_2, block_size);		print_to_console(vector->data + global_vector, vector_size);
		cblas_dgemv(
			CblasRowMajor, // const CBLAS_LAYOUT
			CblasNoTrans, // const CBLAS_TRANSPOSE
			n, lda, // const MKL_size_t (s)
			alpha, // const double
			unfold_2, lda, // const double*, const MKL_size_t
			(vector->data + global_vector), incx, // const double*, const MKL_size_t
			beta, // const float
			(result_tensor->data + global_result), incy); // const double*, const MKL_size_t
	}

	free(mul);
}


void *
tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_producer(void *arg) {
	// printf("producer: begin\n");
	/////////////////////////////////////////////////////////////////////////////////////////////////////////

	// buffer object comes as argument
	buffer_t *buffer = (buffer_t*)arg;

	// get both locks
	pthread_mutex_lock(&buffer->monitor_begin);
	pthread_mutex_lock(&buffer->monitor_end);

	while(1) {
		// Producer always come back to the point where it waits for the lock...
		pthread_mutex_lock(&buffer->monitor_on_main);
		// Do not proceed if tensor is NULL
		if (buffer->tensor == NULL) {
			break;
		}

		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Otherwise: proceed and prepare for computation by getting all those in place
		// 2. Init vars for TVM

		const struct tensor_storage const * tensor = buffer->tensor;

		// above: all const, below: no const(!) -> we modify in producer (but in consumer maybe const const)
		DTYPE * __restrict__ const unfold_1 = buffer->unfold_1;
		DTYPE * __restrict__ const unfold_2 = buffer->unfold_2;

		/////////////////////////////////////////////////////////////////////////////////////////////////////////

		const size_t mode = buffer->mode;
		const size_t dim = tensor->dim;
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
		size_t right_size = 1;
		size_t block_size = 1;
		for (size_t d=dim-1; d<dim; --d) {
			if (d > mode) {
				right_size *= tensor->block_layout[tensor->layout_perm[d]];
			}
			block_size *= tensor->block_layout[tensor->layout_perm[d]];
		}
		const size_t vector_size = tensor->block_layout[mode];
		const size_t result_size = block_size / vector_size;
		const size_t left_size = block_size / right_size / vector_size;
		size_t really_global_result = 0;
		size_t global_tensor = 0;
		size_t global_result = 0;
		size_t global_vector = 0;
		size_t size;

		DTYPE * tensor_ptr = tensor->lin.data;
		DTYPE * unfold_base = unfold_1;

		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		// 3. Wait for start signal
		size_t el = 1;

		if (mode != 0 & mode != (dim-1)) {
			for (size_t i=0; i<left_size; ++i) {
				for (size_t j=0; j<vector_size; ++j) {
					CopyWithSSEPrefetchNT(unfold_base + j*result_size, tensor_ptr, right_size);
					tensor_ptr += right_size;
				}
				unfold_base += right_size;
			}
			unfold_base = unfold_1;
		} else {
			CopyWithSSEPrefetchNT(unfold_1, tensor_ptr, block_size);
			tensor_ptr += block_size;
		}

		// DTYPE * source_ptr = tensor->lin.data;
		// DTYPE * unfold_ptr = unfold_1;
		// size = block_size;
		// CopyWithSSEPrefetchNT(unfold_1, tensor->lin.data, block_size);

		// global_result += result_size;
		// global_tensor += block_size;
		// for (size_t i=0; i<=mode; ++i) {
		// 	if (el % mul[i] == 0) {
		// 		if (i == mode) {
		// 			global_result = really_global_result;
		// 			global_vector += vector_size;
		// 		} else {
		// 			really_global_result = global_result;
		// 			global_vector = 0;
		// 		}
		// 		break; // quit - no need go further in the loop
		// 	}
		// }

		// printf("PRODUCER view:\n");
		// print_to_console(unfold_1, block_size);
		// int done = 1;
		// buffer->a = rand() % 50;
		// This will continue only if there is an appropriate wait on the monitor on main 
		pthread_cond_signal(&buffer->preface);
		pthread_mutex_unlock(&buffer->monitor_on_main);

		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		// 4. TVM loop

		while (1) {

			// DTYPE * source_ptr = tensor->lin.data + global_tensor;
			// DTYPE * unfold_ptr = unfold_2;
			// size = block_size;

			// nontemp_memcpy(unfold_2, tensor->lin.data + global_tensor, block_size);
			// CopyWithSSEPrefetchNT(unfold_2, tensor->lin.data + global_tensor, block_size);
			unfold_base = unfold_2;
			if (mode != 0 & mode != (dim-1)) {
				for (size_t i=0; i<left_size; ++i) {
					for (size_t j=0; j<vector_size; ++j) {
						CopyWithSSEPrefetchNT(unfold_base + j*result_size, tensor_ptr, right_size);
						tensor_ptr += right_size;
					}
					unfold_base += right_size;
				}
				unfold_base = unfold_2;
			} else {
				CopyWithSSEPrefetchNT(unfold_2, tensor_ptr, block_size);
				tensor_ptr += block_size;
			}

			pthread_cond_wait(&buffer->steady_state, &buffer->monitor_begin);
			// if (buffer->size == done) break;
			if (++el == blocks) break;

			// global_result += result_size;
			// global_tensor += block_size;
			// for (size_t i=0; i<=mode; ++i) {
			// 	if (el % mul[i] == 0) {
			// 		if (i == mode) {
			// 			global_result = really_global_result;
			// 			global_vector += vector_size;
			// 		} else {
			// 			really_global_result = global_result;
			// 			global_vector = 0;
			// 		}
			// 		break; // quit - no need go further in the loop
			// 	}
			// }

			/////////////////////////////////////////////////////////////////////////////////////////////////////////
			// 4. Loop (2)
			// source_ptr = tensor->lin.data + global_tensor;
			// unfold_ptr = unfold_1;
			// size = block_size;
			// nontemp_memcpy(unfold_1, tensor->lin.data + global_tensor, block_size);
			// CopyWithSSEPrefetchNT(unfold_1, tensor->lin.data + global_tensor, block_size);

			unfold_base = unfold_1;
			if (mode != 0 & mode != (dim-1)) {
				for (size_t i=0; i<left_size; ++i) {
					for (size_t j=0; j<vector_size; ++j) {
						CopyWithSSEPrefetchNT(unfold_base + j*result_size, tensor_ptr, right_size);
						tensor_ptr += right_size;
					}
					unfold_base += right_size;
				}
				unfold_base = unfold_1;
			} else {
				CopyWithSSEPrefetchNT(unfold_1, tensor_ptr, block_size);
				tensor_ptr += block_size;
			}

			pthread_cond_wait(&buffer->steady_state, &buffer->monitor_end);
			// if (buffer->size == done) break;
			if (++el == blocks) break;

			// global_result += result_size;
			// global_tensor += block_size;
			// for (size_t i=0; i<=mode; ++i) {
			// 	if (el % mul[i] == 0) {
			// 		if (i == mode) {
			// 			global_result = really_global_result;
			// 			global_vector += vector_size;
			// 		} else {
			// 			really_global_result = global_result;
			// 			global_vector = 0;
			// 		}
			// 		break; // quit - no need go further in the loop
			// 	}
			// }
		}

		free(mul);
	}

	// Jumps here ONLY if tensor is NULL
	pthread_mutex_unlock(&buffer->monitor_begin);
	pthread_mutex_unlock(&buffer->monitor_end);
	return NULL;

}

void *
tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_producer_multicore(void *arg) {
	// printf("producer: begin\n");
	/////////////////////////////////////////////////////////////////////////////////////////////////////////

	// buffer object comes as argument
	buffer_t *buffer = (buffer_t*)arg;

	// get both locks
	pthread_mutex_lock(&buffer->monitor_begin);
	pthread_mutex_lock(&buffer->monitor_end);

	while(1) {
		// Producer always come back to the point where it waits for the lock...
		pthread_mutex_lock(&buffer->monitor_on_main);
		// Do not proceed if tensor is NULL
		if (buffer->tensor == NULL) {
			break;
		}

		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Otherwise: proceed and prepare for computation by getting all those in place
		// 2. Init vars for TVM

		const struct tensor_storage const * tensor = buffer->tensor;

		// above: all const, below: no const(!) -> we modify in producer (but in consumer maybe const const)
		DTYPE * unfold_1 = buffer->unfold_1;
		DTYPE * unfold_2 = buffer->unfold_2;

		/////////////////////////////////////////////////////////////////////////////////////////////////////////

		const size_t mode = buffer->mode;
		const size_t dim = tensor->dim;
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
		size_t right_size = 1;
		size_t block_size = 1;
		for (size_t d=dim-1; d<dim; --d) {
			if (d > mode) {
				right_size *= tensor->block_layout[tensor->layout_perm[d]];
			}
			block_size *= tensor->block_layout[tensor->layout_perm[d]];
		}
		const size_t vector_size = tensor->block_layout[mode];
		const size_t result_size = block_size / vector_size;
		size_t really_global_result = 0;
		size_t global_tensor = 0;
		size_t global_result = 0;
		size_t global_vector = 0;
		size_t size;

		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		// 3. Wait for start signal
		size_t el = 1;

		// DTYPE * source_ptr = tensor->lin.data;
		// DTYPE * unfold_ptr = unfold_1;
		// size = block_size;
		CopyWithSSEPrefetchNT(unfold_1, tensor->lin.data, block_size);

		global_result += result_size;
		global_tensor += block_size;
		for (size_t i=0; i<=mode; ++i) {
			if (el % mul[i] == 0) {
				if (i == mode) {
					global_result = really_global_result;
					global_vector += vector_size;
				} else {
					really_global_result = global_result;
					global_vector = 0;
				}
				break; // quit - no need go further in the loop
			}
		}

		// printf("PRODUCER view:\n");
		// print_to_console(unfold_1, block_size);
		// int done = 1;
		// buffer->a = rand() % 50;
		// This will continue only if there is an appropriate wait on the monitor on main 
		pthread_cond_signal(&buffer->preface);
		pthread_mutex_unlock(&buffer->monitor_on_main);

		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		// 4. TVM loop

		while (1) {

			// DTYPE * source_ptr = tensor->lin.data + global_tensor;
			// DTYPE * unfold_ptr = unfold_2;
			// size = block_size;

			// nontemp_memcpy(unfold_2, tensor->lin.data + global_tensor, block_size);
			CopyWithSSEPrefetchNT(unfold_2, tensor->lin.data + global_tensor, block_size);
			pthread_cond_wait(&buffer->steady_state, &buffer->monitor_begin);
			// if (buffer->size == done) break;
			if (++el == blocks) break;

			global_result += result_size;
			global_tensor += block_size;
			for (size_t i=0; i<=mode; ++i) {
				if (el % mul[i] == 0) {
					if (i == mode) {
						global_result = really_global_result;
						global_vector += vector_size;
					} else {
						really_global_result = global_result;
						global_vector = 0;
					}
					break; // quit - no need go further in the loop
				}
			}

			/////////////////////////////////////////////////////////////////////////////////////////////////////////
			// 4. Loop (2)
			// source_ptr = tensor->lin.data + global_tensor;
			// unfold_ptr = unfold_1;
			// size = block_size;
			// nontemp_memcpy(unfold_1, tensor->lin.data + global_tensor, block_size);
			CopyWithSSEPrefetchNT(unfold_1, tensor->lin.data + global_tensor, block_size);
			pthread_cond_wait(&buffer->steady_state, &buffer->monitor_end);
			// if (buffer->size == done) break;
			if (++el == blocks) break;

			global_result += result_size;
			global_tensor += block_size;
			for (size_t i=0; i<=mode; ++i) {
				if (el % mul[i] == 0) {
					if (i == mode) {
						global_result = really_global_result;
						global_vector += vector_size;
					} else {
						really_global_result = global_result;
						global_vector = 0;
					}
					break; // quit - no need go further in the loop
				}
			}
		}

		free(mul);
	}

	// Jumps here ONLY if tensor is NULL
	pthread_mutex_unlock(&buffer->monitor_begin);
	pthread_mutex_unlock(&buffer->monitor_end);
	return NULL;

}
