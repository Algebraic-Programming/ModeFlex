#include <algorithms.h>
#include <rand_utils.h>
#include <file_utils.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <space_curves.h>
#include <mkl.h>
#include <immintrin.h>
#include <smmintrin.h>
#include <stdint.h>

void
tvm_hilbert_POWERS_libx(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

	// printf("Welcome to Hilbert curve LIBX TV algorithm because why not.\n");
	
	// initialize LIBXSMM
	// int prefetch = LIBXSMM_PREFETCH_AUTO;
	libxsmm_dmmfunction kernel;
	
	const size_t dim = tensor->dim;

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
	// size_t * const morton_block_indices2 = calloc(morton_block_levels, sizeof(size_t));
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
  		kernel = libxsmm_dmmdispatch(right_size, nn, kk, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
  	} else {
  		kernel = libxsmm_dmmdispatch(nn, result_size, kk, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
  	}

    size_t temp = block_counter_threshold[0];
    size_t nbits = 0;
    while (temp >>= 1) { ++nbits; }
    
		// print_to_console_sizet(block_counter, dim);
	size_t result_inddd = hilbert_c2i_result(dim, nbits, block_counter, mode);
	
		// printf("result index atthis point is %zu\n", result_inddd);

	size_t el = 0;
	while (1) {

		// printf("elements in the current block:\n");
		// print_to_console(tensor_ptr, block_size);
		// printf("we multiply this with the following vector (possibly in a loop):\n");
		// print_to_console(vector_ptr, vector_size);

		if (mode != dim-1) {
			
			next = 0;
			next_result = 0;
			for (size_t i=0; i<left_mat_size; ++i) {
			    const double *const tensor_next = tensor_ptr + i*mat_size;
			    double *const result_next = result_ptr + i*right_size;
			    kernel(tensor_next, vector_ptr, result_next);//, NULL, NULL, NULL);
			}
			
		} else {
			kernel(vector_ptr, tensor_ptr, result_ptr);//, NULL, NULL, NULL);
		}

		// printf("(whole) result after:\n");
		// print_to_console(result_tensor->data, result_tensor->size);

		if (++el == blocks) {
			break;
		}

		// old_global_vector = block_counter[mode];
		// global_result += result_size;
		tensor_ptr += block_size;

		// block_counters are calculated from the index (el)
		hilbert_incr(dim, nbits, block_counter);

		// potential optimization: if coord[mode] moved, then we simply do not change the result_ptr(!)

		// print_to_console_sizet(block_counter, dim);
		// size_t result_coord = 0;
		result_inddd = hilbert_c2i_result(dim, nbits, block_counter, mode);
		// printf("result index atthis point is %zu\n", result_inddd);
		result_ptr = base_result_ptr + (result_inddd * result_size);

		// printf("result_coord at this point its %zu\n", result_coord);
		// wow, job done :D

		// result_ptr = base_result_ptr + global_result;

		// VECTOR HAS TO CHANGE???
		global_vector = block_counter[mode] * tensor->block_layout[mode];
		vector_ptr = base_vector_ptr + global_vector;
	}

	free(morton_block_indices);
	free(block_counter);
	free(block_counter_threshold);
	free(mul);
}

void
tvm_hilbert_POWERS_mkl(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

	// printf("Welcome to Hilbert curve MKL TV algorithm because why not.\n");
	
	// initialize LIBXSMM
	// int prefetch = LIBXSMM_PREFETCH_AUTO;
	// libxsmm_dmmfunction kernel;
	
	const size_t dim = tensor->dim;

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
	// size_t * const morton_block_indices2 = calloc(morton_block_levels, sizeof(size_t));
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
   //  if (mode != dim-1) {
  	// 	kernel = libxsmm_dmmdispatch(right_size, nn, kk, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
  	// } else {
  	// 	kernel = libxsmm_dmmdispatch(nn, result_size, kk, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
  	// }

    size_t temp = block_counter_threshold[0];
    size_t nbits = 0;
    while (temp >>= 1) { ++nbits; }

	// print_to_console_sizet(block_counter, dim);
	size_t result_inddd = hilbert_c2i_result(dim, nbits, block_counter, mode);
	// printf("result index atthis point is %zu\n", result_inddd);

	size_t el = 0;
	while (1) {

		// printf("elements in the current block:\n");
		// print_to_console(tensor_ptr, block_size);
		// printf("we multiply this with the following vector (possibly in a loop):\n");
		// print_to_console(vector_ptr, vector_size);

		if (mode != dim-1) {
			next = 0;
			next_result = 0;
			for (size_t i=0; i<left_mat_size; ++i) {
			    const double *const tensor_next = tensor_ptr + i*mat_size;
			    double *const result_next = result_ptr + i*right_size;
			    // kernel(tensor_next, vector_ptr, result_next, NULL, NULL, NULL);
				cblas_dgemv(
					CblasRowMajor, // const CBLAS_LAYOUT
					CblasTrans, // const CBLAS_TRANSPOSE
					n, lda,
					alpha, // const double
					tensor_next, lda, // const double*, const MKL_size_t
					vector_ptr, incx, // const double*, const MKL_size_t
					beta, // const float
					result_next, incy); // const double*, const MKL_size_t
			}
		} else {
			cblas_dgemv(
				CblasRowMajor, // const CBLAS_LAYOUT
				CblasNoTrans, // const CBLAS_TRANSPOSE
				lda2, n,
				alpha, // const double
				tensor_ptr, n, // const double*, const MKL_size_t
				vector_ptr, incx, // const double*, const MKL_size_t
				beta, // const float
				result_ptr, incy); // const double*, const MKL_size_t
		}

		// printf("(whole) result after:\n");
		// print_to_console(result_tensor->data, result_tensor->size);

		if (++el == blocks) {
			break;
		}

		// old_global_vector = block_counter[mode];
		// global_result += result_size;
		tensor_ptr += block_size;

		// block_counters are calculated from the index (el)
		hilbert_incr(dim, nbits, block_counter);

		// potential optimization: if coord[mode] moved, then we simply do not change the result_ptr(!)
		// printf("one of elements %f:\n", result_tensor->data[0]);
// 
		// printf("BEFORE\n");
		// print_to_console_sizet(block_counter, dim);
		// size_t result_coord = 0;
		result_inddd = hilbert_c2i_result(dim, nbits, block_counter, mode);
			// printf("AFTER\n");

		// printf("result index atthis point is %zu\n", result_inddd);
		result_ptr = base_result_ptr + (result_inddd * result_size);

		// printf("one of elements %f:\n", result_tensor->data[0]);
		// printf("result_coord at this point its %zu\n", result_coord);
		// wow, job done :D

		// result_ptr = base_result_ptr + global_result;

		// VECTOR HAS TO CHANGE???
		global_vector = block_counter[mode] * tensor->block_layout[mode];
		vector_ptr = base_vector_ptr + global_vector;

		// break;
	}

	free(morton_block_indices);
	free(block_counter);
	free(block_counter_threshold);
	free(mul);
}
