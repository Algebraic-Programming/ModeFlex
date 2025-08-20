#include <algorithms.h>
#include <rand_utils.h>
#include <file_utils.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <mkl.h>
#include <immintrin.h>
#include <smmintrin.h>
#include <stdint.h>
#define PREFIXMODE 1
#include <gen_utils.h>
#include <omp.h>

void
tvm_vector_major_BLAS_col_benchmarkable(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode, DTYPE * const restrict unfold) {

	const size_t dim = tensor->dim;

	size_t right_size = 1;
	for (size_t i=dim-1; i>mode; --i) {
		right_size *= tensor->layout[tensor->layout_perm[i]];
	}
	size_t left_size = 1;
	for (size_t i=0; i<mode; ++i) {
		left_size *= tensor->layout[tensor->layout_perm[i]];
	}

	const MKL_INT n = result_tensor->size;
	const MKL_INT mode_size= vector->size;
	const MKL_INT lda = n;

	size_t result_size = tensor->lin.size / mode_size;
	size_t tensor_diff = 0;
	size_t tensor_index = 0;
	size_t next = 0;
	
	DTYPE * unfold_base = unfold;
	DTYPE * tensor_ptr = tensor->lin.data;

	for (size_t i=0; i<left_size; ++i) {
		for (size_t j=0; j<mode_size; ++j) {
			// memcpy(unfold_base + j*result_size, tensor_ptr, right_size*sizeof(DTYPE));
			CopyWithSSEPrefetchNT(unfold_base + j*result_size, tensor_ptr, right_size);
			tensor_ptr += right_size;
		}
		unfold_base += right_size;
	}

	const double alpha = 1;
	const double beta = 1;
	const MKL_INT incx = 1;
	const MKL_INT incy = 1;

	// cblas_dgemv(
	// 	CblasColMajor, // const CBLAS_LAYOUT
	// 	CblasNoTrans, // const CBLAS_TRANSPOSE
	// 	n, mode_size, // const MKL_size_t (s)
	// 	alpha, // const double
	// 	tensor->lin.data, lda, // const double*, const MKL_size_t
	// 	vector->data, incx, // const double*, const MKL_size_t
	// 	beta, // const float
	// 	result_tensor->data, incy); // const double*, const MKL_size_t
	
	cblas_dgemv(
		CblasColMajor, // const CBLAS_LAYOUT
		CblasNoTrans, // const CBLAS_TRANSPOSE
		n, mode_size, // const MKL_size_t (s)
		alpha, // const double
		unfold, lda, // const double*, const MKL_size_t
		vector->data, incx, // const double*, const MKL_size_t
		beta, // const float
		result_tensor->data, incy); // const double*, const MKL_size_t

}

void
tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_v3_mine(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

	// printf("HARD TO BELIEVE THIS IS CORRECT!\n");

	#pragma omp parallel
	{

	int my_rank = omp_get_thread_num();
	int world_size = omp_get_num_threads();
    size_t output_index = 0;

    // printf("Hello from processor %d out of %d!\n", my_rank, world_size);
    
	const size_t dim = tensor->dim;
	size_t blocks = 1;
	size_t mul_mode = 1;
	size_t mul_left = 1;
	for (size_t d=dim-1; d<dim; --d) {
		size_t temp = (tensor->layout[d] + tensor->block_layout[d] - 1) / tensor->block_layout[d];
		blocks *= temp;
		if (d > mode) {
			mul_mode *= temp;
		} else if (d == mode) {
			mul_left = mul_mode * temp;
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
	const size_t mat_size = (size_t) right_size * (size_t) vector_size;
	const size_t left_mat_size = block_size / mat_size;

	size_t really_global_result = 0;
	size_t global_tensor = 0;
	size_t global_result = 0;
	size_t global_vector = 0;

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

    const double * tensor_ptr = tensor->lin.data;
    double * result_ptr = result_tensor->data;
	// printf("mul_mode=%d, mul_lef=%d\n", mul_mode, mul_left);

	size_t el = 0;
	while (1) {

		if (output_index == my_rank) {

			if (mode != dim-1) {
				//printf("left_mat_size=%d, n=%d, lda=%d - lda2=%d \n", left_mat_size, n, lda, lda2);

				next = 0;
				next_result = 0;
				for (size_t i=0; i<left_mat_size; ++i) {
					
					const double *const tensor_next = (tensor_ptr + global_tensor) + i*mat_size;
				    double *const result_next = (result_ptr + global_result) + i*right_size;

					cblas_dgemv(
						CblasRowMajor, // const CBLAS_LAYOUT
						CblasTrans, // const CBLAS_TRANSPOSE
						n, lda,
						alpha, // const double
						tensor_next, lda, // const double*, const MKL_size_t
						(vector->data + global_vector), incx, // const double*, const MKL_size_t
						beta, // const float
						result_next, incy); // const double*, const MKL_size_t

				}
			} else {
				//printf("left_mat_size=%d, n=%d, lda=%d - lda2=%d \n", left_mat_size, n, lda, lda2);

				// try this code
				// next = 0;
				// next_result = 0;
				// for (size_t i=0; i<left_mat_size; ++i) {
				// 	cblas_dgemv(
				// 		CblasRowMajor, // const CBLAS_LAYOUT
				// 		CblasTrans, // const CBLAS_TRANSPOSE
				// 		n, lda,
				// 		alpha, // const double
				// 		(tensor->lin.data + global_tensor + next), lda, // const double*, const MKL_size_t
				// 		(vector->data + global_vector), incx, // const double*, const MKL_size_t
				// 		beta, // const float
				// 		(result_tensor->data + global_result + next_result), incy); // const double*, const MKL_size_t
				// 	next_result += right_size;
				// 	next += mat_size;
				// }

				// instead of this
				cblas_dgemv(
					CblasRowMajor, // const CBLAS_LAYOUT
					CblasNoTrans, // const CBLAS_TRANSPOSE
					lda2, n, // const MKL_size_t (s)
					alpha, // const double
					(tensor_ptr + global_tensor), n, // const double*, const MKL_size_t
					(vector->data + global_vector), incx, // const double*, const MKL_size_t
					beta, // const float
					(result_ptr + global_result), incy); // const double*, const MKL_size_t
			}
		}

		if (++el == blocks) {
			break;
		}

		global_result += result_size;
		global_tensor += block_size;


		if (mode>0 && el % mul_left == 0) {
				// old_result_ptr = result_ptr;
				// vector_ptr = vector->data;
					really_global_result = global_result;
					global_vector = 0;				
		} else if (el % mul_mode == 0) {
				// result_ptr = old_result_ptr;
				// vector_ptr += vector_size;		
					global_result = really_global_result;
					global_vector += vector_size;
		}

		output_index = (global_result/result_size) % world_size;

	}

	}

}


void
tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_v2(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

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
	const size_t left_size = block_size / right_size;

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
	while( 1) { 
	//for (size_t el=0;;) {

		mkl_dimatcopy(
			'R', 'T', left_size, right_size,
			alpha, tensor->lin.data + global_tensor,
			right_size, left_size);
	
		cblas_dgemv(
			CblasRowMajor, // const CBLAS_LAYOUT
			CblasNoTrans, // const CBLAS_TRANSPOSE
			n, lda, // const MKL_size_t (s)
			alpha, // const double
			tensor->lin.data + global_tensor, lda, // const double*, const MKL_size_t
			(vector->data + global_vector), incx, // const double*, const MKL_size_t
			beta, // const float
			(result_tensor->data + global_result), incy); // const double*, const MKL_size_t

		if (++el == blocks) {
			break;
		}

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

void
tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_v2_unfold(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode, DTYPE * const restrict unfold) {

	// initialize LIBXSMM
	int prefetch = LIBXSMM_PREFETCH_AUTO;
	libxsmm_dmmfunction kernel;

	const size_t dim = tensor->dim;
	size_t blocks = 1;	
	size_t block_size = 1;
	size_t right_size = 1;
	size_t mul_mode = 1;
	size_t mul_left = 1;
	for (size_t d=dim-1; d<dim; --d) {
		size_t temp = (tensor->layout[d] + tensor->block_layout[d] - 1) / tensor->block_layout[d];
		blocks *= temp;
		block_size *= tensor->block_layout[tensor->layout_perm[d]];
		if (d > mode) {
			right_size *= tensor->block_layout[tensor->layout_perm[d]];
			mul_mode *= temp;
		} else if (d == mode) {
			mul_left = mul_mode * temp;
		}
	}

	const size_t vector_size = tensor->block_layout[mode];
	const size_t result_size = block_size / vector_size;
	const size_t left_size = block_size / right_size / vector_size;

	DTYPE * tensor_ptr = tensor->lin.data;
	DTYPE * unfold_base = unfold;

    /* JIT Kernel */
    // if (mode != dim-1) {
  	kernel = libxsmm_dmmdispatch(result_size, 1, vector_size, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
  	// } else {
  	// 	kernel = libxsmm_dmmdispatch(nn, result_size, kk, NULL, NULL, NULL, NULL, NULL, NULL, &prefetch);
  	// }


	// // compute: right, left, block, result sizes
	// int right_size = 1;
	// size_t block_size = 1;
	// for (size_t d=dim-1; d<dim; --d) {
	// 	if (d > mode) {
	// 		right_size *= tensor->block_layout[tensor->layout_perm[d]];
	// 	}
	// 	block_size *= tensor->block_layout[tensor->layout_perm[d]];
	// }
	// const size_t vector_size = tensor->block_layout[mode];
	// const size_t result_size = block_size / vector_size;
	// const size_t mat_size = (size_t) right_size * (size_t) vector_size;
	// const size_t left_mat_size = block_size / mat_size;

	// size_t really_global_result = 0;
	// size_t global_tensor = 0;
	// size_t global_result = 0;
	// size_t global_vector = 0;

	size_t diff_new = result_size * mul_mode;

 //    const int nn = 1;
 //    const int kk = vector_size;
 //    const double * tensor_ptr = tensor->lin.data;
    const double * vector_ptr = vector->data;
    double * result_ptr = result_tensor->data;
 //    // double * old_result_ptr = result_tensor->data;

	size_t el = 0;
	while (1) {

		if (mode != 0) {
			for (size_t i=0; i<left_size; ++i) {
				for (size_t j=0; j<vector_size; ++j) {
					CopyWithSSEPrefetchNT(unfold_base + j*result_size, tensor_ptr, right_size);
					tensor_ptr += right_size;
				}
				unfold_base += right_size;
			}
			unfold_base = unfold;
		} else {
			CopyWithSSEPrefetchNT(unfold, tensor_ptr, block_size);
			tensor_ptr += block_size;
		}

		kernel(unfold, vector_ptr, result_ptr);//, NULL, NULL, NULL);

		if (++el == blocks) {
			break;
		}

		result_ptr += result_size;
		// tensor_ptr += block_size;

		if (mode>0 && el % mul_left == 0) {
				vector_ptr = vector->data;
		} else if (el % mul_mode == 0) {
				result_ptr = result_ptr - diff_new;
				vector_ptr += vector_size;			
		}

	}
}

void
tvm_block_major_input_aligned_output_aligned_BLAS_POWERS(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

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
	const size_t mat_size = right_size * vector_size;
	const size_t left_mat_size = block_size / mat_size;

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
	while (1) {
	//for (size_t el=0;;) {

		if (mode == 0) {
			mkl_dimatcopy(
				'R', 'T', vector_size, result_size,
				alpha, tensor->lin.data + global_tensor,
				result_size, vector_size);
		} else {
			for (size_t i=0; i<left_mat_size; ++i) {
				mkl_dimatcopy(
					'R', 'T', vector_size, right_size,
					alpha, tensor->lin.data + global_tensor + (mat_size*i),
					right_size, vector_size);
			}
		}

		cblas_dgemv(
			CblasRowMajor, // const CBLAS_LAYOUT
			CblasNoTrans, // const CBLAS_TRANSPOSE
			n, lda, // const MKL_size_t (s)
			alpha, // const double
			tensor->lin.data + global_tensor, lda, // const double*, const MKL_size_t
			(vector->data + global_vector), incx, // const double*, const MKL_size_t
			beta, // const float
			(result_tensor->data + global_result), incy); // const double*, const MKL_size_t

		if (++el == blocks) {
			break;
		}

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

void
tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_partcopy_stream(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode, DTYPE * const restrict unfold) {

	// DTYPE * dest __attribute__ ((aligned (32))) = unfold;
	// DTYPE * source __attribute__ ((aligned (32))) = tensor->lin.data;

	// const size_t dim = tensor->dim;
	// size_t block_size = 1;
	// for (size_t d=dim-1; d<dim; --d) {
		// block_size *= tensor->block_layout[tensor->layout_perm[d]];
	// }
	// size_t global_tensor = tensor->lin.data;
	// double checksum = 0;
	// if (block_size != tensor->lin.size) {
	// 	printf("block_size=%d, tensor_size=%d", block_size, tensor->lin.size);
	// 	exit(-1);
	// }
	CopyWithSSEPrefetchNT(unfold, tensor->lin.data, tensor->lin.size);
	// memcpy(unfold, tensor->lin.data, tensor->lin.size*sizeof(DTYPE));
	// checksum += unfold[0];
	
}

void
tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_partcopy(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode, DTYPE * const restrict unfold) {

	const size_t dim = tensor->dim;
	size_t blocks = 1;	
	size_t block_size = 1;
	size_t right_size = 1;

	for (size_t d=dim-1; d<dim; --d) {
		size_t temp = (tensor->layout[d] + tensor->block_layout[d] - 1) / tensor->block_layout[d];
		blocks *= temp;
		block_size *= tensor->block_layout[tensor->layout_perm[d]];
		if (d > mode) {
			right_size *= tensor->block_layout[tensor->layout_perm[d]];
		} 
	}

	const size_t vector_size = tensor->block_layout[mode];
	const size_t result_size = block_size / vector_size;
	const size_t left_size = block_size / right_size / vector_size;

	DTYPE * tensor_ptr = tensor->lin.data;
	DTYPE * unfold_base = unfold;

	size_t el = 0;
	while (1) {

		if (mode != 0) {
			for (size_t i=0; i<left_size; ++i) {
				for (size_t j=0; j<vector_size; ++j) {
					CopyWithSSEPrefetchNT(unfold_base + j*result_size, tensor_ptr, right_size);
					tensor_ptr += right_size;
				}
				unfold_base += right_size;
			}
			unfold_base = unfold;
		} else {
			CopyWithSSEPrefetchNT(unfold, tensor_ptr, block_size);
			tensor_ptr += block_size;
		}

		if (++el == blocks) {
			break;
		}

	}
}

void
tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_intocol_linout(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode, DTYPE * const restrict unfold) {

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
	const size_t mat_size = right_size * vector_size;
	const size_t left_mat_size = block_size / mat_size;

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

	// allocate space for the unfold
	//DTYPE * unfold = malloc(block_size * sizeof(DTYPE));
	size_t el = 0;

	while (1) {

		if (mode == 0) {
			mkl_domatcopy(
				'R', 'T', vector_size, result_size,
				alpha, tensor->lin.data + global_tensor,
				result_size, unfold, vector_size);
		} else {
			for (size_t i=0; i<left_mat_size; ++i) {
				mkl_domatcopy(
					'R', 'T', vector_size, right_size,
					alpha, tensor->lin.data + global_tensor + (mat_size*i),
					right_size, unfold + (mat_size*i), vector_size);
			}
		}

		cblas_dgemv(
			CblasRowMajor, // const CBLAS_LAYOUT
			CblasNoTrans, // const CBLAS_TRANSPOSE
			n, lda, // const MKL_size_t (s)
			alpha, // const double
			unfold, lda, // const double*, const MKL_size_t
			(vector->data + global_vector), incx, // const double*, const MKL_size_t
			beta, // const float
			(result_tensor->data + global_result), incy); // const double*, const MKL_size_t

		if (++el == blocks) {
			break;
		}

		// global_result += result_size;
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

void
tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_intocol_linin(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode, DTYPE * const restrict unfold) {

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
	const size_t left_size = block_size / right_size / vector_size;
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

	// size_t tensor_index = 0;
	// size_t tensor_diff = 0;
	size_t stride = 0;
	// size_t next = 0;
	if (mode != 0) {
		stride = right_size * (tensor->block_layout[tensor->layout_perm[mode]] - 1);
	} else {
		stride = 0;
	}

	DTYPE * tensor_ptr = tensor->lin.data;
	DTYPE * unfold_base = unfold;

	while (1) {

		if (mode != 0) {
			for (size_t i=0; i<left_size; ++i) {
				for (size_t j=0; j<vector_size; ++j) {
					CopyWithSSEPrefetchNT(unfold_base + j*result_size, tensor_ptr, right_size);
					tensor_ptr += right_size;
				}
				unfold_base += right_size;
			}
			unfold_base = unfold;
		} else {
			CopyWithSSEPrefetchNT(unfold, tensor_ptr, block_size);
			tensor_ptr += block_size;
		}
	
		cblas_dgemv(
			CblasColMajor, // const CBLAS_LAYOUT
			CblasNoTrans, // const CBLAS_TRANSPOSE
			n, lda, // const MKL_size_t (s)
			alpha, // const double
			unfold, n, // const double*, const MKL_size_t
			(vector->data + global_vector), incx, // const double*, const MKL_size_t
			beta, // const float
			(result_tensor->data + global_result), incy); // const double*, const MKL_size_t

		if (++el == blocks) {
			break;
		}

		global_result += result_size;
		// global_tensor += block_size;
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

		// vector_ptr = vector->data + global_vector;
		// result_ptr = result_tensor->data + global_result;
		// next = 0;
	
	}
	free(mul);
}

// JUSTCOPY
// void
// tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_intocol_linin(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode, DTYPE * const restrict unfold) {

// 	const size_t dim = tensor->dim;
// 	size_t blocks = 1;
// 	for (size_t d=dim-1; d<dim; --d) {
// 		size_t temp = (tensor->layout[d] + tensor->block_layout[d] - 1) / tensor->block_layout[d];
// 		blocks *= temp;

// 	}
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
// 	const size_t left_size = block_size / right_size / vector_size;

// 	size_t el = 0;

// 	size_t stride = 0;
// 	if (mode != 0) {
// 		stride = right_size * (tensor->block_layout[tensor->layout_perm[mode]] - 1);
// 	} else {
// 		stride = 0;
// 	}

// 	DTYPE * tensor_ptr = tensor->lin.data;
// 	DTYPE * unfold_base = unfold;

// 	while (1) {

// 		if (mode != 0) {
// 			for (size_t i=0; i<left_size; ++i) {
// 				for (size_t j=0; j<vector_size; ++j) {
// 					CopyWithSSEPrefetchNT(unfold_base + j*result_size, tensor_ptr, right_size);
// 					tensor_ptr += right_size;
// 				}
// 				unfold_base += right_size;
// 			}
// 			unfold_base = unfold;
// 		} else {
// 			CopyWithSSEPrefetchNT(unfold, tensor_ptr, block_size);
// 			tensor_ptr += block_size;
// 		}
	
// 		if (++el == blocks) {
// 			break;
// 		}

	
// 	}
// }

void
tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_intocol_linin2(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode, DTYPE * const restrict unfold) {

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
	const size_t left_size = block_size / right_size / vector_size;
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

	size_t tensor_index = 0;
	size_t tensor_diff = 0;
	size_t stride = 0;
	size_t next = 0;
	if (mode != 0) {
		stride = right_size * (tensor->block_layout[tensor->layout_perm[mode]] - 1);
	} else {
		stride = 0;
	}

	DTYPE * tensor_ptr = tensor->lin.data;
	DTYPE * unfold_base = unfold;

	while (1) {

		next = 0;
		tensor_diff = global_tensor;
		tensor_index = 0;
	    for (size_t j=0; j<vector_size; ++j) {
			tensor_index = tensor_diff;
			for (size_t i=0; i<result_size; ++i) {
				if ((i!=0) & (i % right_size == 0)) {
					tensor_index += stride;
				}
				unfold[next++] = tensor->lin.data[tensor_index + i];
			}
			tensor_diff += right_size;
		}

		cblas_dgemv(
			CblasColMajor, // const CBLAS_LAYOUT
			CblasNoTrans, // const CBLAS_TRANSPOSE
			n, lda, // const MKL_size_t (s)
			alpha, // const double
			unfold, n, // const double*, const MKL_size_t
			(vector->data + global_vector), incx, // const double*, const MKL_size_t
			beta, // const float
			(result_tensor->data + global_result), incy); // const double*, const MKL_size_t

		if (++el == blocks) {
			break;
		}

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

		// vector_ptr = vector->data + global_vector;
		// result_ptr = result_tensor->data + global_result;
		// next = 0;
	
	}
	free(mul);
}

void
tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_intocol_linin3(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode, DTYPE * const restrict unfold) {

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
	const size_t left_size = block_size / right_size / vector_size;
	const size_t helper = block_size - (right_size*vector_size);

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
	DTYPE * tensor_ptr = tensor->lin.data;
	DTYPE * unfold_base = unfold;

	size_t stride = 0;
	if (mode != 0) {
		stride = right_size * (tensor->block_layout[tensor->layout_perm[mode]]);
	}

	while (1) {

		if (mode != 0) {
		    for (size_t j=0; j<vector_size; ++j) {
				for (size_t i=0; i<left_size; ++i) {
					CopyWithSSEPrefetchNT(unfold_base, tensor_ptr + i*stride, right_size);
					unfold_base += right_size;
				}
				tensor_ptr += right_size;
			}
			unfold_base = unfold;
			tensor_ptr += helper;
		} else {
			CopyWithSSEPrefetchNT(unfold, tensor_ptr, block_size);
			tensor_ptr += block_size;
		}
		
		cblas_dgemv(
			CblasColMajor, // const CBLAS_LAYOUT
			CblasNoTrans, // const CBLAS_TRANSPOSE
			n, lda, // const MKL_size_t (s)
			alpha, // const double
			unfold, n, // const double*, const MKL_size_t
			(vector->data + global_vector), incx, // const double*, const MKL_size_t
			beta, // const float
			(result_tensor->data + global_result), incy); // const double*, const MKL_size_t

		if (++el == blocks) {
			break;
		}

		global_result += result_size;
		// global_tensor += block_size;
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

		// vector_ptr = vector->data + global_vector;
		// result_ptr = result_tensor->data + global_result;
		// next = 0;
	
	}
	free(mul);
}

void
tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_intorow2(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode, DTYPE * const restrict unfold) {

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
	const size_t left_size = block_size / right_size / vector_size;
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
	size_t tensor_index = 0;
	size_t tensor_diff = 0;
	size_t stride = 0;
	size_t next = 0;
	if (mode != 0) {
		stride = right_size * (tensor->block_layout[tensor->layout_perm[mode]] - 1);
	} else {
		stride = 0;
	}

	DTYPE * tensor_ptr = tensor->lin.data;
	DTYPE * unfold_base = unfold;

	while (1) {

		next = 0;
		tensor_index = global_tensor;
		tensor_diff = 0;
		for (size_t i=0; i<result_size; ++i) {
			if ((i>0) & (i % right_size == 0)) {
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

		if (++el == blocks) {
			break;
		}

		global_result += result_size;
		// global_tensor += block_size;
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

void
tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_intorow(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode, DTYPE * const restrict unfold) {

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
	const size_t left_size = block_size / right_size / vector_size;
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
	size_t tensor_index = 0;
	size_t tensor_diff = 0;
	size_t stride = 0;
	size_t next = 0;
	if (mode != 0) {
		stride = right_size * (tensor->block_layout[tensor->layout_perm[mode]] - 1);
	} else {
		stride = 0;
	}

	DTYPE * tensor_ptr = tensor->lin.data;
	DTYPE * unfold_base = unfold;

	while (1) {

		next = 0;
		tensor_index = global_tensor;
		tensor_diff = 0;
		for (size_t i=0; i<result_size; ++i) {
			if ((i>0) & (i % right_size == 0)) {
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

		if (++el == blocks) {
			break;
		}

		global_result += result_size;
		// global_tensor += block_size;
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

void
tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode, DTYPE * const restrict unfold) {

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
	const size_t mat_size = right_size * vector_size;
	const size_t left_mat_size = block_size / mat_size;

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

	// allocate space for the unfold
	//DTYPE * unfold = malloc(block_size * sizeof(DTYPE));

	size_t el = 0;
	while (1) {
	//for (size_t el=0;;) {

		// ONLY IN NON-BENCH VERSION
		// you do the copy no matter what
		// just add "global_tensor"
		//for (size_t i=0; i<block_size; ++i) {
			//unfold[i] = tensor->lin.data[global_tensor + i];
		//}

		if (mode == 0) {
			mkl_domatcopy(
				'R', 'T', vector_size, result_size,
				alpha, tensor->lin.data + global_tensor,
				result_size, unfold, vector_size);
		} else {
			for (size_t i=0; i<left_mat_size; ++i) {
				mkl_domatcopy(
					'R', 'T', vector_size, right_size,
					alpha, tensor->lin.data + global_tensor + (mat_size*i),
					right_size, unfold + (mat_size*i), vector_size);
			}
		}

		cblas_dgemv(
			CblasRowMajor, // const CBLAS_LAYOUT
			CblasNoTrans, // const CBLAS_TRANSPOSE
			n, lda, // const MKL_size_t (s)
			alpha, // const double
			unfold, lda, // const double*, const MKL_size_t
			(vector->data + global_vector), incx, // const double*, const MKL_size_t
			beta, // const float
			(result_tensor->data + global_result), incy); // const double*, const MKL_size_t

		if (++el == blocks) {
			break;
		}

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
	//free(unfold);
	free(mul);
}

void
tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_v3_old(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

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
		if (d > PREFIXMODE) {
			right_size *= tensor->block_layout[tensor->layout_perm[d]];
		}
		block_size *= tensor->block_layout[tensor->layout_perm[d]];
	}
	const size_t vector_size = tensor->block_layout[mode];
	const size_t result_size = block_size / vector_size;
	const size_t mat_size = right_size * tensor->block_layout[PREFIXMODE];
	size_t left_mat_size = block_size / mat_size;

	size_t really_global_result = 0;
	size_t global_tensor = 0;
	size_t global_result = 0;
	size_t global_vector = 0;

	size_t next;
	size_t next_result;

	short mini_next = 0;

	// BLAS call constants
	const double alpha = 1;
	const double beta = 1;
	const MKL_INT incx = 1;
	const MKL_INT incy = 1;
	const MKL_INT lda = right_size;
	// const MKL_INT lda2 = result_size;

	//MKL_INT n;
	//if (mode == 0) {
		//n = tensor->block_layout[PREFIXMODE];
	//} else {
	MKL_INT n = vector_size;
	//}

#if 0
	// extended vector for a block must be of the size of block_size / mode, right? so of the result_size
	size_t srcsize = tensor->block_layout[PREFIXMODE];
	size_t completesize;
	size_t remaining;
	//printf("srcsize = %d\n", srcsize);
	if (srcsize == 1) {
		completesize = srcsize;
		remaining = 0;
	} else {
		completesize = srcsize * vector_size;
		remaining = log2(vector_size);
	}
	DTYPE * extended_vector = calloc(completesize, sizeof(DTYPE));
	const size_t remainder = completesize % (size_t) pow(2, remaining);
	//printf("remainder is %d, because remaining(log2) is %d, and 2^ramining is %d \n", remainder, remaining, (int) pow(2,remaining));
	int vector_changed = 1;
#endif

	size_t el = 0;
	while (1) {
	//for (size_t el=0;;) {

#if 0
		if (vector_changed) {
			// 2^0 case
			memcpy(extended_vector, vector->data + global_vector, vector_size*sizeof(DTYPE));
			size_t power_of_2 = 1;
			//printf("remaining part is %d because vector[pref] size is %d while vec_size is %d \n", remaining, srcsize, vector_size);
			// 2^n case
			for (size_t c=0; c<remaining; ++c) {
				memcpy(extended_vector + power_of_2*vector_size, extended_vector, power_of_2*vector_size*sizeof(DTYPE));
				power_of_2<<=1;
				//printf("after loop, extended vec is \n");
				//print_to_console(extended_vector, completesize);
			}
			// remainder case
			int remainder = vector_size % power_of_2;
			memcpy(extended_vector + power_of_2*vector_size, extended_vector, remainder*sizeof(DTYPE));
			//printf("remainder is %d, so finally extended vector is \n", remainder);
			print_to_console(extended_vector, completesize);
		}

		printf("mode=%d -> n=%d, lda=%d, left_mat_size=%d, mat_size=%d, right_size=%d\n",
			mode, n, lda, left_mat_size, mat_size, right_size);

		if (mode == 2) {
			next = 0;
			next_result = 0;

			printf("n=%d, lda=%d, completesize=%d\n", n * srcsize, lda * srcsize, completesize);
			printf("completesize=%d, n*srcsize=%d, lda*srcsize=%d\n", completesize, n*srcsize, lda*srcsize);

			printf("TENSOR BLOCK=\n");
			print_to_console(tensor->lin.data + global_tensor, block_size);

			printf("(extended) VECTOR FOR THIS BLOCK=\n");
			print_to_console(extended_vector, completesize);

			for (size_t i=0; i<left_mat_size; ++i) {
		
				printf("TENSOR SUBBLOCK=\n");
				print_to_console(tensor->lin.data + global_tensor + next, block_size/left_mat_size);

				cblas_dgemv(
					CblasRowMajor, // const CBLAS_LAYOUT
					CblasTrans, // const CBLAS_TRANSPOSE
					n * srcsize, lda * srcsize,
					alpha, // const double
					(tensor->lin.data + global_tensor + next), lda * srcsize, // const double*, const MKL_size_t
					extended_vector, incx, // const double*, const MKL_size_t
					beta, // const float
					(result_tensor->data + global_result + next_result), incy); // const double*, const MKL_size_t

				printf("all result-block:\n");
				print_to_console(result_tensor->data + global_result, result_size);

				next += mat_size * srcsize;
				next_result += right_size * srcsize;

			}
#endif
		if (mode == 0) {
			next = 0;
			mini_next = 0;
			for (size_t i=0; i<left_mat_size; ++i) {
				cblas_dgemv(
					CblasRowMajor, // const CBLAS_LAYOUT
					CblasNoTrans, // const CBLAS_TRANSPOSE
					mat_size, 1,
					alpha, // const double
					(tensor->lin.data + global_tensor + next), 1, // const double*, const MKL_size_t
					(vector->data + global_vector + mini_next), incx, // const double*, const MKL_size_t
					beta, // const float
					(result_tensor->data + global_result), incy); // const double*, const MKL_size_t
				next += mat_size;
				++mini_next;
			}
		} else if (mode != dim-1) {
			next = 0;
			next_result = 0;
			for (size_t i=0; i<left_mat_size; ++i) {
				cblas_dgemv(
					CblasRowMajor, // const CBLAS_LAYOUT
					CblasTrans, // const CBLAS_TRANSPOSE
					n, lda,
					alpha, // const double
					(tensor->lin.data + global_tensor + next), lda, // const double*, const MKL_size_t
					(vector->data + global_vector), incx, // const double*, const MKL_size_t
					beta, // const float
					(result_tensor->data + global_result + next_result), incy); // const double*, const MKL_size_t
				next += mat_size;
				next_result += right_size;
			}
		} else {
			printf("THATS MY TURNS!\n");
			next = 0;
			next_result = 0;
			for (size_t i=0; i<left_mat_size; ++i) {
				cblas_dgemv(
					CblasRowMajor, // const CBLAS_LAYOUT
					CblasNoTrans, // const CBLAS_TRANSPOSE
					lda, n,
					alpha, // const double
					(tensor->lin.data + global_tensor + next), n, // const double*, const MKL_size_t
					(vector->data + global_vector), incx, // const double*, const MKL_size_t
					beta, // const float
					(result_tensor->data + global_result + next_result), incy); // const double*, const MKL_size_t
				next += mat_size;
				next_result += right_size;
			}
		}
#if 0
			cblas_dgemv(
				CblasRowMajor, // const CBLAS_LAYOUT
				CblasNoTrans, // const CBLAS_TRANSPOSE
				lda2, n, // const MKL_size_t (s)
				alpha, // const double
				tensor->lin.data + global_tensor, n, // const double*, const MKL_size_t
				(vector->data + global_vector), incx, // const double*, const MKL_size_t
				beta, // const float
				(result_tensor->data + global_result), incy); // const double*, const MKL_size_t
		}
#endif

		if (++el == blocks) {
			break;
		}

		//vector_changed = 0;
		global_result += result_size;
		global_tensor += block_size;
		for (size_t i=0; i<=mode; ++i) {
			if (el % mul[i] == 0) {
				if (i == mode) {
					global_result = really_global_result;
					global_vector += vector_size;
					//vector_changed = 1;
				} else {
					really_global_result = global_result;
					global_vector = 0;
					//vector_changed = 1;
				}
				break; // quit - no need go further in the loop
			}
		}

	}

	free(mul);

}

void
tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_mode1(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode, DTYPE * const restrict unfold) {
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
	const size_t mat_size = right_size * vector_size;
	const size_t left_mat_size = block_size / mat_size;

	size_t really_global_result = 0;
	size_t global_tensor = 0;
	size_t global_result = 0;
	size_t global_vector = 0;

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

	size_t el = 0;
	size_t size;

	while (1) {

		if (mode != dim-1) {
			next = 0;
			next_result = 0;
			for (size_t i=0; i<left_mat_size; ++i) {
				cblas_dgemv(
					CblasRowMajor, // const CBLAS_LAYOUT
					CblasTrans, // const CBLAS_TRANSPOSE
					n, lda,
					alpha, // const double
					tensor->lin.data + global_tensor + next, lda, // const double*, const MKL_size_t
					(vector->data + global_vector), incx, // const double*, const MKL_size_t
					beta, // const float
					(result_tensor->data + global_result + next_result), incy); // const double*, const MKL_size_t
				next_result += right_size;
				next += mat_size;
			}
		} else {
			// no-unfold scenario
			cblas_dgemv(
				CblasRowMajor, // const CBLAS_LAYOUT
				CblasNoTrans, // const CBLAS_TRANSPOSE
				lda2, n, // const MKL_size_t (s)
				alpha, // const double
				tensor->lin.data + global_tensor, n, // const double*, const MKL_size_t
				(vector->data + global_vector), incx, // const double*, const MKL_size_t
				beta, // const float
				(result_tensor->data + global_result), incy); // const double*, const MKL_size_t
		}

		if (++el == blocks) {
			break;
		}

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

void
tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_v3_libx(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {
	
	// initialize LIBXSMM
	int prefetch = LIBXSMM_PREFETCH_AUTO;
	libxsmm_dmmfunction kernel;

	const size_t dim = tensor->dim;
	size_t blocks = 1;
	size_t mul_mode = 1;
	size_t mul_left = 1;
	for (size_t d=dim-1; d<dim; --d) {
		size_t temp = (tensor->layout[d] + tensor->block_layout[d] - 1) / tensor->block_layout[d];
		blocks *= temp;
		if (d > mode) {
			mul_mode *= temp;
		} else if (d == mode) {
			mul_left = mul_mode * temp;
		}
	}

	// compute: right, left, block, result sizes
	int right_size = 1;
	size_t block_size = 1;
	for (size_t d=dim-1; d<dim; --d) {
		if (d > mode) {
			right_size *= tensor->block_layout[tensor->layout_perm[d]];
		}
		block_size *= tensor->block_layout[tensor->layout_perm[d]];
	}
	const size_t vector_size = tensor->block_layout[mode];
	const size_t result_size = block_size / vector_size;
	const size_t mat_size = (size_t) right_size * (size_t) vector_size;
	const size_t left_mat_size = block_size / mat_size;

	size_t really_global_result = 0;
	size_t global_tensor = 0;
	size_t global_result = 0;
	size_t global_vector = 0;

	printf("hello from block\n");

	size_t diff_new = result_size * mul_mode;

    const int nn = 1;
    const int kk = vector_size;
    const double * tensor_ptr = tensor->lin.data;
    const double * vector_ptr = vector->data;
    double * result_ptr = result_tensor->data;
    // double * old_result_ptr = result_tensor->data;
    /* JIT Kernel */
    if (mode != dim-1) {
  		kernel = libxsmm_dmmdispatch(right_size, nn, kk, NULL, NULL, NULL, NULL, NULL, NULL, &prefetch);
  	} else {
  		kernel = libxsmm_dmmdispatch(nn, result_size, kk, NULL, NULL, NULL, NULL, NULL, NULL, &prefetch);
  	}
  	
	size_t el = 0;
	while (1) {

		// if (mode == 0) {
		    // kernel(tensor_ptr, vector_ptr, result_ptr, tensor_ptr + mat_size, NULL, NULL);
		if (mode != dim-1) {
		// } else if (mode != dim-1) {
			for (size_t i=0; i<left_mat_size; ++i) {
			    const double *const tensor_next = tensor_ptr + i*mat_size;
			    double *const result_next = result_ptr + i*right_size;
			    kernel(tensor_next, vector_ptr, result_next);//, NULL, NULL, NULL);
			}

		} else {
			kernel(vector_ptr, tensor_ptr, result_ptr);//, NULL, NULL, NULL);
		}
		
		if (++el == blocks) {
			break;
		}
		result_ptr += result_size;
		tensor_ptr += block_size;
		global_result += result_size;

		if (el % mul_left == 0) {
				vector_ptr = vector->data;
		} else if (el % mul_mode == 0) {
				result_ptr = result_ptr - diff_new;
				vector_ptr += vector_size;			
		}

	}
	
}

void
tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_v3(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

	const size_t dim = tensor->dim;
	size_t blocks = 1;
	size_t mul_mode = 1;
	size_t mul_left = 1;
	for (size_t d=dim-1; d<dim; --d) {
		size_t temp = (tensor->layout[d] + tensor->block_layout[d] - 1) / tensor->block_layout[d];
		blocks *= temp;
		if (d > mode) {
			mul_mode *= temp;
		} else if (d == mode) {
			mul_left = mul_mode * temp;
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
	const size_t mat_size = (size_t) right_size * (size_t) vector_size;
	const size_t left_mat_size = block_size / mat_size;

	size_t really_global_result = 0;
	size_t global_tensor = 0;
	size_t global_result = 0;
	size_t global_vector = 0;

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

	size_t el = 0;
	while (1) {
	//for (size_t el=0;;) {

		if (mode != dim-1) {
			//printf("left_mat_size=%d, n=%d, lda=%d - lda2=%d \n", left_mat_size, n, lda, lda2);

			next = 0;
			next_result = 0;
			for (size_t i=0; i<left_mat_size; ++i) {
				cblas_dgemv(
					CblasRowMajor, // const CBLAS_LAYOUT
					CblasTrans, // const CBLAS_TRANSPOSE
					n, lda,
					alpha, // const double
					(tensor->lin.data + global_tensor + next), lda, // const double*, const MKL_size_t
					(vector->data + global_vector), incx, // const double*, const MKL_size_t
					beta, // const float
					(result_tensor->data + global_result + next_result), incy); // const double*, const MKL_size_t
				next_result += right_size;
				next += mat_size;
			}
		} else {
			//printf("left_mat_size=%d, n=%d, lda=%d - lda2=%d \n", left_mat_size, n, lda, lda2);

			// try this code
			// next = 0;
			// next_result = 0;
			// for (size_t i=0; i<left_mat_size; ++i) {
			// 	cblas_dgemv(
			// 		CblasRowMajor, // const CBLAS_LAYOUT
			// 		CblasTrans, // const CBLAS_TRANSPOSE
			// 		n, lda,
			// 		alpha, // const double
			// 		(tensor->lin.data + global_tensor + next), lda, // const double*, const MKL_size_t
			// 		(vector->data + global_vector), incx, // const double*, const MKL_size_t
			// 		beta, // const float
			// 		(result_tensor->data + global_result + next_result), incy); // const double*, const MKL_size_t
			// 	next_result += right_size;
			// 	next += mat_size;
			// }

			// instead of this
			cblas_dgemv(
				CblasRowMajor, // const CBLAS_LAYOUT
				CblasNoTrans, // const CBLAS_TRANSPOSE
				lda2, n, // const MKL_size_t (s)
				alpha, // const double
				tensor->lin.data + global_tensor, n, // const double*, const MKL_size_t
				(vector->data + global_vector), incx, // const double*, const MKL_size_t
				beta, // const float
				(result_tensor->data + global_result), incy); // const double*, const MKL_size_t

		}

		if (++el == blocks) {
			break;
		}

		global_result += result_size;
		global_tensor += block_size;

		if (mode>0 && el % mul_left == 0) {
				// old_result_ptr = result_ptr;
				// vector_ptr = vector->data;
					really_global_result = global_result;
					global_vector = 0;				
		} else if (el % mul_mode == 0) {
				// result_ptr = old_result_ptr;
				// vector_ptr += vector_size;		
					global_result = really_global_result;
					global_vector += vector_size;
		}

	}

}

void
tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_v4(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

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
	const size_t mat_size = right_size * vector_size;
	const size_t left_mat_size = block_size / mat_size;

	size_t really_global_result = 0;
	size_t global_tensor = 0;
	size_t global_result = 0;
	size_t global_vector = 0;

	size_t next;
	size_t next_result;

	// BLAS call constants
	const double alpha = 1;
	const double beta = 1;
	const MKL_INT incx = 1;
	const MKL_INT incy = 1;
	const MKL_INT lda = right_size;
	const MKL_INT n = vector_size;

	size_t el = 0;
	while (1) {
	//for (size_t el=0;;) {

		next = 0;
		next_result = 0;
		for (size_t i=0; i<left_mat_size; ++i) {
			cblas_dgemv(
				CblasColMajor, // const CBLAS_LAYOUT
				CblasNoTrans, // const CBLAS_TRANSPOSE
				lda, n,
				alpha, // const double
				(tensor->lin.data + global_tensor + next), lda, // const double*, const MKL_size_t
				(vector->data + global_vector), incx, // const double*, const MKL_size_t
				beta, // const float
				(result_tensor->data + global_result + next_result), incy); // const double*, const MKL_size_t
			next += mat_size;
			next_result += right_size;
		}

		if (++el == blocks) {
			break;
		}

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

// block BLAS bench ( working!!! )
void
tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_mine(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

	const size_t dim = tensor->dim;
	const size_t vector_size = tensor->block_layout[mode];
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
	size_t left_size = 1;
	size_t block_size = 1;
	//size_t unfold_mul = 1;
	//size_t unfold_diff = 0;
	for (size_t d=dim-1; d<dim; --d) {
		if (d > mode) {
			right_size *= tensor->block_layout[tensor->layout_perm[d]];
			//unfold_mul *= tensor->block_layout[tensor->layout_perm[d]];
		} else if (d < mode) {
			left_size *= tensor->block_layout[tensor->layout_perm[d]];
		}
		block_size *= tensor->block_layout[tensor->layout_perm[d]];
	}

	size_t result_size = block_size / vector_size;
	size_t really_global_result = 0;
	size_t global_result = 0;
	size_t global_vector = 0;
	size_t next = 0;

	size_t stride;
	if (mode != 0) {
		stride = right_size * (tensor->block_layout[tensor->layout_perm[mode]] - 1);
	} else {
		stride = 0;
	}

	size_t global_tensor = 0;
	// size_t tensor_index, tensor_diff;
	// BLAS call constants
	const double alpha = 1;
	const double beta = 1;
	const MKL_INT incx = 1;
	const MKL_INT incy = 1;
	const MKL_INT lda = vector_size;
	//MKL_INT n;
	//unfold, n
	size_t UNFOLD_REQ;
	DTYPE * unfold;
	if (mode != dim-1) {
		UNFOLD_REQ = 1;
		unfold = malloc(block_size * sizeof(DTYPE));
		//n = tensor->block_layout[dim-1]; // layout_perm not needed (?) we are referring to original
	} else {
		UNFOLD_REQ = 0;
		//n = tensor->block_layout[dim-2]; // layout_perm not needed (?) we are referring to original
	}
	
	// int global_next = 0;
	// int k_index = 0; 	
	// int unfold_index = 0;

	for ( size_t el=0; ; ) {

		if (UNFOLD_REQ) {

			int out_out = 0;
			int out = 0;
			for (size_t i=0; i<left_size; ++i) {
				out_out = i * (right_size * vector_size);
				for (size_t v=0; v<vector_size; ++v) {
					out = out_out + v;
					for (size_t j=0; j<right_size; ++j) {
						unfold[out] = tensor->lin.data[global_tensor++];
						out += vector_size;
					}
				}
			}

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

		next += block_size;
		global_result += result_size;
		// increment the block element
		if (++el == blocks) {
			break;
		}
		for (size_t i=0; i<=mode; ++i) {
			if (el % mul[i] == 0) {
				if (i == mode) {
					global_result = really_global_result;
					global_vector += vector_size;
				} else {
					really_global_result = global_result;
					global_vector = 0;
				}
				// quit - no need go further in the loop
				break;
			}
		}

	}
	if (UNFOLD_REQ) {
		free(unfold);
	}
	free(mul);
}

#if 0
void
tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_mine(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

	const size_t dim = tensor->dim;
	const size_t vector_size = tensor->block_layout[mode];
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
	size_t left_size = 1;
	size_t block_size = 1;
	//size_t unfold_mul = 1;
	//size_t unfold_diff = 0;
	for (size_t d=dim-1; d<dim; --d) {
		if (d > mode) {
			right_size *= tensor->block_layout[tensor->layout_perm[d]];
			//unfold_mul *= tensor->block_layout[tensor->layout_perm[d]];
		} else if (d < mode) {
			left_size *= tensor->block_layout[tensor->layout_perm[d]];
		}
		block_size *= tensor->block_layout[tensor->layout_perm[d]];
	}

	size_t result_size = block_size / vector_size;
	size_t really_global_result = 0;
	size_t global_result = 0;
	size_t global_vector = 0;
	size_t next = 0;

	size_t stride;
	if (mode != 0) {
		stride = right_size * (tensor->block_layout[tensor->layout_perm[mode]] - 1);
	} else {
		stride = 0;
	}

	size_t global_tensor = 0;
	size_t tensor_index, tensor_diff;
	// BLAS call constants
	const double alpha = 1;
	const double beta = 1;
	const MKL_INT incx = 1;
	const MKL_INT incy = 1;
	const MKL_INT lda = vector_size;
	//MKL_INT n;
	//unfold, n
	size_t UNFOLD_REQ;
	DTYPE * unfold;
	if (mode != dim-1) {
		UNFOLD_REQ = 1;
		unfold = malloc(block_size * sizeof(DTYPE));
		//n = tensor->block_layout[dim-1]; // layout_perm not needed (?) we are referring to original
	} else {
		UNFOLD_REQ = 0;
		//n = tensor->block_layout[dim-2]; // layout_perm not needed (?) we are referring to original
	}

	for ( size_t el=0; ; ) {

		if (UNFOLD_REQ) {
			next = 0;
			tensor_index = global_tensor;
			tensor_diff = 0;
			for (size_t i=0; i<result_size; ++i) {
				if ((i>0) & (i % right_size == 0)) {
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

		next += block_size;
		global_result += result_size;
		// increment the block element
		if (++el == blocks) {
			break;
		}
		for (size_t i=0; i<=mode; ++i) {
			if (el % mul[i] == 0) {
				if (i == mode) {
					global_result = really_global_result;
					global_vector += vector_size;
				} else {
					really_global_result = global_result;
					global_vector = 0;
				}
				// quit - no need go further in the loop
				break;
			}
		}

	}
	if (UNFOLD_REQ) {
		free(unfold);
	}
	free(mul);
}
#endif

