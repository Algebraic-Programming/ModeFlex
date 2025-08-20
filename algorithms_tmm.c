#include <algorithms.h>
#include <rand_utils.h>
#include <file_utils.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <mkl.h>
// #include <blis.h>
#include <gen_utils.h>
#include <smmintrin.h>
#include <immintrin.h>

// so far:
// looped, blocked
// mkl, libx
// total: 4 algorithms
// possible variations: orientation of the matrix (!)
// possible variations: using tmv routines (!)
// adding morton to the group(!)

void
tmm_looped_mkl(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict matrix, struct lin_storage * result, const size_t mode, const int l_dimension, const int l_block_dimension) {

	size_t dim = tensor->dim;
	size_t right_size = 1;
	for (size_t d=dim-1; d<dim; --d) {
		if (d > mode) {
			right_size *= tensor->layout[tensor->layout_perm[d]];
		}
	}

if (TEST_ENV==1) {
			// printf("right_size=%zu\n", right_size);
 			// round_numbers(tensor->lin.data, tensor->lin.size);
			// round_numbers(matrix->data, matrix->size);
			// printf("Tensor (size %zu): ", tensor->lin.size);
			// print_to_console(tensor->lin.data, tensor->lin.size);
			// printf("Matrix (size %zu): ", matrix->size);
			// print_to_console(matrix->data, matrix->size);
}

	const MKL_INT incx = 1; 
	const MKL_INT incy = 1; 
	double alpha = 1;
	double beta = 1;

	const double *const tensor_ptr = tensor->lin.data;
	const double *const matrix_ptr = matrix->data; 
	double *const result_ptr = result->data;

	const MKL_INT mode_size = tensor->layout[mode];
	const MKL_INT n = right_size;

    size_t mat_size = (size_t) mode_size * (size_t) n; 
    size_t left_size = tensor->lin.size / mat_size;

	if (mode != dim-1) {

if (TEST_ENV==1) {
		printf("RIGHT MODE\n");
		printf("We are doing %d, %d, %d removal\n", l_dimension, n, mode_size);
}

		// k,l,m are all 
		// Mathematically, 4x2, 2x3 (because we want to remove from left side)
		MKL_INT k = l_dimension; // 4
		MKL_INT l = n; // 3
		MKL_INT m = mode_size; // 2

		for (size_t i=0; i<left_size; ++i) {

			const double * next = tensor_ptr + i*mat_size;
			double * next_result = result_ptr + i*k*l;

			cblas_dgemm(
			CblasRowMajor, // const CBLAS_LAYOUT
			// diff 1
			CblasNoTrans, CblasNoTrans,
			// diff 2
			k, l, m,
			alpha, // const double
			matrix_ptr, m, // const double*, const MKL_size_t
			next, l, // const double*, const MKL_size_t
			beta, // const float
			next_result, l); // const double*, const MKL_size_t

if (TEST_ENV==1) {
			printf("Iteration %zu\n", i);
			// print_to_console(result_ptr, result->size);
}

		}

	} else {

if (TEST_ENV==1) {
		printf("RIGHT MODE\n");
		printf("We are doing %zu, %d, %d removal\n", left_size, l_dimension, mode_size);
}

		// MATHEMATICALLY IT IS
		// 2x3 and 3x4 matrices
		MKL_INT k = left_size;//2;
		MKL_INT l = l_dimension;//4;
		MKL_INT m = mode_size;//3;

		const double * next = tensor_ptr;
		double * next_result = result_ptr;

		cblas_dgemm(
		CblasRowMajor, // const CBLAS_LAYOUT
		CblasNoTrans, CblasNoTrans,
		k, l, m,
		alpha, // const double
		next, m, // const double*, const MKL_size_t
		matrix_ptr, l, // const double*, const MKL_size_t
		beta, // const float
		next_result, l); // const double*, const MKL_size_t

	}

}

void
tmm_looped_libx(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict matrix, struct lin_storage * result, const size_t mode, const int l_dimension, const int l_block_dimension) {

	size_t dim = tensor->dim;
	size_t right_size = 1;
	for (size_t d=dim-1; d<dim; --d) {
		if (d > mode) {
			right_size *= tensor->layout[tensor->layout_perm[d]];
		}
	}

if (TEST_ENV==1) {
			printf("right_size=%zu\n", right_size);
 			// round_numbers(tensor->lin.data, tensor->lin.size);
			// round_numbers(matrix->data, matrix->size);
			printf("Tensor (size %zu): ", tensor->lin.size);
			// print_to_console(tensor->lin.data, tensor->lin.size);
			printf("Matrix (size %zu): ", matrix->size);
			// print_to_console(matrix->data, matrix->size);
}

	const MKL_INT incx = 1; 
	const MKL_INT incy = 1; 
	double alpha = 1;
	double beta = 1;

	const double *const tensor_ptr = tensor->lin.data;
	const double *const matrix_ptr = matrix->data; 
	double *const result_ptr = result->data;

	const MKL_INT mode_size = tensor->layout[mode];
	const MKL_INT n = right_size;

    size_t mat_size = (size_t) mode_size * (size_t) n; 
    size_t left_size = tensor->lin.size / mat_size;

	if (mode != dim-1) {

if (TEST_ENV==1) {
		printf("RIGHT MODE\n");
		printf("We are doing %d, %d, %d removal\n", l_dimension, n, mode_size);
}

		// k,l,m are all 
		// Mathematically, 4x2, 2x3 (because we want to remove from left side)
		MKL_INT k = l_dimension; // 4
		MKL_INT l = n; // 3
		MKL_INT m = mode_size; // 2

		for (size_t i=0; i<left_size; ++i) {

			const double * next = tensor_ptr + i*mat_size;
			double * next_result = result_ptr + i*n*l_dimension;

			cblas_dgemm(
			CblasRowMajor, // const CBLAS_LAYOUT
			// diff 1
			CblasNoTrans, CblasNoTrans,
			// diff 2
			k, l, m,
			alpha, // const double
			matrix_ptr, m, // const double*, const MKL_size_t
			next, l, // const double*, const MKL_size_t
			beta, // const float
			next_result, l); // const double*, const MKL_size_t

if (TEST_ENV==1) {
			printf("Iteration %zu\n", i);
			// print_to_console(result_ptr, result->size);
}

		}

	} else {

if (TEST_ENV==1) {
		printf("RIGHT MODE\n");
		printf("We are doing %zu, %d, %d removal\n", left_size, l_dimension, mode_size);
}

		// MATHEMATICALLY IT IS
		// 2x3 and 3x4 matrices
		MKL_INT k = left_size;//2;
		MKL_INT l = l_dimension;//4;
		MKL_INT m = mode_size;//3;

		const double * next = tensor_ptr;
		double * next_result = result_ptr;

		cblas_dgemm(
		CblasRowMajor, // const CBLAS_LAYOUT
		CblasNoTrans, CblasNoTrans,
		k, l, m,
		alpha, // const double
		next, m, // const double*, const MKL_size_t
		matrix_ptr, l, // const double*, const MKL_size_t
		beta, // const float
		next_result, l); // const double*, const MKL_size_t

	}

}

// Input: L x n_k times n_K x rest
// OR
// rest x n_k times n_k x L
// the output will be:
// (n_k, 0) (n_k x L)
// (0, n_k) (L x n_k)
// so L is not going to be a blocked dimension
void
tmm_blocked_mkl(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict matrix, struct lin_storage * result, const size_t mode, const int l_dimension, const int l_block_dimension) {

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

if (TEST_ENV==1) {
 			// round_numbers(tensor->lin.data, tensor->lin.size);
			// round_numbers(matrix->data, matrix->size);
			printf("Tensor (size %zu): ", tensor->lin.size);
			// print_to_console(tensor->lin.data, tensor->lin.size);
			printf("Matrix (size %zu): ", matrix->size);
			// print_to_console(matrix->data, matrix->size);
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

  libxsmm_dmmfunction kernel; 
  MKL_INT k,l,m;

	if (mode!= dim-1) {
			// k,l,m are all 
			// Mathematically, 4x2, 2x3 (because we want to remove from left side)
			k = l_dimension; // 4
			l = lda; // 3
			m = n; // 2
  		kernel = libxsmm_dmmdispatch(l, k, m, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
			// sizes available: n, lda
	} else {
			// available: lda2, n
			// MATHEMATICALLY IT IS
			// 2x3 and 3x4 matrices
			k = lda2;//2;
			l = l_dimension;//4;
			m = n;//3;
		kernel = libxsmm_dmmdispatch(l, k, m, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	}

	size_t el = 0;
	while (1) {

		if (mode != dim-1) {

			next = 0;
			next_result = 0;


if (TEST_ENV==1) {
		printf("LEFT MODE\n");
		printf("We are doing (a loop of) %d, %d, %d removal\n", k, l, m);
}
			
			for (size_t i=0; i<left_mat_size; ++i) {

				const double * next = (tensor->lin.data + global_tensor) + i*mat_size;
				double * next_result = (result->data + global_result) + i*right_size*l_dimension;
				// kernel(next, matrix->data + global_vector, next_result, NULL, NULL, NULL);


				cblas_dgemm(
				CblasRowMajor, // const CBLAS_LAYOUT
				// diff 1
				CblasNoTrans, CblasNoTrans,
				// diff 2
				k, l, m,
				alpha, // const double
				(matrix->data + global_vector), m, // const double*, const MKL_size_t
				next, l, // const double*, const MKL_size_t
				beta, // const float
				next_result, l); // const double*, const MKL_size_t

			}

		} else {


			
if (TEST_ENV==1) {
		printf("RIGHT MODE\n");
		printf("We are doing %d, %d, %d removal\n", k, l, m);
}
		 	// kernel(matrix->data + global_vector, tensor->lin.data + global_tensor, result->data + global_result, NULL, NULL, NULL);

			// const double * next = tensor_ptr;
			// double * next_result = result_ptr;

			cblas_dgemm(
			CblasRowMajor, // const CBLAS_LAYOUT
			CblasNoTrans, CblasNoTrans,
			k, l, m,
			alpha, // const double
			tensor->lin.data + global_tensor, m, // const double*, const MKL_size_t
			(matrix->data + global_vector), l, // const double*, const MKL_size_t
			beta, // const float
			(result->data + global_result), l); // const double*, const MKL_size_t


		}
if (TEST_ENV==1) {
			printf("After finishing that blocked we have:\n");
			// print_to_console(result->data, result->size);
}

		if (++el == blocks) {
			break;
		}

		global_result += (result_size*l_dimension);
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
					global_vector += (vector_size*l_dimension);
		}

if (TEST_ENV==1) {
		// SUPER USEFUL DEBUGGING
		printf("Move to global_vector=%zu, global_result=%zu, global_tensor=%zu\n", global_vector, global_result, global_tensor);
}

	}

}

void
tmm_blocked_trans(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict matrix, struct lin_storage * result, const size_t mode, const int l_dimension, const int l_block_dimension) {

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

	// examine potential block size for the L dimension

	// for now, get the block size to be the same as the l_dimension in itself
	size_t l_blocks = l_dimension / l_block_dimension;
	size_t mul_blocks = mul_left * l_blocks;

if (TEST_ENV==1) {
 			// round_numbers(tensor->lin.data, tensor->lin.size);
			// round_numbers(matrix->data, matrix->size);
			printf("Tensor (size %zu): ", tensor->lin.size);
			// print_to_console(tensor->lin.data, tensor->lin.size);
			printf("Matrix (size %zu): ", matrix->size);
			// print_to_console(matrix->data, matrix->size);
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
	const size_t mat_size = (size_t) right_size * (size_t) vector_size;
	const size_t left_mat_size = block_size / mat_size;

	size_t really_global_result = 0;
	size_t really_global_tensor = 0;
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

  libxsmm_dmmfunction kernel; 
  MKL_INT k,l,m;

	if (mode!= dim-1) {
			// k,l,m are all 
			// Mathematically, 4x2, 2x3 (because we want to remove from left side)
			k = l_dimension; // 4
			l = lda; // 3
			m = n; // 2
  		// kernel = libxsmm_dmmdispatch(l, k, m, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
			// sizes available: n, lda
	} else {
			// available: lda2, n
			// MATHEMATICALLY IT IS
			// 2x3 and 3x4 matrices
			k = lda2;//2;
			l = l_dimension;//4;
			m = n;//3;
		// kernel = libxsmm_dmmdispatch(l, k, m, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	}

	size_t el = 0;
	while (1) {

		if (mode != dim-1) {

			next = 0;
			next_result = 0;


if (TEST_ENV==1) {
		printf("LEFT MODE\n");
		printf("We are doing (a loop of) %d, %d, %d removal\n", k, l, m);
}
			for (size_t i=0; i<left_mat_size; ++i) {
				const double * next = (tensor->lin.data + global_tensor) + i*mat_size;
				double * next_result = (result->data + global_result) + i*right_size*(l_block_dimension);

				// kernel(next, matrix->data + global_vector, next_result, NULL, NULL, NULL);

				cblas_dgemm(
				CblasRowMajor, // const CBLAS_LAYOUT
				// diff 1
				CblasTrans, CblasNoTrans,
				// diff 2
				k, l, m,
				alpha, // const double
				(matrix->data + global_vector), l, // const double*, const MKL_size_t
				next, l, // const double*, const MKL_size_t
				beta, // const float
				next_result, l); // const double*, const MKL_size_t
				// cblas_dgemm(
				// CblasRowMajor, // const CBLAS_LAYOUT
				// // diff 1
				// CblasNoTrans, CblasNoTrans,
				// // diff 2
				// k, l, m,
				// alpha, // const double
				// (matrix->data + global_vector), m, // const double*, const MKL_size_t
				// next, l, // const double*, const MKL_size_t
				// beta, // const float
				// next_result, l); // const double*, const MKL_size_t

			}
		} else {

			
if (TEST_ENV==1) {
		printf("RIGHT MODE\n");
		printf("We are doing %d, %d, %d removal\n", k, l, m);
}
		 	// kernel(matrix->data + global_vector, tensor->lin.data + global_tensor, result->data + global_result, NULL, NULL, NULL);

			// const double * next = tensor_ptr;
			// double * next_result = result_ptr;

			cblas_dgemm(
			CblasRowMajor, // const CBLAS_LAYOUT
			CblasNoTrans, CblasTrans,
			k, l, m,
			alpha, // const double
			tensor->lin.data + global_tensor, m, // const double*, const MKL_size_t
			(matrix->data + global_vector), m, // const double*, const MKL_size_t
			beta, // const float
			(result->data + global_result), l); // const double*, const MKL_size_t

			// cblas_dgemm(
			// CblasRowMajor, // const CBLAS_LAYOUT
			// CblasNoTrans, CblasNoTrans,
			// k, l, m,
			// alpha, // const double
			// tensor->lin.data + global_tensor, m, // const double*, const MKL_size_t
			// (matrix->data + global_vector), l, // const double*, const MKL_size_t
			// beta, // const float
			// (result->data + global_result), l); // const double*, const MKL_size_t


		}
if (TEST_ENV==1) {
			printf("After finishing that blocked we have:\n");
			// print_to_console(result->data, result->size);
}

		if (++el == blocks * l_blocks) {
			break;
		}

		global_result += (result_size*l_block_dimension);
		global_tensor += block_size;

		if (el % mul_blocks == 0) {
			// printf("we have gone through mul_blocks\n");
			global_vector = 0;
			// this is necessary as we have finished with the current part of the "matrix" as understood by the MM product
			// we have completed "that" submatrix
			really_global_tensor = global_tensor;
			really_global_result = global_result;
		} else if (el % mul_left == 0) {
			// CAREFUL: here we cannot have mode>0 ANYMORE(!)
			// printf("reset tensor, move vector on L dimension\n");
			global_tensor = really_global_tensor;
			global_vector += (vector_size*l_block_dimension);
			// old_result_ptr = result_ptr;
			// vector_ptr = vector->data;
			really_global_result = global_result;
			// global_vector = 0;				
		} else if (el % mul_mode == 0) {
			// printf("Move vector on k dimension \n");
			// result_ptr = old_result_ptr;
			// vector_ptr += vector_size;		
			global_result = really_global_result;
			global_vector += (vector_size*l_block_dimension);
		}
if (TEST_ENV==1) {
		// SUPER USEFUL DEBUGGING
		printf("Move to global_vector=%zu, global_result=%zu, global_tensor=%zu\n", global_vector, global_result, global_tensor);
}
	}

}

// Blocking looks like the following:
// the matrix is bloced only on n_K, not on L
// the problem is row major vs oclumn major:
// row-major: L x n_k times n_k x rest -> we traverse the matrix in the following way: (L x n_k) (L x n_k)
// so inner most agrees with outermost
// column-major: rest x n_k times n_k x l -> we travers ehte matrix in the following way: (L x n_k) (n_k x L)
// now the innermost change is different than the block one 
// SOLUTION 1: use col trans
// SOLUTION 2: find a way to block differently than innermost dimension to actually get correctness
void
tmm_blocked_libx(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict matrix, struct lin_storage * result, const size_t mode, const int l_dimension, const int l_block_dimension) {

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

	// examine potential block size for the L dimension

	// for now, get the block size to be the same as the l_dimension in itself
	size_t l_blocks = l_dimension / l_block_dimension;
	size_t mul_blocks = mul_left * l_blocks;

if (TEST_ENV==1) {
 			// round_numbers(tensor->lin.data, tensor->lin.size);
			// round_numbers(matrix->data, matrix->size);
			printf("Tensor (size %zu): ", tensor->lin.size);
			// print_to_console(tensor->lin.data, tensor->lin.size);
			printf("Matrix (size %zu): ", matrix->size);
			// print_to_console(matrix->data, matrix->size);
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
	const size_t mat_size = (size_t) right_size * (size_t) vector_size;
	const size_t left_mat_size = block_size / mat_size;

	size_t really_global_result = 0;
	size_t really_global_tensor = 0;
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

  libxsmm_dmmfunction kernel; 
  MKL_INT k,l,m;

	if (mode!= dim-1) {
			// k,l,m are all 
			// Mathematically, 4x2, 2x3 (because we want to remove from left side)
			k = l_dimension; // 4
			l = lda; // 3
			m = n; // 2
  		kernel = libxsmm_dmmdispatch(l, k, m, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
			// sizes available: n, lda
	} else {
			// available: lda2, n
			// MATHEMATICALLY IT IS
			// 2x3 and 3x4 matrices
			k = lda2;//2;
			l = l_dimension;//4;
			m = n;//3;
		kernel = libxsmm_dmmdispatch(l, k, m, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	}

	size_t el = 0;
	while (1) {

		if (mode != dim-1) {

			next = 0;
			next_result = 0;


if (TEST_ENV==1) {
		printf("LEFT MODE\n");
		printf("We are doing (a loop of) %d, %d, %d removal\n", k, l, m);
}
			for (size_t i=0; i<left_mat_size; ++i) {
				const double * next = (tensor->lin.data + global_tensor) + i*mat_size;
				double * next_result = (result->data + global_result) + i*right_size*(l_block_dimension);

				// kernel(next, matrix->data + global_vector, next_result, NULL, NULL, NULL);

				// cblas_dgemm(
				// CblasRowMajor, // const CBLAS_LAYOUT
				// // diff 1
				// CblasTrans, CblasNoTrans,
				// // diff 2
				// k, l, m,
				// alpha, // const double
				// (matrix->data + global_vector), l, // const double*, const MKL_size_t
				// next, l, // const double*, const MKL_size_t
				// beta, // const float
				// next_result, l); // const double*, const MKL_size_t
				cblas_dgemm(
				CblasRowMajor, // const CBLAS_LAYOUT
				// diff 1
				CblasNoTrans, CblasNoTrans,
				// diff 2
				k, l, m,
				alpha, // const double
				(matrix->data + global_vector), m, // const double*, const MKL_size_t
				next, l, // const double*, const MKL_size_t
				beta, // const float
				next_result, l); // const double*, const MKL_size_t

			}
		} else {

			
if (TEST_ENV==1) {
		printf("RIGHT MODE\n");
		printf("We are doing %d, %d, %d removal\n", k, l, m);
}
		 	// kernel(matrix->data + global_vector, tensor->lin.data + global_tensor, result->data + global_result, NULL, NULL, NULL);

			// const double * next = tensor_ptr;
			// double * next_result = result_ptr;
// 
			// cblas_dgemm(
			// CblasRowMajor, // const CBLAS_LAYOUT
			// CblasNoTrans, CblasTrans,
			// k, l, m,
			// alpha, // const double
			// tensor->lin.data + global_tensor, m, // const double*, const MKL_size_t
			// (matrix->data + global_vector), m, // const double*, const MKL_size_t
			// beta, // const float
			// (result->data + global_result), l); // const double*, const MKL_size_t

			cblas_dgemm(
			CblasRowMajor, // const CBLAS_LAYOUT
			CblasNoTrans, CblasNoTrans,
			k, l, m,
			alpha, // const double
			tensor->lin.data + global_tensor, m, // const double*, const MKL_size_t
			(matrix->data + global_vector), l, // const double*, const MKL_size_t
			beta, // const float
			(result->data + global_result), l); // const double*, const MKL_size_t


		}
if (TEST_ENV==1) {
			printf("After finishing that blocked we have:\n");
			// print_to_console(result->data, result->size);
}

		if (++el == blocks * l_blocks) {
			break;
		}

		global_result += (result_size*l_block_dimension);
		global_tensor += block_size;

		if (el % mul_blocks == 0) {
			// printf("we have gone through mul_blocks\n");
			global_vector = 0;
			// this is necessary as we have finished with the current part of the "matrix" as understood by the MM product
			// we have completed "that" submatrix
			really_global_tensor = global_tensor;
			really_global_result = global_result;
		} else if (el % mul_left == 0) {
			// CAREFUL: here we cannot have mode>0 ANYMORE(!)
			// printf("reset tensor, move vector on L dimension\n");
			global_tensor = really_global_tensor;
			global_vector += (vector_size*l_block_dimension);
			// old_result_ptr = result_ptr;
			// vector_ptr = vector->data;
			really_global_result = global_result;
			// global_vector = 0;				
		} else if (el % mul_mode == 0) {
			// printf("Move vector on k dimension \n");
			// result_ptr = old_result_ptr;
			// vector_ptr += vector_size;		
			global_result = really_global_result;
			global_vector += (vector_size*l_block_dimension);
		}
if (TEST_ENV==1) {
		// SUPER USEFUL DEBUGGING
		printf("Move to global_vector=%zu, global_result=%zu, global_tensor=%zu\n", global_vector, global_result, global_tensor);
}
	}

}

// for now malfunctioninig
void
tmm_mortonblocked_libx(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict matrix, struct lin_storage * result, const size_t mode, const int l_dimension, const int l_block_dimension) {

	// // initialize LIBXSMM
	// libxsmm_dmmfunction kernel;
	// const size_t dim = tensor->dim;

	// // Morton stuff (1)
	// size_t * const block_counter = calloc(dim, sizeof(size_t));
	// size_t * const block_counter_threshold = calloc(dim, sizeof(size_t));

	// size_t mul_mode = 1;
	// size_t mul_left = 1;
	// size_t right_size = 1;
	// size_t block_size = 1;
	// size_t blocks = 1;
	// size_t max_block = 0;

	// size_t * const mul = malloc(dim * sizeof(size_t));
	// mul[dim-1] = 1;
	// for (size_t i=dim-1; i<dim; --i) {
	// 	/// BASICS
	// 	size_t temp = tensor->block_layout[tensor->layout_perm[i]];
	// 	if (i > mode) {
	// 		right_size *= temp;
	// 	}
	// 	block_size *= temp;
	// 	/// +
	// 	block_counter_threshold[i] = (tensor->layout[i] + temp -1) / temp;
	// 	blocks *= block_counter_threshold[i];
	// 	if (block_counter_threshold[i] > max_block) {
	// 		max_block = block_counter_threshold[i];
	// 	}
	// 	if (i > mode) {
	// 		mul_mode *= block_counter_threshold[i];
	// 	} else if (i == mode) {
	// 		mul_left = mul_mode * block_counter_threshold[i];
	// 	}
	// 	if (i!=0) {
	// 		mul[i-1] = mul[i] * block_counter_threshold[i];
	// 	}
	// }

	// const size_t vector_size = tensor->block_layout[mode];
	// const size_t result_size = block_size / vector_size;
	// const size_t mat_size = right_size * vector_size;
	// const size_t left_mat_size = block_size / mat_size;

	// // Morton stuff (2)
	// const size_t morton_block_levels = log2(max_block)+1; // round-up (take front size_teger and add 1)
	// size_t * const morton_block_indices = calloc(morton_block_levels, sizeof(size_t));
	// size_t global_tensor = 0;
	// size_t global_result = 0;
	// size_t global_vector = 0;
	// size_t old_global_vector = 0;

	// size_t next;
	// size_t next_result;

	// // BLAS call constants
	// const double alpha = 1;
	// const double beta = 1;
	// const MKL_INT incx = 1;
	// const MKL_INT incy = 1;
	// const MKL_INT lda = right_size;
	// const MKL_INT lda2 = result_size;
	// const MKL_INT n = vector_size;

	// // int output_block = 0;

	// // MORTON-CURVE ONLY (3)
	// size_t mask;
	// size_t level;
	// size_t inc_game;
	// size_t offset;

	// int block_diff;
	// double block_diff_log;

 //    const int nn = 1;
 //    const int kk = vector_size;
 //    const double * tensor_ptr = tensor->lin.data;
 //    const double * vector_ptr = vector->data;
 //    double * result_ptr = result_tensor->data;
 //    double * base_result_ptr = result_tensor->data;
 //    const double * base_vector_ptr = vector->data;
 //    /* JIT Kernel */
 //    if (mode != dim-1) {
 //  		kernel = libxsmm_dmmdispatch(right_size, nn, kk, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
 //  	} else {
 //  		kernel = libxsmm_dmmdispatch(nn, result_size, kk, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
 //  	}

	// size_t el = 0;
	// while (1) {

	// 	// printf("current_block=%d, Output_block=%d\n", el, output_block);
	// 	if (mode != dim-1) {
			
	// 		next = 0;
	// 		next_result = 0;
	// 		for (size_t i=0; i<left_mat_size; ++i) {
	// 		    const double *const tensor_next = tensor_ptr + i*mat_size;
	// 		    double *const result_next = result_ptr + i*right_size;
	// 		    kernel(tensor_next, vector_ptr, result_next, NULL, NULL, NULL);
	// 			/** Batched matrix multiplications (explicit data representation). */
	// 			// libxsmm_mmbatch(kernel, 0, 0, 
	// 			//   0, 0, 0,
	// 			//   tensor_next, vector_ptr, result_next,
	// 			//   // const void* a, const void* b, void* c,
	// 			//   1,
	// 			//   0, 1);
	// 		}
	// 	} else {
	// 		kernel(vector_ptr, tensor_ptr, result_ptr, NULL, NULL, NULL);
	// 	}

	// 	if (++el == blocks) {
	// 		break;
	// 	}

	// 	old_global_vector = block_counter[mode];
	// 	global_result += result_size;
	// 	tensor_ptr += block_size;

	// 	int inc_count = -1;
	// 	mask = 1;
	// 	level = 0;
	// 	inc_game = 1;
	// 	offset = dim-1;
	// 	while (inc_game) {
	// 		inc_count++;
	// 		if (block_counter[offset] & mask) {
	// 			block_counter[offset] &= ~mask;
	// 			if (offset == 0) {
	// 				mask <<= 1;
	// 				level += 1;
	// 				offset = dim-1;
	// 			} else {
	// 				offset -= 1;
	// 			}
	// 		} else {
	// 			if ((block_counter[offset] | mask) >= block_counter_threshold[offset]) {
	// 				if (offset == 0) {
	// 					mask <<= 1;
	// 					level += 1;
	// 					offset = dim-1;
	// 				} else {
	// 					offset -= 1;
	// 				}
	// 			} else {
	// 				inc_game = 0;
	// 			}
	// 		}
	// 	}
	// 	block_counter[offset] |= mask;
	// 	// print_to_console_sizet(block_counter, dim);
	// 	// printf("level=%zu\n", level);
	// 	// printf("inc took place=%d\n", inc_count);
	// 	// RESULT HAS TO CHANGE(!)
	// 	// Perhaps we can use this as an indication of the change...
	// 	if (offset == mode) {
	// 		size_t temp = global_result;
	// 		global_result = morton_block_indices[level];
	// 		morton_block_indices[level] = temp;
	// 		block_diff = block_counter_threshold[mode] - block_counter[mode];
	// 		if (block_diff != 0) {
	// 			 block_diff_log = log2(block_diff);
	// 			if (block_diff_log == (int) block_diff_log) {
	// 				block_diff = block_diff_log;
	// 			} else {
	// 				block_diff = block_diff_log+1;
	// 			}
	// 		} else {
	// 			block_diff = 0;
	// 		}
	// 		if (block_diff < level) {
	// 			if (block_diff > 0) {
	// 			for (size_t i=0; i<=block_diff-1; ++i) {
	// 				morton_block_indices[i] = global_result;
	// 			}
	// 			}
	// 		} else {
	// 			if (level > 0) {
	// 			for (size_t i=0; i<=level-1; ++i) {
	// 				morton_block_indices[i] = global_result;
	// 			}
	// 			}
	// 		}
	// 	}
	// 	result_ptr = base_result_ptr + global_result;
	// 	// output_block = global_result/result_size;
	// 	// VECTOR HAS TO CHANGE???
	// 	global_vector = block_counter[mode] * tensor->block_layout[mode];
	// 	vector_ptr = base_vector_ptr + global_vector;
	// }

	// free(morton_block_indices);
	// free(block_counter);
	// free(block_counter_threshold);
	// free(mul);
}

// are we right in not using l_block_dimension?
void
tmm_mortonblocked_trans(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict matrix, struct lin_storage * result, const size_t mode, const int l_dimension, const int l_block_dimension) {

	const size_t dim = tensor->dim;
	// Morton stuff (1)
	size_t * const block_counter = calloc(dim, sizeof(size_t));
	size_t * const block_counter_threshold = calloc(dim, sizeof(size_t));
	// compute: right, block, result sizes
	// + blocks, block_counter_thresholds, max_block
	size_t right_size = 1;
	size_t block_size = 1;
	size_t blocks = 1;
	size_t max_block = 0;
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
	}
	const size_t vector_size = tensor->block_layout[mode];
	const size_t result_size = block_size / vector_size;
	const size_t mat_size = right_size * vector_size;
	const size_t left_mat_size = block_size / mat_size;

	// Morton stuff (2)
	const size_t morton_block_levels = log2(max_block)+1; // round-up (take front size_teger and add 1)
	size_t * const morton_block_indices = calloc(morton_block_levels, sizeof(size_t));

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

	int block_diff;
	double block_diff_log;

	// MORTON-CURVE ONLY (3)
	size_t mask;
	size_t level;
	size_t inc_game;
	size_t offset;

  libxsmm_dmmfunction kernel; 
  MKL_INT k,l,m;

	if (mode!= dim-1) {
			// k,l,m are all 
			// Mathematically, 4x2, 2x3 (because we want to remove from left side)
			k = l_dimension; // 4
			l = lda; // 3
			m = n; // 2
  		// kernel = libxsmm_dmmdispatch(l, k, m, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
			// sizes available: n, lda
	} else {
			k = lda2;//2;
			l = l_dimension;//4;
			m = n;//3;
		// kernel = libxsmm_dmmdispatch(l, k, m, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	}


	size_t el = 0;
	while (1) {

		//printf("mode=%d -> n=%d, lda=%d, left_mat_size=%d, mat_size=%d, right_size=%d\n",
		//		mode, n, lda, left_mat_size, mat_size, right_size);

		if (mode != dim-1) {
			next = 0;
			next_result = 0;


if (TEST_ENV==1) {
		printf("LEFT MODE\n");
		printf("We are doing (a loop of) %d, %d, %d removal\n", k, l, m);
}

			for (size_t i=0; i<left_mat_size; ++i) {
				const double * next = (tensor->lin.data + global_tensor) + i*mat_size;
				double * next_result = (result->data + global_result) + i*right_size*l_dimension;
				// cblas_dgemm(
				// CblasRowMajor, // const CBLAS_LAYOUT
				// // diff 1
				// CblasNoTrans, CblasNoTrans,
				// // diff 2
				// k, l, m,
				// alpha, // const double
				// (matrix->data + global_vector), m, // const double*, const MKL_size_t
				// next, l, // const double*, const MKL_size_t
				// beta, // const float
				// next_result, l); 

				cblas_dgemm(
				CblasRowMajor, // const CBLAS_LAYOUT
				// diff 1
				CblasTrans, CblasNoTrans,
				// diff 2
				k, l, m,
				alpha, // const double
				(matrix->data + global_vector), l, // const double*, const MKL_size_t
				next, l, // const double*, const MKL_size_t
				beta, // const float
				next_result, l); // const double*, const MKL_size_t

  				// kernel(next, matrix->data + global_vector, next_result, NULL, NULL, NULL);

			}
		} else {
			// MATHEMATICALLY IT IS
			// 2x3 and 3x4 matrices

if (TEST_ENV==1) {
		printf("RIGHT MODE\n");
		printf("We are doing %d, %d, %d removal\n", k, l, m);
}

			// cblas_dgemm(
			// CblasRowMajor, // const CBLAS_LAYOUT
			// CblasNoTrans, CblasNoTrans,
			// k, l, m,
			// alpha, // const double
			// tensor->lin.data + global_tensor, m, // const double*, const MKL_size_t
			// (matrix->data + global_vector), l, // const double*, const MKL_size_t
			// beta, // const float
			// (result->data + global_result), l); // const double*, const MKL_size_t

			cblas_dgemm(
			CblasRowMajor, // const CBLAS_LAYOUT
			CblasNoTrans, CblasTrans,
			k, l, m,
			alpha, // const double
			tensor->lin.data + global_tensor, m, // const double*, const MKL_size_t
			(matrix->data + global_vector), m, // const double*, const MKL_size_t
			beta, // const float
			(result->data + global_result), l); // const double*, const MKL_size_t

		  // kernel(matrix->data + global_vector, tensor->lin.data + global_tensor, result->data + global_result, NULL, NULL, NULL);
  
		}

if (TEST_ENV==1) {
			printf("After finishing that blocked we have:\n");
			// print_to_console(result->data, result->size);
}

		if (++el == blocks) {
			break;
		}

		global_tensor += block_size;
		global_result += (result_size*l_dimension);

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
			if (block_diff < (int) level) {
				if (block_diff > 0) {
				for (int i=0; i<=block_diff-1; ++i) {
					morton_block_indices[i] = global_result;
				}
				}
			} else {
				if (level > 0) {
				for (int i=0; i<=(int)level-1; ++i) {
					morton_block_indices[i] = global_result;
				}
				}
			}
		}
		global_vector = block_counter[mode] * tensor->block_layout[mode] * l_dimension;

if (TEST_ENV==1) {
		// SUPER USEFUL DEBUGGING
		printf("Move to global_vector=%zu, global_result=%zu, global_tensor=%zu\n", global_vector, global_result, global_tensor);
}

	}

	free(morton_block_indices);
	free(block_counter);
	free(block_counter_threshold);
}


// are we right in not using l_block_dimension?
void
tmm_mortonblocked_mkl(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict matrix, struct lin_storage * result, const size_t mode, const int l_dimension, const int l_block_dimension) {

	const size_t dim = tensor->dim;
	// Morton stuff (1)
	size_t * const block_counter = calloc(dim, sizeof(size_t));
	size_t * const block_counter_threshold = calloc(dim, sizeof(size_t));
	// compute: right, block, result sizes
	// + blocks, block_counter_thresholds, max_block
	size_t right_size = 1;
	size_t block_size = 1;
	size_t blocks = 1;
	size_t max_block = 0;
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
	}
	const size_t vector_size = tensor->block_layout[mode];
	const size_t result_size = block_size / vector_size;
	const size_t mat_size = right_size * vector_size;
	const size_t left_mat_size = block_size / mat_size;

	// Morton stuff (2)
	const size_t morton_block_levels = log2(max_block)+1; // round-up (take front size_teger and add 1)
	size_t * const morton_block_indices = calloc(morton_block_levels, sizeof(size_t));

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

	int block_diff;
	double block_diff_log;

	// MORTON-CURVE ONLY (3)
	size_t mask;
	size_t level;
	size_t inc_game;
	size_t offset;

  libxsmm_dmmfunction kernel; 
  MKL_INT k,l,m;

	if (mode!= dim-1) {
			// k,l,m are all 
			// Mathematically, 4x2, 2x3 (because we want to remove from left side)
			k = l_dimension; // 4
			l = lda; // 3
			m = n; // 2
  		kernel = libxsmm_dmmdispatch(l, k, m, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
			// sizes available: n, lda
	} else {
			k = lda2;//2;
			l = l_dimension;//4;
			m = n;//3;
		kernel = libxsmm_dmmdispatch(l, k, m, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	}


	size_t el = 0;
	while (1) {

		//printf("mode=%d -> n=%d, lda=%d, left_mat_size=%d, mat_size=%d, right_size=%d\n",
		//		mode, n, lda, left_mat_size, mat_size, right_size);

		if (mode != dim-1) {
			next = 0;
			next_result = 0;


if (TEST_ENV==1) {
		printf("LEFT MODE\n");
		printf("We are doing (a loop of) %d, %d, %d removal\n", k, l, m);
}

			for (size_t i=0; i<left_mat_size; ++i) {
				const double * next = (tensor->lin.data + global_tensor) + i*mat_size;
				double * next_result = (result->data + global_result) + i*right_size*l_dimension;
				cblas_dgemm(
				CblasRowMajor, // const CBLAS_LAYOUT
				// diff 1
				CblasNoTrans, CblasNoTrans,
				// diff 2
				k, l, m,
				alpha, // const double
				(matrix->data + global_vector), m, // const double*, const MKL_size_t
				next, l, // const double*, const MKL_size_t
				beta, // const float
				next_result, l); 

  				// kernel(next, matrix->data + global_vector, next_result, NULL, NULL, NULL);

			}
		} else {
			// MATHEMATICALLY IT IS
			// 2x3 and 3x4 matrices

if (TEST_ENV==1) {
		printf("RIGHT MODE\n");
		printf("We are doing %d, %d, %d removal\n", k, l, m);
}

			cblas_dgemm(
			CblasRowMajor, // const CBLAS_LAYOUT
			CblasNoTrans, CblasNoTrans,
			k, l, m,
			alpha, // const double
			tensor->lin.data + global_tensor, m, // const double*, const MKL_size_t
			(matrix->data + global_vector), l, // const double*, const MKL_size_t
			beta, // const float
			(result->data + global_result), l); // const double*, const MKL_size_t

		  // kernel(matrix->data + global_vector, tensor->lin.data + global_tensor, result->data + global_result, NULL, NULL, NULL);
  
		}

if (TEST_ENV==1) {
			printf("After finishing that blocked we have:\n");
			// print_to_console(result->data, result->size);
}

		if (++el == blocks) {
			break;
		}

		global_tensor += block_size;
		global_result += (result_size*l_dimension);

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
			if (block_diff < (int) level) {
				if (block_diff > 0) {
				for (int i=0; i<=block_diff-1; ++i) {
					morton_block_indices[i] = global_result;
				}
				}
			} else {
				if (level > 0) {
				for (int i=0; i<=(int)level-1; ++i) {
					morton_block_indices[i] = global_result;
				}
				}
			}
		}
		global_vector = block_counter[mode] * tensor->block_layout[mode] * l_dimension;

if (TEST_ENV==1) {
		// SUPER USEFUL DEBUGGING
		printf("Move to global_vector=%zu, global_result=%zu, global_tensor=%zu\n", global_vector, global_result, global_tensor);
}

	}

	free(morton_block_indices);
	free(block_counter);
	free(block_counter_threshold);
}
