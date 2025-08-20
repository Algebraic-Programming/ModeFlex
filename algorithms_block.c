#include<algorithms.h>
#include<rand_utils.h>
#include<file_utils.h>
#include<string.h>
#include<stdlib.h>
#include<math.h>
#include<space_curves.h>
#include<mkl.h>

/////////////////////////////////////// SPIS TRESCI
////////// BLAS_POWERS_bench	WORKS
////////// BLAS_POWERS		WORKS -> v2
////////// POWERS_2  		WORKS
////////// POWERS     		little bug
////////// POWERS_3  		more than little bug(s)

////////////////// VERSION: UNFOLD only

void
tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_UNFOLD(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

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

	const size_t result_size = block_size / vector_size;
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

	size_t UNFOLD_REQ;
	DTYPE * unfold;
	if (mode != dim-1) {
		UNFOLD_REQ = 1;
		unfold = malloc(block_size * sizeof(DTYPE));
	} else {
		UNFOLD_REQ = 0;
	}

	for ( size_t el=0; ; ) {

		// Let's make this the same as case number 3)



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
		} else {
			next += block_size;
		}

		if (++el == blocks) {
			break;
		}

		global_result += result_size;

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

		//printf("so finally, the first element is: %d\n", unfold[0]);

	}

	if (UNFOLD_REQ) {
		result_tensor->data[0] = unfold[0];
		free(unfold);
	}

	free(mul);

}

////////////////// VERSION: BLAS only : no UNFOLD

void
tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_BLAS(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

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
	for (size_t d=dim-1; d<dim; --d) {
		if (d > mode) {
			right_size *= tensor->block_layout[tensor->layout_perm[d]];
		} else if (d < mode) {
			left_size *= tensor->block_layout[tensor->layout_perm[d]];
		}
		block_size *= tensor->block_layout[tensor->layout_perm[d]];
	}

	const size_t result_size = block_size / vector_size;
	size_t really_global_result = 0;
	size_t global_result = 0;
	size_t global_vector = 0;
	size_t next = 0;
	const double alpha = 1;
	const double beta = 1;
	const MKL_INT incx = 1;
	const MKL_INT incy = 1;
	const MKL_INT lda = vector_size;

	for ( size_t el=0;;) {

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
				break;
			}
		}
	}
	free(mul);
}

////////// BLAS_POWERS_v2	WORKS but when I was silly with mat loop...

void
tvm_block_major_input_aligned_output_aligned_aux_v1(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

	const size_t dim = tensor->dim;
	//const size_t vector_size = tensor->block_layout[mode];
	size_t block_size = 1;
	for (size_t d=dim-1; d<dim; --d) {
		block_size *= tensor->block_layout[tensor->layout_perm[d]];
	}

	size_t * block_counter_threshold = calloc(dim, sizeof(size_t));
	size_t * block_counter = calloc(dim, sizeof(size_t));
	size_t blocks = 1;
	for (size_t i=dim; i>0; --i) {
		block_counter_threshold[i-1] = (tensor->layout[i-1] + tensor->block_layout[i-1] -1)
			/ tensor->block_layout[i-1];
		blocks *= block_counter_threshold[i-1];
	}

	size_t really_global_result = 0;
	size_t global_result = 0;
	for (size_t b=0; b<blocks; ++b) {

		++block_counter[dim-1];

		for (size_t d=dim-1; d<=dim-1; --d) {
			if (block_counter[d] == block_counter_threshold[d]) {
				block_counter[d] = 0;
				if (d!=0) {
					++block_counter[d-1];
				}
			} else {
				if (d == mode) {
					global_result = really_global_result;
				} else if (d < mode) {
					really_global_result = global_result;
				}
				break;
			}
		}
	}
	free(block_counter);
	free(block_counter_threshold);
}

///// POWERS version
void
tvm_block_major_input_aligned_output_aligned_aux_v3(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

	const size_t dim = tensor->dim;
	//const size_t vector_size = tensor->block_layout[mode];
	size_t block_size = 1;
	for (size_t d=dim-1; d<dim; --d) {
		block_size *= tensor->block_layout[tensor->layout_perm[d]];
	}
	//const size_t result_size = block_size / vector_size;

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
	size_t really_global_result = 0;
	size_t global_result = 0;

	for(size_t el=0;;) {
		if (++el == blocks) {
			break;
		}

		for (size_t i=0; i<=mode; ++i) {
			if (el % mul[i] == 0) {
				if (i == mode) {
					global_result = really_global_result;
					//global_vector += vector_size;
				} else {
					really_global_result = global_result;
					//global_vector = 0;
				}
				break;
			}
		}
	}
	free(mul);
}

void
tvm_block_major_input_aligned_output_aligned_aux_v2(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

	const size_t dim = tensor->dim;
	//const size_t vector_size = tensor->block_layout[mode];
	size_t block_size = 1;
	for (size_t d=dim-1; d<dim; --d) {
		block_size *= tensor->block_layout[tensor->layout_perm[d]];
	}

	size_t * const block_counter_threshold = malloc(dim * sizeof(size_t));
	size_t * const block_counter = calloc(dim, sizeof(size_t));
	for (size_t d=dim-1; d<dim; --d) {
		block_counter_threshold[d] = (tensor->layout[d]+tensor->block_layout[d]-1)/tensor->block_layout[d];
	}

	size_t really_global_result = 0;
	size_t global_result = 0;
	size_t d;

	for(;;) {
	
		d = block_inc(block_counter, block_counter_threshold, dim-1);
		if (d>dim) {
			break;
		} else if (d == mode) {
			global_result = really_global_result;
			//global_vector += vector_block_size;
		} else if (d < mode) {
			really_global_result = global_result;
			//global_vector = 0;
		}

	}
	free(block_counter);
	free(block_counter_threshold);
}

// block BLAS bench ( working!!! )
void
tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_bench(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

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
	size_t block_size = 1;
	for (size_t d=dim-1; d<dim; --d) {
		if (d > mode) {
			right_size *= tensor->block_layout[tensor->layout_perm[d]];
		}
		block_size *= tensor->block_layout[tensor->layout_perm[d]];
	}
	const size_t result_size = block_size / vector_size;

	size_t really_global_result = 0;
	size_t global_result = 0;
	size_t global_vector = 0;
	size_t next = 0;
	const double alpha = 1;
	const double beta = 1;
	const MKL_INT incx = 1;
	const MKL_INT incy = 1;
	const MKL_INT lda = vector_size;

	//MKL_INT n;

	//if (mode != dim-1) {
	//	n = tensor->block_layout[dim-1]; // layout_perm not needed (?) we are referring to original
	//} else {
	//	n = tensor->block_layout[dim-2]; // layout_perm not needed (?) we are referring to original
	//}

	for(size_t el=0;;) {

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
				break;
			}
		}
	}

}

// block - unfold - no BLAS (working?)
void
tvm_block_major_input_aligned_output_aligned_POWERS_2(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

	const size_t dim = tensor->dim;
	const size_t vector_block_size = tensor->block_layout[mode];
	const size_t vector_size = tensor->block_layout[mode];

	size_t * const block_counter_threshold = malloc(dim * sizeof(size_t));
	size_t * const block_counter = calloc(dim, sizeof(size_t));
	for (size_t d=dim-1; d<dim; --d) {
		block_counter_threshold[d] = (tensor->layout[d]+tensor->block_layout[d]-1)/tensor->block_layout[d];
	}

	size_t right_size = 1;
	size_t left_size = 1;
	size_t block_size = 1;

	size_t unfold_mul = 1;
	//size_t unfold_diff = 0;

	for (size_t d=dim-1; d<dim; --d) {
		if (d > mode) {
			right_size *= tensor->block_layout[tensor->layout_perm[d]];
			unfold_mul *= tensor->block_layout[tensor->layout_perm[d]];
		} else if (d < mode) {
			left_size *= tensor->block_layout[tensor->layout_perm[d]];
		}
		block_size *= tensor->block_layout[tensor->layout_perm[d]];
	}
	const size_t result_size = block_size / vector_size;

	size_t really_global_result = 0;
	size_t global_result = 0;
	size_t global_vector = 0;
	size_t next = 0;
	size_t d;

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
	MKL_INT n;

	size_t UNFOLD_REQ;
	DTYPE * unfold;
	if (mode != dim-1) {
		UNFOLD_REQ = 1;
		unfold = malloc(block_size * sizeof(DTYPE));
		n = tensor->block_layout[dim-1]; // layout_perm not needed (?) we are referring to original
	} else {
		// mode is already in the right position
		UNFOLD_REQ = 0;
		n = tensor->block_layout[dim-2]; // layout_perm not needed (?) we are referring to original
	}

	// BLAS matrix-style processing
	const size_t left_mat_size = result_size / n;
	const size_t mat_size = vector_size * n;
	size_t next_result;

	for(;;) {
		
		if (UNFOLD_REQ) {
			next = 0;
			tensor_index = global_tensor;
			tensor_diff = 0;
			for (size_t i=0; i<result_size; ++i) {
				if ((i>0) & (i % right_size == 0)) {
					tensor_diff += stride;
				}
				tensor_index = global_tensor + i + tensor_diff;
				// not using j -> improve this?
				for (size_t j=0; j<vector_size; ++j) {
					unfold[next++] = tensor->lin.data[tensor_index];
					tensor_index += right_size;
				}
			}
			global_tensor += block_size;

			next = 0;
#if 0
			for(size_t i=0; i<result_size; ++i) {
				for(size_t j=0; j<vector_size; ++j) {
					result_tensor->data[global_result + i] += unfold[next++] * vector->data[global_vector + j];
				}
			}
#endif
			next_result = 0;
			for (size_t i=0; i<left_mat_size; ++i) {
				cblas_dgemv(
					CblasRowMajor, CblasNoTrans,
					n, lda, // vector_size, const MKL_INT*2
					alpha, // const double
					(unfold + next), lda, // const double*, const MKL_INT
					(vector->data + global_vector), incx, // const double*, const MKL_INT
					beta, // const double
					(result_tensor->data + global_result + next_result), incy); // const double*, const MKL_INT
				next += mat_size;
				next_result += n;
			}
#if 0
		} else {
			// you never reset 'next'
			for(size_t i=0; i<result_size; ++i) {
				for(size_t j=0; j<vector_size; ++j) {
					result_tensor->data[global_result + i] += tensor->lin.data[next++] * vector->data[global_vector + j];
				}
			}
#endif
			// only for the last mode
		} else {
			// I'm not using i -> ?
			next_result = 0;
			//printf("begin (next=%d, global_result=%d, next_result=%d, global_vector=%d \n", next, global_result, next_result, global_vector);
			for (size_t i=0; i<left_mat_size; ++i) {
				cblas_dgemv(
					CblasRowMajor, CblasNoTrans,
					n, lda, // vector_size, const MKL_INT*2
					alpha, // const double
					(tensor->lin.data + next), lda, // const double*, const MKL_INT
					(vector->data + global_vector), incx, // const double*, const MKL_INT
					beta, // const double
					(result_tensor->data + global_result + next_result), incy); // const double*, const MKL_INT
				//prsize_t_to_console(result_tensor->data+global_result+next_result, block_size);
				// inc next not by 1 but by mat_size
				next += mat_size;
				next_result += n;
			}
		}

		global_result += result_size;
	
		d = block_inc(block_counter, block_counter_threshold, dim-1);
		if (d>dim) {
			break;
		} else if (d == mode) {
			global_result = really_global_result;
			global_vector += vector_block_size;
		} else if (d < mode) {
			really_global_result = global_result;
			global_vector = 0;
		}

	}

	if (UNFOLD_REQ) {
		free(unfold);
	}
	free(block_counter);
	free(block_counter_threshold);
}
 
// input aligned block (working)
void
tvm_block_major_input_aligned_output_aligned_POWERS(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

	const size_t dim = tensor->dim;
	const size_t vector_block_size = tensor->block_layout[mode];
	const size_t vector_size = tensor->block_layout[mode];

	size_t * const block_counter_threshold = malloc(dim * sizeof(size_t));
	size_t * const block_counter = calloc(dim, sizeof(size_t));
	for (size_t d=dim-1; d<dim; --d) {
		block_counter_threshold[d] = (tensor->layout[d]+tensor->block_layout[d]-1)/tensor->block_layout[d];
	}

	size_t right_size = 1;
	size_t left_size = 1;
	size_t block_size = 1;

	size_t unfold_mul = 1;
	//size_t unfold_diff = 0;

	for (size_t d=dim-1; d<dim; --d) {
		if (d > mode) {
			right_size *= tensor->block_layout[tensor->layout_perm[d]];
			unfold_mul *= tensor->block_layout[tensor->layout_perm[d]];
		} else if (d < mode) {
			left_size *= tensor->block_layout[tensor->layout_perm[d]];
		}
		block_size *= tensor->block_layout[tensor->layout_perm[d]];
	}
	size_t result_size = block_size / vector_size;

	size_t really_global_result = 0;
	size_t global_result = 0;
	size_t global_vector = 0;
	size_t t = 0;
	size_t d;

	for(;;) {

		for (size_t i=0; i<result_size; ++i) {
			for (size_t j=0; j<vector_size; ++j) {
				result_tensor->data[global_result + i] += tensor->lin.data[t++] * vector->data[global_vector + j];
			}
		}

#if 0
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
#endif
		
		//global_result += temp_result;
		global_result += result_size;
	
		d = block_inc(block_counter, block_counter_threshold, dim-1);
		if (d>dim) {
			break;
		} else if (d == mode) {
			global_result = really_global_result;
			global_vector += vector_block_size;
		} else if (d < mode) {
			really_global_result = global_result;
			global_vector = 0;
		}

	}

	free(block_counter);
	free(block_counter_threshold);
}

// failed attempt: stuck with erorrs ( unfold - blas )
void
tvm_block_major_input_aligned_output_aligned_POWERS_3(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

	const size_t dim = tensor->dim;
	//const size_t vector_block_size = tensor->block_layout[mode];
	const size_t vector_size = tensor->block_layout[mode];

	// BLAS call constants
	//const double alpha = 1;
	//const double beta = 1;
	//const long incx = 1;
	//const long incy = 1;
	//const long lda = vector_size;

	size_t * const block_counter_threshold = malloc(dim * sizeof(size_t));
	size_t * const block_counter = calloc(dim, sizeof(size_t));

	// this doesnt care about permutation: the order of block counter works the same
	for (size_t d=dim-1; d<dim; --d) {
		block_counter_threshold[d] = (tensor->layout[d]+tensor->block_layout[d]-1)/tensor->block_layout[d];
	}

	size_t right_size = 1;
	size_t left_size = 1;
	size_t block_size = 1;
	//size_t result_size = 1;
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


	// for block processing
	size_t stride;
	if (mode != 0) {
		stride = right_size * (tensor->block_layout[mode] - 1);
	} else {
		stride = 0;
	}

	size_t UNFOLD_REQ;
	DTYPE * unfold;
	//size_t n;
	if (mode != dim-1) {
		UNFOLD_REQ = 1;
		unfold = malloc(block_size * sizeof(DTYPE));
		//n = tensor->block_layout[dim-1];
	} else {
		// mode is already in the right position
		UNFOLD_REQ = 0;
		//n = tensor->block_layout[dim-2];
	}
	
	// d stands for ... dimension to be inc by block_counter
	// THE ONLY REQUIRED TO BE SIGNED VARIABLE
	size_t d;

	// for matrix-based BLAS call
        //const size_t part_result_size = n;
	//const size_t mat_size = vector_size * n;
    //const size_t left_mat_size = block_size / mat_size; // / vector_size / n;

	size_t next, tensor_diff, tensor_index;

	for(;;) {

		// switch more of the friends to size-t?

		// UNFOLD if UNFOLD_REQ (mode != dim-1)
		if (UNFOLD_REQ) {
			next = 0;
			tensor_diff = 0;
			tensor_index = 0;
			for (size_t i=0; i<result_size; ++i) {
				if ((i>0) & (i % right_size == 0)) {
					tensor_diff += stride;
				}
				tensor_index = i + tensor_diff;
				// not using j -> improve this?
				for (size_t j=0; j<vector_size; ++j) {
					unfold[next++] = tensor->lin.data[tensor_index];
					tensor_index += right_size;
				}
			}
		}

		next = 0;
		// now we are aligned within the block!!!
		for (size_t i=0; i<result_size; ++i) {
			for(size_t j=0; j<vector_size; ++j) {
				result_tensor->data[i] += tensor->lin.data[next++] * vector->data[j];
			}
		}
		global_result += result_size; // ????

#if 0
		// BLAS call
		// here: I can ensure that the types are exactly the same as required by the function
		size_t next = 0;
		size_t next_result = 0;

		// I'm not using i at all - fix this?
		for (size_t i=0; i<left_mat_size; ++i) {
			cblas_dgemv(
				CblasRowMajor, CblasNoTrans,
				n, vector_size, // const MKL_INT (s)
				alpha, // const double
				(unfold + next), lda, // const double*, const MKL_INT
				(vector->data + global_vector), incx, // const double*, const MKL_INT
				beta, // const float
				(result_tensor->data + global_result + next_result), incy); // const double*, const MKL_INT
			next += mat_size;
			next_result += n;
		}

		// this is after the loop?
		global_result += next_result;
#endif

#if 0
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
		global_result += temp_result;
#endif

		// Move on to the next block
		d = block_inc(block_counter, block_counter_threshold, dim-1);
		if (d>dim) {
			break;
		} else if (d == mode) {
			global_result = really_global_result;
			global_vector += vector_size;
		} else if (d < mode) {
			really_global_result = global_result;
			global_vector = 0;
		}

	}

	if (UNFOLD_REQ) {
		free(unfold);
	}
	free(block_counter);
	free(block_counter_threshold);
}


void
tvm_block_major_input_aligned_output_aligned(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

	size_t dim = tensor->dim;

	size_t * block_counter_threshold = calloc(dim, sizeof(size_t));
	size_t * block_counter = calloc(dim, sizeof(size_t));
	//size_t * block_counter2 = calloc(dim, sizeof(size_t));
	size_t * remainder_index = calloc(dim, sizeof(size_t));

	size_t blocks = 1;

	// We must compute the block_counter_thresholds (num of blocks in each dimension)
	for (size_t i=dim-1; i<=dim-1; --i) {
		// size_teger division with rounding up formula
		block_counter_threshold[i] = (tensor->layout[i] + tensor->block_layout[i] -1)
			/ tensor->block_layout[i];
		blocks *= block_counter_threshold[i];
		if (tensor->layout[i] % tensor->block_layout[i] != 0) {
			remainder_index[i] = tensor->layout[i] % tensor->block_layout[i];
		} else {
			remainder_index[i] = tensor->block_layout[i];
			// when it's not an "edgy" block-dim then just assign regular size
		}
	}

	size_t vector_block_size = tensor->block_layout[mode];

	long really_global_result = 0;
	long global_result = 0;
	long global_vector = 0;

	long t = 0;

	for(size_t b=0; b<blocks; ++b) {

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
		long temp_result = 0;
		for(size_t i=0; i<left_size; ++i) {
			for(size_t v=0; v<vector_size; ++v) {
				for(size_t j=0; j<right_size; ++j) {
					result_tensor->data[global_result+temp_result+j] += 
						tensor->lin.data[t++] * vector->data[global_vector+v];
				}
			}
			temp_result += right_size;
		}

		//if (b == blocks-1) {
			//break;
		//}

		global_result += temp_result;

		++block_counter[dim-1]; 
		// carry the ticks over
		for (size_t d=dim-1; d<=dim-1; --d) {
			// this guy reached the max - reset him and inc the next counter
			if (block_counter[d] == block_counter_threshold[d]) {
				block_counter[d] = 0;
				if (d!=0) {
					++block_counter[d-1];
				}
			} else {
				// we incremented d (not reached threshold, just incremented)
				if (d == mode) {
					global_result = really_global_result;
					global_vector += vector_block_size;
				} else if (d < mode) {
					really_global_result = global_result;
					global_vector = 0;
				}
				// we can exit the loop - we are in dim which did not reach threshold
				// so therefore no more tasks 
				break;
			}
		}

	}
	free(remainder_index);
	free(block_counter);
	free(block_counter_threshold);
}
// BROKEN DUE TO SIZE_T
#if 0
void
tvm_block_major_input_aligned_output_aligned_3(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

	size_t dim = tensor->dim;
	size_t * block_counter_threshold = calloc(dim, sizeof(size_t));
	size_t * block_counter = calloc(dim, sizeof(size_t));
	size_t * remainder_index = calloc(dim, sizeof(size_t));
	size_t blocks = 1;
	// We must compute the block_counter_thresholds (num of blocks in each dimension)
	for (size_t i=dim; i>0; --i) {
		// size_teger division with rounding up formula
		block_counter_threshold[i-1] = (tensor->layout[i-1] + tensor->block_layout[i-1] -1)
			/ tensor->block_layout[i-1];
		blocks *= block_counter_threshold[i-1];
		if (tensor->layout[i-1] % tensor->block_layout[i-1] != 0) {
			remainder_index[i-1] = tensor->layout[i-1] % tensor->block_layout[i-1];
		} else {
			remainder_index[i-1] = tensor->block_layout[i-1];
			// when it's not an "edgy" block-dim then just assign regular size
		}
	}
	size_t vector_block_size = tensor->block_layout[mode];

	size_t really_global_result = 0;
	size_t global_result = 0;

	size_t tensor_index = 0;

	//size_t vector_index = 0;
	size_t global_vector = 0;

	for(size_t b=0; b<blocks; ++b) {
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
						tensor->lin.data[tensor_index++] * vector->data[global_vector+v];
				}
			}
			temp_result += right_size;
		}
		global_result += temp_result;
		////////// INCREMENT BLOCK COUNTERS
		++block_counter[dim-1]; 
		// carry the ticks over
		for (size_t d=dim-1; d<=dim-1; --d) {
			// this guy reached the max - reset him and inc the next counter
			if (block_counter[d] == block_counter_threshold[d]) {
				block_counter[d] = 0;
				if (d!=0) {
					++block_counter[d-1];
				}
			} else {
				// we incremented d (not reached threshold, just incremented)
				if (d == mode) {
					global_result = really_global_result;
					global_vector += vector_block_size;
				} else if (d < mode) {
					really_global_result = global_result;
					global_vector = 0;
				}
				// we can exit the loop - we are in dim which did not reach threshold
				// so therefore no more tasks 
				break;
			}
		}
	}
	free(remainder_index);
	free(block_counter);
	free(block_counter_threshold);
}
#endif
void
tvm_block_major(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {
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
	for (size_t i=dim; i>0; --i) {
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
	long result_index = 0;
	long tensor_index = 0;
	size_t result_offset = 0;
	size_t tensor_offset = 0;
	size_t vector_index = 0;
	// BLOCK-LEVEL LOOP (using counter method)
	for(size_t b=0; b<blocks; ++b) {
#if 0
		new_mul[dim-1] = 1;
		for (size_t d=dim-2; d<=dim-2; --d) {
			if (d+1 == mode) {
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
#endif 
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
		for(size_t t=0; t<block_size; ++t) {
			vector_index = block_counter[mode] * tensor->block_layout[mode] + counter[mode];
			//printf("vector_index=%d\n", vector_index);
			//printf("tensor=%d\n",tensor_index+tensor_offset);
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
		++block_counter[dim-1];
		for (size_t d=dim-1; d<=dim-1; --d) {
			if (block_counter[d] == block_counter_threshold[d]) {
				if (d!=0) {
					++block_counter[d-1];
				}
				// alt: put below if size_to the above if (???)
				if (d != mode) {
					result_index = 0;
				}
				block_counter[d] = 0;
			}
			if (d != mode) {
				result_index += block_counter[d] * mul[d] * tensor->block_layout[d];
			}
			tensor_index += block_counter[d] * block_mul[d] * tensor->block_layout[d];
		}
		//prsize_t_to_console_size_t(block_counter, dim);
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
tvm_block_major_old(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

	size_t dim = tensor->dim;
	// We must compute the block_counter_thresholds (num of blocks in each dimension)
	size_t * block_counter_threshold = calloc(dim, sizeof(size_t));
	size_t * block_counter = calloc(dim, sizeof(size_t));
	size_t * remainder_index = calloc(dim, sizeof(size_t));
	// counter_threshold are basically block_layout sizes
	size_t * mul = calloc(tensor->dim, sizeof(size_t));
	size_t * block_mul = calloc(tensor->dim, sizeof(size_t));

	size_t block_size = 1;
	size_t blocks = 1;
	//size_t right_block_size = 1;
	mul[dim-1] = 1;
	block_mul[dim-1] = 1;
	// PRECOMPUTE block_counter_threshold
	for (size_t i=dim; i>0; --i) {

		if (i-1>0) {
			if (i-1 == mode) {
				mul[i-2] = mul[i-1];
			} else {
				mul[i-2] = mul[i-1] * tensor->layout[tensor->layout_perm[i-1]];
			}
			block_mul[i-2] = block_mul[i-1] * tensor->layout[tensor->layout_perm[i-1]];
			//printf("mul[%d=%d\n", i-2, mul[i-2]);
			//block_mul[i-2] = block_mul[i-1] * tensor->block_layout[i-1];
			//if (tensor->layout[i-1] < block_mul[i-2]) {
				//block_mul[i-2] = tensor->layout[i-1];
			//}
		}

		//if (i-1>mode) {
			//right_block_size *= tensor->block_layout[i-1];
		//}
		// size_teger division with rounding up formula :)
		block_counter_threshold[i-1] = (tensor->layout[i-1] + tensor->block_layout[i-1] -1)
			/ tensor->block_layout[i-1];
		blocks *= block_counter_threshold[i-1];
		block_size *= tensor->block_layout[i-1];
		if (tensor->layout[i-1] % tensor->block_layout[i-1] != 0) {
			remainder_index[i-1] = tensor->layout[i-1] % tensor->block_layout[i-1];
		} else {
			remainder_index[i-1] = tensor->block_layout[i-1];
			//remainder_index[i-1] = -1;
		}
	}
	//prsize_t_to_console_size_t(block_counter_threshold, dim);
	//prsize_t_to_console_size_t(remainder_index, dim);
	//printf("blocks = %d\n", blocks);

	long tensor_index = 0;
	long result_index = 0;
	long result_offset = 0;
	size_t offset = 0;
	size_t skip_me = 0;
	size_t vector_index = 0;

	// BLOCK-LEVEL LOOP (using counter method)
	for(size_t b=0; b<blocks; ++b) {
		//printf("(block) tensor_index = %d\n", tensor_index);
		// AT THIS POsize_t: we have a starting index of the block to be processed
		// TENSOR-LEVEL LOOP
		offset = 0;
	 	vector_index = 0;
		size_t * counter = calloc(dim, sizeof(size_t));
		skip_me = 0;
		//printf("STARTING A NEW BLOCK!!!!\n");
		for(size_t t=0; t<block_size; ++t) {
			//printf("guard skip_me =%d \n", skip_me);
			if (skip_me == 0) {
				//block_tensor->data[next++] = tensor->lin.data[tensor_index+offset];
				//printf("counter[mode]=%d, block_counter[mode]=%d\n", counter[mode], block_counter[mode]);
				vector_index = block_counter[mode] * tensor->block_layout[mode] + counter[mode];
				//printf("tensor=%d, vector=%d, result=%d \n", 
						//tensor_index+offset, vector_index, result_index+result_offset);
				//size_t test = tensor->lin.data[tensor_index+offset];
				result_tensor->data[result_index+result_offset] += 
					tensor->lin.data[tensor_index+offset] * vector->data[vector_index];
				//size_t test2 = result_tensor->data[result_index+result_offset];
				//size_t test3 = vector->data[vector_index];
				//printf("%d %d %d \n", test, test2, test3);
			}
			//} else {
				//printf("we don't go in - why?\n");
			//}
			// process the block
			result_offset = 0;
			//reset_result_offset = 0;
			offset = 0;
			// tick the smallest dimension....
			//prsize_t_to_console_size_t(counter, dim);
			++counter[dim-1];
			// carry the ticks over
			for (size_t d=dim; d!=0; --d) {

				//printf("we are here - before all?\n");
				if (block_counter_threshold[d-1]-1 == block_counter[d-1]) {
					//printf("we are in last block for certain dimension");
					if (remainder_index[d-1] <= counter[d-1]) {
						//printf("skip it!\n");
						skip_me = 1;
						break;
					}
				}
				skip_me = 0;
				//printf("SKIP_ME STATUS=%d\n", skip_me);
				if (counter[d-1] == tensor->block_layout[d-1]) {
					// increment the higher dimension (if possible)
					if (d-1>0) {
						++counter[d-2];
					}
					if (d-1==mode) {
						result_offset = 0;
						//reset_result_offset = 1;
						//printf("reset result_offset because we updated mode\n");
					}
					// reset the current dimension
					counter[d-1] = 0;
				}
				// multiply the absolute location
				offset += counter[d-1] * block_mul[d-1];
				if (d-1!=mode) {
					//printf("this is not mode so inc result_offset by %d (counter%d * mul%d) \n", 
							//counter[d-1] * mul[d-1], counter[d-1], mul[d-1]);
					result_offset += counter[d-1] * mul[d-1];
				}
				//printf("result_offset=%d\n", result_offset);
			}
		}
		tensor_index = 0;
		result_index = 0;
		// tick the smallest dimension on block_counters
		++block_counter[dim-1];
		// carry the ticks over
		for (size_t d=dim; d!=0; --d) {
			if (block_counter[d-1] == block_counter_threshold[d-1]) {
				// increment the higher dimension (if possible)
				if (d-1>0) {
					++block_counter[d-2];
					if (d-1==mode) {
						result_index = 0;
						//printf("RESET NOW?\n");
					}	
				}
				// reset the current dimension
				block_counter[d-1] = 0;
			}
			//printf("inc by %d\n", block_counter[d-1] * mul[d-1] * tensor->block_layout[d-1]);
			if (d-1 != mode) {
					result_index += block_counter[d-1] * mul[d-1] * tensor->block_layout[d-1];
					//printf("We incremented result_index, %d \n", result_index);
			}
			// multiply the absolute location
			tensor_index += block_counter[d-1] * block_mul[d-1] * tensor->block_layout[d-1];
			//printf("we inc now ten_in to %d !\n", tensor_index);
			//printf("%d, %d, %d, %d\n", tensor_index, block_counter[d-1], mul[d-1], tensor->block_layout[d-1]);
		}
		free(counter);
	}
	//printf("THE END \n");
	free(mul);
	free(block_counter);
	free(block_counter_threshold);
}

void
tvm_block_major_input_aligned_old(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

	size_t dim = tensor->dim;
	// We must compute the block_counter_thresholds (num of blocks in each dimension)
	size_t * block_counter_threshold = calloc(dim, sizeof(size_t));
	size_t * block_counter = calloc(dim, sizeof(size_t));
	size_t * remainder_index = calloc(dim, sizeof(size_t));
	// counter_threshold are basically block_layout sizes
	size_t * mul = calloc(tensor->dim, sizeof(size_t));
	size_t * block_mul = calloc(tensor->dim, sizeof(size_t));

	size_t block_size = 1;
	size_t blocks = 1;
	//size_t right_block_size = 1;
	mul[dim-1] = 1;
	block_mul[dim-1] = 1;
	// PRECOMPUTE block_counter_threshold
	for (size_t i=dim; i>0; --i) {

		if (i-1>0) {
			if (i-1 == mode) {
				mul[i-2] = mul[i-1];
			} else {
				mul[i-2] = mul[i-1] * tensor->layout[tensor->layout_perm[i-1]];
			}
			block_mul[i-2] = block_mul[i-1] * tensor->layout[tensor->layout_perm[i-1]];
			//printf("mul[%d=%d\n", i-2, mul[i-2]);
			//block_mul[i-2] = block_mul[i-1] * tensor->block_layout[i-1];
			//if (tensor->layout[i-1] < block_mul[i-2]) {
				//block_mul[i-2] = tensor->layout[i-1];
			//}
		}

		//if (i-1>mode) {
			//right_block_size *= tensor->block_layout[i-1];
		//}
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
	//prsize_t_to_console_size_t(block_counter_threshold, dim);
	//prsize_t_to_console_size_t(remainder_index, dim);
	//printf("blocks = %d\n", blocks);

	long tensor_index = 0;
	long result_index = 0;
	long result_offset = 0;
	size_t skip_me = 0;
	size_t vector_index = 0;

	// BLOCK-LEVEL LOOP (using counter method)
	for(size_t b=0; b<blocks; ++b) {
		//printf("(block) tensor_index = %d\n", tensor_index);
		// AT THIS POsize_t: we have a starting index of the block to be processed
		// TENSOR-LEVEL LOOP
		//offset = 0;
	 	vector_index = 0;
		size_t * counter = calloc(dim, sizeof(size_t));
		skip_me = 0;
		//printf("STARTING A NEW BLOCK!!!!\n");
		for(size_t t=0; t<block_size; ++t) {
			//printf("guard skip_me =%d \n", skip_me);
			if (skip_me == 0) {
				//block_tensor->data[next++] = tensor->lin.data[tensor_index+offset];
				//printf("counter[mode]=%d, block_counter[mode]=%d\n", counter[mode], block_counter[mode]);
				vector_index = block_counter[mode] * tensor->block_layout[mode] + counter[mode];
				//printf("tensor=%d, vector=%d, result=%d \n", 
						//tensor_index, vector_index, result_index+result_offset);
				//size_t test = tensor->lin.data[tensor_index+offset];
				result_tensor->data[result_index+result_offset] += 
					tensor->lin.data[tensor_index++] * vector->data[vector_index];
				//printf("res_index=%d, res-offset=%d\n", result_index, result_offset);
				//size_t test2 = result_tensor->data[result_index+result_offset];
				//size_t test3 = vector->data[vector_index];
				//printf("%d %d %d \n", test, test2, test3);
			}
			//} else {
				//printf("we don't go in - why?\n");
			//}
			// process the block
			result_offset = 0;
			//reset_result_offset = 0;
			//offset = 0;
			// tick the smallest dimension....
			//prsize_t_to_console_size_t(counter, dim);
			++counter[dim-1];
			// carry the ticks over

			for (size_t d=dim; d!=0; --d) {

				if (block_counter_threshold[d-1]-1 == block_counter[d-1]) {
					//printf("we are in last block for certain dimension");
					if (remainder_index[d-1] <= counter[d-1]) {
						//printf("skip it!\n");
						skip_me = 1;
						break;
					}
				}
				skip_me = 0;
				//printf("SKIP_ME STATUS=%d\n", skip_me);
				if (counter[d-1] == tensor->block_layout[d-1]) {
					// increment the higher dimension (if possible)
					if (d-1>0) {
						++counter[d-2];
					}
					if (d-1==mode) {
						result_offset = 0;
						//printf("reset cause it's mode, %d\n", result_offset);
						//printf("reset occured");
						//reset_result_offset = 1;
						//printf("reset result_offset because we updated mode\n");
					}
					// reset the current dimension
					counter[d-1] = 0;
				}
				// multiply the absolute location
				//offset += counter[d-1] * block_mul[d-1];
				if (d-1!=mode) {
					//printf("this is not mode so inc result_offset by %d (counter%d * mul%d) \n", 
							//counter[d-1] * mul[d-1], counter[d-1], mul[d-1]);
					result_offset += counter[d-1] * mul[d-1];
					//printf("addition made inc by %d*%d to off=%d \n", counter[d-1], mul[d-1], result_offset);
				}

				//prsize_t_to_console_size_t(counter,dim);
				//printf("result_offset=%d\n", result_offset);
			}
		}
		//tensor_index = 0;
		result_index = 0;
		// tick the smallest dimension on block_counters
		++block_counter[dim-1];
		// carry the ticks over
		for (size_t d=dim; d!=0; --d) {
			if (block_counter[d-1] == block_counter_threshold[d-1]) {
				// increment the higher dimension (if possible)
				if (d-1>0) {
					++block_counter[d-2];
					if (d-1==mode) {
						result_index = 0;
						//printf("RESET NOW?\n");
					}	
				}
				// reset the current dimension
				block_counter[d-1] = 0;
			}
			//printf("inc by %d\n", block_counter[d-1] * mul[d-1] * tensor->block_layout[d-1]);
			if (d-1 != mode) {
					result_index += block_counter[d-1] * mul[d-1] * tensor->block_layout[d-1];
					//printf("We incremented result_index, %d \n", result_index);
			}
			// multiply the absolute location
			//tensor_index += block_counter[d-1] * block_mul[d-1] * tensor->block_layout[d-1];
			//printf("we inc now ten_in to %d !\n", tensor_index);
			//printf("%d, %d, %d, %d\n", tensor_index, block_counter[d-1], mul[d-1], tensor->block_layout[d-1]);
		}
		free(counter);
	}
	//printf("THE END \n");
	free(mul);
	free(block_counter);
	free(block_counter_threshold);
}

void
tvm_block_major_input_aligned(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

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
	for (size_t i=dim; i>0; --i) {
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
	long result_index = 0;
	long tensor_index = 0;
	long result_offset = 0;
	size_t vector_index = 0;
	// BLOCK-LEVEL LOOP (using counter method)
	for(size_t b=0; b<blocks; ++b) {
#if 0
		new_mul[dim-1] = 1;
		for (size_t d=dim-2; d<=dim-2; --d) {
			if (d+1 == mode) {
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
#endif
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
		for(size_t t=0; t<block_size; ++t) {
			vector_index = block_counter[mode] * tensor->block_layout[mode] + counter[mode];
			//printf("vector_index=%d\n", vector_index);
			//printf("ten=%d, res_ind=%d, res_off=%d, res=%d, vec=%d \n", tensor_index, result_index, result_offset, result_offset+result_index, vector_index);
			result_tensor->data[result_index+result_offset] +=
				tensor->lin.data[tensor_index++] * vector->data[vector_index];
			result_offset = 0;
			++counter[dim-1];



			//printf("input_major only function: ");
			//prsize_t_to_console_size_t(mul, dim);



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
						//printf("reset res_offset\n");
					}
					counter[d] = 0;
				}
				if (d != mode) {
					result_offset += counter[d] * mul[d];
					//printf("inc res_offset to %d (%d*%d) \n", result_offset, counter[d], mul[d]);
				}
			}
			// handle the 0 case
			if (counter[0] == limits[0]) {
				break;
			}

			if (0 != mode) {
				//printf("before result_offset=%d\n", result_offset);
				result_offset += counter[0] * mul[0];
				//printf("(0) inc res_offset to %d (%d*%d) \n", result_offset, counter[0], mul[0]);
			}


		}
		result_index = 0;
		++block_counter[dim-1];
		for (size_t d=dim-1; d<=dim-1; --d) {
			if (block_counter[d] == block_counter_threshold[d]) {
				if (d!=0) {
					++block_counter[d-1];
				}
				// alt: put below if size_to the above if (???)
				if (d != mode) {
					result_index = 0;
				}
				block_counter[d] = 0;
				//if (d == mode) {
				//} else if (d < mode) {
					//result_index += block_counter[d-1] * mul[d-1] * tensor->block_layout[d-1];
				//}
				//break;
			}
			if (d != mode) {
				result_index += block_counter[d] * mul[d] * tensor->block_layout[d];
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

void
tvm_block_major_input_aligned_output_aligned_2(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

	size_t dim = tensor->dim;
	// We must compute the block_counter_thresholds (num of blocks in each dimension)
	size_t * block_counter_threshold = calloc(dim, sizeof(size_t));
	size_t * block_counter = calloc(dim, sizeof(size_t));
	size_t * remainder_index = calloc(dim, sizeof(size_t));
	size_t * new_mul = calloc(dim, sizeof(size_t));
	//size_t * mul = calloc(dim, sizeof(size_t));
	// counter_threshold are basically block_layout sizes
	size_t * limits = calloc(dim, sizeof(size_t));
	//size_t * block_mul = calloc(dim, sizeof(size_t));
	size_t block_size = 1;
	size_t blocks = 1;
	//mul[dim-1] = 1;
	//block_mul[dim-1] = 1;
	// PRECOMPUTE block_counter_threshold
	for (size_t i=dim; i>0; --i) {
#if 0
		if (i-1>0) {
			if (i-1 == mode) {
				mul[i-2] = mul[i-1];
			} else {
				mul[i-2] = mul[i-1] * tensor->layout[tensor->layout_perm[i-1]];
			}
			block_mul[i-2] = block_mul[i-1] * tensor->layout[tensor->layout_perm[i-1]];
		}
#endif
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
	long tensor_index = 0;
	long result_offset = 0;
	long really_global_result = 0;
	size_t vector_index = 0;
	long global_result = 0;
	long last_offset = 0;
	// BLOCK-LEVEL LOOP (using counter method)
	for(size_t b=0; b<blocks; ++b) {
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



#if 0
		for (size_t d=dim; d!=0; --d) {
			if (block_counter_threshold[d-1]-1 == block_counter[d-1]) {
				limits[d-1] = remainder_index[d-1];
			} else {
				limits[d-1] = tensor->block_layout[d-1];
			}
		}
#endif
		for (size_t d=dim-1; d<=dim-1; --d) {
			if (block_counter_threshold[d]-1 == block_counter[d]) {
				limits[d] = remainder_index[d];
				//printf("we entered here for d=%d why? \n", d);
			} else {
				limits[d] = tensor->block_layout[d];
			}
		}


		//printf("BLOCK__________________\n");
		// TENSOR-LEVEL LOOP
	 	vector_index = 0;
		size_t * counter = calloc(dim, sizeof(size_t));
		for(size_t t=0; t<block_size; ++t) {

			//printf("global_result=%ld, result_offset=%ld, sum=%ld \n", global_result, result_offset, global_result+result_offset);

			vector_index = block_counter[mode] * tensor->block_layout[mode] + counter[mode];
			//printf("vector_index=%d\n", vector_index);
			result_tensor->data[global_result+result_offset] +=
				tensor->lin.data[tensor_index++] * vector->data[vector_index];
			// tick the smallest dimension....
			// carry the ticks over
			last_offset = result_offset;
			result_offset = 0;
			++counter[dim-1];
			// change dthe codnition to d!=0... we want the last one to check!

			//printf("output_aligned function: ");
			//prsize_t_to_console_size_t(new_mul, dim);

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
			}
			// break this loop if we reached max on the top dim
			if (counter[0] == limits[0]) {
				global_result += last_offset + 1;
				// this break is imporatnt as it ensures we quit the block once it ticked the top dimension
				// to the limit
				// i.e. we should exit otherwise we will start iterating from 0,0,0 again (e.g.)
				break;
			}
			if (0 != mode) {
				result_offset += counter[0] * new_mul[0];
			}
		}
		// tick the smallest dimension on block_counters
		++block_counter[dim-1];
		//printf("block_counters: ");
		//prsize_t_to_console_size_t(block_counter, dim);
		// carry the ticks over
		for (size_t d=dim-1; d<=dim-1; --d) {
			if (block_counter[d] == block_counter_threshold[d]) {
				// this guy reached the max - reset him and inc the next counter
				block_counter[d] = 0;
				if (d!=0) {
					++block_counter[d-1];
				}
			} else {
				if (d == mode) {
					//printf("mode=%d\n",mode);
					global_result = really_global_result;
					//printf(" SO THATS IT?!!!!!!!!!!!!!!");
				} else if (d < mode) {
					really_global_result = global_result;
				}
				break;
			}
		}
		free(counter);
	}
	//free(mul);
	free(limits);
	free(new_mul);
	//free(block_mul);
	free(remainder_index);
	free(block_counter);
	free(block_counter_threshold);
}

