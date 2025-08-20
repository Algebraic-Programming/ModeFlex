#include <algorithms.h>
#include <mkl.h>
#include <libxsmm.h>
#include <omp.h>
#include <math.h>
#include <file_utils.h>
#include <gen_utils.h> // for reset_array_sizet

#define COUNT_ENABLED 1

#if(TEST_ENV == 1) && !defined(LIKWID_PERFMON)
	#define COUNT_ENABLED 1
#endif

void
tvm_vector_major_BLAS_col_mode_multicore(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result, const size_t mode) {

    const double *const tensor_ptr = tensor->lin.data;
    double *const result_ptr = result->data;

    const MKL_INT incx = 1; 
    const MKL_INT incy = 1; 
    double alpha = 1;
    double beta = 1;
    size_t dim = tensor->dim;
	size_t right_size = 1;
	for (size_t d=dim-1; d<dim; --d) {
		if (d > mode) {
			right_size *= tensor->layout[d];
		}
	}

    const MKL_INT mode_size = vector->size;
    const MKL_INT n = right_size;
    size_t mat_size = (size_t) mode_size * (size_t) n; 
    size_t left_size = tensor->lin.size / mat_size;
    const MKL_INT n2 = result->size;
    
    // int parallelization_count = 0;
    if (mode != dim-1) {
    	// printf("Left size=%zu\n", left_size);
    	#pragma omp parallel for
        for (size_t i=0; i<left_size; ++i) {
        	// #pragma omp critical
        	// {
        	// 	parallelization_count+=1;
        		// printf("Core %d, loop_i=%zu\n", omp_get_thread_num(), i);
		    // }
        	// printf("Loop parallelized; Core %d\n", omp_get_thread_num());
			const double * next = tensor_ptr + i*mat_size;
			double * next_result = result_ptr + i*n;
			cblas_dgemv(
			CblasColMajor, // const CBLAS_LAYOUT
			CblasNoTrans, // const CBLAS_TRANSPOSE
			n, mode_size, // const MKL_size_t (s)
			alpha, // const double
			next, n, // const double*, const MKL_size_t
			vector->data, incx, // const double*, const MKL_size_t
			beta, // const float
			next_result, incy); // const double*, const MKL_size_t
        }
    } else {
		cblas_dgemv(
			CblasRowMajor, // const CBLAS_LAYOUT
			CblasNoTrans, // const CBLAS_TRANSPOSE
			n2, mode_size,
			alpha, // const double
			tensor_ptr, mode_size, // const double*, const MKL_size_t
			vector->data, incx, // const double*, const MKL_size_t
			beta, // const float
			result_ptr, incy); // const double*, const MKL_size_t
    }

    // printf("Parallelized into %zu regions.\n", parallelization_count);

}

void
tvm_vector_major_BLAS_col_mode_multicore2(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result, const size_t mode) {

    const double *const tensor_ptr = tensor->lin.data;
    double *const result_ptr = result->data;

    const MKL_INT incx = 1; 
    const MKL_INT incy = 1; 
    double alpha = 1;
    double beta = 1;
    size_t dim = tensor->dim;
	size_t right_size = 1;
	for (size_t d=dim-1; d<dim; --d) {
		if (d > mode) {
			right_size *= tensor->layout[d];
		}
	}

    const MKL_INT mode_size = vector->size;
    const MKL_INT n = right_size;
    size_t mat_size = (size_t) mode_size * (size_t) n; 
    size_t left_size = tensor->lin.size / mat_size;
    const MKL_INT n2 = result->size;
    
    // int parallelization_count = 0;
    if (mode != dim-1) {
    	// printf("Left size=%zu\n", left_size);
		#pragma omp parallel for simd
		for (size_t i=0; i<left_size; ++i) {
		//   	#pragma omp critical
		//   	{
		//   		parallelization_count+=1;
		//   		printf("Core %d, loop_i=%zu\n", omp_get_thread_num(), i);
		    // }
			// printf("Loop parallelized; Core %d\n", omp_get_thread_num());
			const double * next = tensor_ptr + i*mat_size;
			double * next_result = result_ptr + i*n;
			cblas_dgemv(
			CblasColMajor, // const CBLAS_LAYOUT
			CblasNoTrans, // const CBLAS_TRANSPOSE
			n, mode_size, // const MKL_size_t (s)
			alpha, // const double
			next, n, // const double*, const MKL_size_t
			vector->data, incx, // const double*, const MKL_size_t
			beta, // const float
			next_result, incy); // const double*, const MKL_size_t
		}
    } else {
		cblas_dgemv(
			CblasRowMajor, // const CBLAS_LAYOUT
			CblasNoTrans, // const CBLAS_TRANSPOSE
			n2, mode_size,
			alpha, // const double
			tensor_ptr, mode_size, // const double*, const MKL_size_t
			vector->data, incx, // const double*, const MKL_size_t
			beta, // const float
			result_ptr, incy); // const double*, const MKL_size_t
    }

    // printf("Parallelized into %zu regions.\n", parallelization_count);

}

void
tvm_vector_major_BLAS_col_mode_multicore3(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result, const size_t mode) {

    const double *const tensor_ptr = tensor->lin.data;
    double *const result_ptr = result->data;

    const MKL_INT incx = 1; 
    const MKL_INT incy = 1; 
    double alpha = 1;
    double beta = 1;
    size_t dim = tensor->dim;
	size_t right_size = 1;
	for (size_t d=dim-1; d<dim; --d) {
		if (d > mode) {
			right_size *= tensor->layout[d];
		}
	}

    const MKL_INT mode_size = vector->size;
    const MKL_INT n = right_size;
    size_t mat_size = (size_t) mode_size * (size_t) n; 
    size_t left_size = tensor->lin.size / mat_size;
    const MKL_INT n2 = result->size;

    int nthreads = tensor->lin.p_size;
    int spread_size = left_size / nthreads;
    // printf("spread_size = %d / %d = %d\n", left_size, nthreads, spread_size);
    int spread_loop_count = 1;
    if (spread_size == 0) {
    	// printf("we are in case 0, because %zu div by %d is %d\n", left_size, nthreads, spread_size);
    	spread_size = 1;
    	spread_loop_count = left_size;
        // less than 1 iteration per thread (!) 
        // Dedicate all possible threads to the outer loop
        // Remove the inner loop (!)
        // Use all remaining threads for mkl
        omp_set_num_threads(left_size);
        mkl_set_num_threads(nthreads / left_size);
        // printf("(s=0) we dedicate %d threads to omp, and %d to mkl\n", left_size, nthreads/left_size);
    } else {
    	// printf("we are in case 1, because %zu div by %d is %d\n", left_size, nthreads, spread_size);
        spread_loop_count = nthreads;
        // (at least 1 loop iteration per thread)
        // each openmp thread has something to do...
        // So no threads left for MKL
        // Dedicate all possible threads to the outer loop
        mkl_set_num_threads(1);
        omp_set_num_threads(nthreads);
        // printf("(s!=0) we dedicate %d threads to omp, and %d to mkl\n", nthreads, 1);
    }
    
 	// do not forget we neeed appropriate flags (!) otherwise this algorithm fails
	//    int test_me = 0;
	//    mkl_set_num_threads(1);
	// test_me = mkl_get_max_threads();
	// printf("test_me = %d\n", test_me);

    #if (TEST_ENV == 1)
    	int mkl_threads, current_omp_threads;
	    #pragma omp parallel
	    #pragma omp single
	    	current_omp_threads = omp_get_num_threads();
    	mkl_threads = mkl_get_max_threads();
	    // printf("INFO: BASELINE3: omp threads = %d, mkl threads = %d which is in total no more than what we have initially (%d)\n", current_omp_threads, mkl_threads, nthreads);
	    assert(current_omp_threads * mkl_threads <= nthreads);
	    assert(current_omp_threads * mkl_threads >= nthreads/2);
    #endif

    if (mode != dim-1) {
    	#pragma omp parallel for
        for (int i=0; i<spread_loop_count; ++i) {
        	// Actual starting position of the loop below: i*spread_size
        	// Size of the loop: spread_size
			for (int j=0; j<spread_size; ++j) {
			  // 	#pragma omp critical
			  // 	{
			  // 		parallelization_count+=1;
					// printf("Core %d, loopI=%d, loopJ=%d, processI=%d\n", omp_get_thread_num(), i, j, i*spread_size+j);
			  //   }
				const double * next = tensor_ptr + (i*spread_size+j)*mat_size;
				double * next_result = result_ptr + (i*spread_size+j)*n;
				cblas_dgemv(
				CblasColMajor, // const CBLAS_LAYOUT
				CblasNoTrans, // const CBLAS_TRANSPOSE
				n, mode_size, // const MKL_size_t (s)
				alpha, // const double
				next, n, // const double*, const MKL_size_t
				vector->data, incx, // const double*, const MKL_size_t
				beta, // const float
				next_result, incy); // const double*, const MKL_size_t
			}
		}
		// There is an implicit barrier here
        if (left_size % nthreads != 0) {
        	mkl_set_num_threads(1);
        	int increment = spread_loop_count*spread_size;
        	int left_size_diff = left_size - increment;
        	// printf("Remainder: %d\n", left_size_diff);
        	#pragma omp parallel for 
	        for (int i=0; i<left_size_diff; ++i) {
				const double * next = tensor_ptr + increment*mat_size + i*mat_size;
				double * next_result = result_ptr + increment*n + i*n;
				cblas_dgemv(
				CblasColMajor, // const CBLAS_LAYOUT
				CblasNoTrans, // const CBLAS_TRANSPOSE
				n, mode_size, // const MKL_size_t (s)
				alpha, // const double
				next, n, // const double*, const MKL_size_t
				vector->data, incx, // const double*, const MKL_size_t
				beta, // const float
				next_result, incy); // const double*, const MKL_size_t
	        }
	    }
    } else {
    	mkl_set_num_threads(nthreads);
		cblas_dgemv(
			CblasRowMajor, // const CBLAS_LAYOUT
			CblasNoTrans, // const CBLAS_TRANSPOSE
			n2, mode_size,
			alpha, // const double
			tensor_ptr, mode_size, // const double*, const MKL_size_t
			vector->data, incx, // const double*, const MKL_size_t
			beta, // const float
			result_ptr, incy); // const double*, const MKL_size_t
    }

    // if (left_size % spread_size	!= 0) {
    // 	exit(-1);
    // }
}

// // This is where I divide the chunks by myself(!)
// // This solution will not work unless I find a way to enlarge the vector memory appropriately... could be smart?
// void
// tvm_vector_major_BLAS_col_mode_multicore3(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result, const size_t mode) {

//     const double *const tensor_ptr = tensor->lin.data;
//     double *const result_ptr = result->data;

//     const MKL_INT incx = 1; 
//     const MKL_INT incy = 1; 
//     double alpha = 1;
//     double beta = 1;
//     size_t dim = tensor->dim;
// 	size_t right_size = 1;
// 	for (size_t d=dim-1; d<dim; --d) {
// 		if (d > mode) {
// 			right_size *= tensor->layout[d];
// 		}
// 	}

//     const MKL_INT mode_size = vector->size;
//     const MKL_INT n = right_size;
//     size_t mat_size = (size_t) mode_size * (size_t) n; 
//     size_t left_size = tensor->lin.size / mat_size;
//     const MKL_INT n2 = result->size;

//     int nthreads;
//     #pragma omp parallel
//     #pragma omp single
//     	nthreads = omp_get_num_threads();

//     printf("THREADS = %d\n", nthreads);
//     int spread_size = left_size / nthreads;
//     if (spread_size == 0) {
//     	spread_size = 1;
//     }
// 	int spread_loop_count = left_size / spread_size;
//     // for instance, 20/8=2, 20/2=10 which is new counter (!)
//     const MKL_INT n_enlarged = n * spread_size;
//     printf("left_size=%zu, spread_size=%d, spread_loop_count=%d\n", left_size, spread_size, spread_loop_count);
//     int parallelization_count = 0;

//     if (mode != dim-1) {
//     	#pragma omp parallel for
//         for (size_t i=0; i<spread_loop_count; ++i) {
//         	#pragma omp critical
//         	{
//         		parallelization_count+=1;
//         		printf("Core %d, loop_i=%zu\n", omp_get_thread_num(), i*spread_size);

//         	printf("Loop parallelized; Core %d\n", omp_get_thread_num());
// 			const double * next = tensor_ptr + i*spread_size*mat_size;
// 			double * next_result = result_ptr + i*spread_size*n;
// 			cblas_dgemv(
// 			CblasColMajor, // const CBLAS_LAYOUT
// 			CblasNoTrans, // const CBLAS_TRANSPOSE
// 			n_enlarged, mode_size, // const MKL_size_t (s)
// 			alpha, // const double
// 			next, n_enlarged, // const double*, const MKL_size_t
// 			vector->data, incx, // const double*, const MKL_size_t
// 			beta, // const float
// 			next_result, spread_size); // const double*, const MKL_size_t
// 			// printf("Current result:\n");
// 			// print_to_console_double(result->data, result->size);
// 		    }
//         }
//         if (left_size % nthreads != 0) {
//         	int left_size_diff = left_size - (spread_loop_count);
//         	printf("Remainder: %d, so we jump by %zu el in tensor and %zu el in result\n", left_size_diff, spread_loop_count*mat_size, spread_loop_count*n);
// 	    	#pragma omp parallel for
// 	        for (size_t i=0; i<left_size_diff; ++i) {
// 	        	#pragma omp critical
// 	        	{
// 	        		parallelization_count+=1;
// 	        		printf("Core %d, loop_i=%zu\n", omp_get_thread_num(), i);
// 			    }
// 	        	printf("Loop parallelized; Core %d\n", omp_get_thread_num());
// 				const double * next = tensor_ptr + spread_loop_count*mat_size + i*mat_size;
// 				double * next_result = result_ptr + spread_loop_count*n + i*n;
// 				cblas_dgemv(
// 				CblasColMajor, // const CBLAS_LAYOUT
// 				CblasNoTrans, // const CBLAS_TRANSPOSE
// 				n, mode_size, // const MKL_size_t (s)
// 				alpha, // const double
// 				next, n, // const double*, const MKL_size_t
// 				vector->data, incx, // const double*, const MKL_size_t
// 				beta, // const float
// 				next_result, incy); // const double*, const MKL_size_t
// 	        }
//         }
//     } else {
// 		cblas_dgemv(
// 			CblasRowMajor, // const CBLAS_LAYOUT
// 			CblasNoTrans, // const CBLAS_TRANSPOSE
// 			n2, mode_size,
// 			alpha, // const double
// 			tensor_ptr, mode_size, // const double*, const MKL_size_t
// 			vector->data, incx, // const double*, const MKL_size_t
// 			beta, // const float
// 			result_ptr, incy); // const double*, const MKL_size_t
//     }

// }

void
tvm_ppower_sync(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

	printf("tvm_ppower_sync\n");

	#if (TEST_ENV == 1 && COUNT_ENABLED == 1)
	// Array of use-counts for each object
	int * const tensor_used = calloc(tensor->lin.size, sizeof(int));
	int * const result_used = calloc(result_tensor->size, sizeof(int));
	int * const vector_used = calloc(vector->size, sizeof(int));
	#endif

	// For now: shitty solution with modifying the structures (!) they were const before
	const size_t dim = tensor->dim;
	size_t * const stripe_mul = malloc(dim * sizeof(size_t));
	size_t * const stripei = malloc(dim * sizeof(size_t));
	stripe_mul[dim-1] = 1;
	size_t stripe_size = 1;
	size_t stripes = 1;
	size_t stripe_mode = 1;
	size_t stripe_left = 1;
	// Wow this is important, that this look is constructed by going downwards (because of the way I compute the stripe left and mode!)
	for (size_t i=dim-1; i<dim; --i) {
		stripe_size *= tensor->layout2[i];
		stripei[i] = tensor->layout[i] / tensor->layout2[i];
		stripes *= stripei[i]; // TRICK: make it think that its not striping over mode 0
		if (i > mode) {
			stripe_mode *= stripei[i];
		} else if (i == mode) {
			stripe_left = stripe_mode * stripei[i];
		}
		if (i!=0) {
			stripe_mul[i-1] = stripe_mul[i] * stripei[i];
		}
	}
	stripes = stripes / (tensor->layout[0] / tensor->layout2[0]);
	size_t output_stripes;
	if (mode == 0) {
		output_stripes = stripes;
	} else {
		output_stripes = stripes / (tensor->layout[mode] / tensor->layout2[mode]);
	}
	size_t output_stripe_size = stripe_size / tensor->layout2[mode];
	size_t vector_stripes = stripes / output_stripes;
	size_t * const block_counter_threshold = malloc(dim * sizeof(size_t));
	size_t * const mul = malloc(dim * sizeof(size_t));
	size_t mul_mode = 1;
	size_t mul_left = 1;
	size_t right_size = 1;
	size_t block_size = 1;
	size_t blocks = 1;
	size_t max_block = 0;
	mul[dim-1] = 1;
	for (size_t i=dim-1; i<dim; --i) {
		size_t temp = tensor->block_layout[tensor->layout_perm[i]];
		if (i > mode) {
			right_size *= temp;
		}
		block_size *= temp;
		block_counter_threshold[i] = (tensor->layout2[i] + temp -1) / temp;
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
	const size_t morton_block_levels = log2(max_block)+1; // round-up (take front size_teger and add 1)

	// printf("Number of parts p=%zu\n", stripes);

	int nthreads;
	#pragma omp parallel
	{
		nthreads = omp_get_num_threads();
	}
	if (nthreads > (int) stripei[0]) nthreads = stripei[0];
	// printf("We limit number of threads to max(threads=%d, stripes_at_mode_0=%d)\n", nthreads, stripei[0]);
	// check if p < p_sync in which case we can use at most p processors (and the rest is idle!)
	// same thing happens if we are not divisible by p, right?
	#pragma omp parallel num_threads(nthreads)
	{
		int tid = omp_get_thread_num();

		#pragma omp critical
		{

		// printf("Thread %d beginning at Tensor stripe %d\n", tid, tid);

	    #if (TEST_ENV == 1 && COUNT_ENABLED == 1)
	    // printf("We work with the tensor master_data\n");
	    const double * tensor_ptr = tensor->lin.master_data + tid*stripes*stripe_size + tid*stripe_size;
	    // print_to_console(tensor_ptr, stripe_size);
		#else
	    const double * tensor_ptr = tensor->lin.local_data[tid] + tid*stripe_size;
		#endif

	    // const double * tensor_ptr = tensor->lin.data + tid*stripes*stripe_size + tid*stripe_size; // Navigate to partition, stripe
		double * base_result_ptr = result_tensor->data + tid*output_stripe_size; // TRICK: do not navigate toward partition, just the stripe(!) (for mode=0)

	    // const double * base_vector_ptr = vector->data + tid*tensor->layout2[mode];
	    const double * base_vector_ptr = vector->local_data[tid];

	    if (mode != 0) {
	    	size_t correct_output_stripe = 0;
			for (size_t i=dim-1; i<dim; --i) {
				if (i < mode) { // You only divide by mode if it's greater than mode (!)
					correct_output_stripe += ((tid / stripe_mul[i]) % stripei[i]) * (stripe_mul[i]/stripei[mode]);
				} else if (i == mode) {
					continue;
				} else {
					correct_output_stripe += ((tid / stripe_mul[i]) % stripei[i]) * stripe_mul[i];
				}
	    	}
		    base_vector_ptr = vector->data + ((tid/stripe_mul[mode])%stripei[mode])*tensor->layout2[mode];
		    // base_result_ptr = result_tensor->data + tid*output_stripes*output_stripe_size + correct_output_stripe*output_stripe_size;
		    base_result_ptr = result_tensor->local_data[tid] + correct_output_stripe*output_stripe_size;
		} else {
		}

	    double * result_ptr = base_result_ptr;
	    const double * vector_ptr = base_vector_ptr;

		for (size_t stripe=tid; stripe<stripes+tid; ++stripe) {
			// printf("Stripe=%zu\n", stripe);
			//////////////////////////////////////////////////////////////////////////////////

			libxsmm_dmmfunction kernel;
			size_t * const block_counter = calloc(dim, sizeof(size_t));
			size_t * const morton_block_indices = calloc(morton_block_levels, sizeof(size_t));
			size_t global_tensor = 0;
			size_t global_result = 0;
			size_t global_vector = 0;
			size_t old_global_vector = 0;
			size_t mask;
			int level;
			size_t inc_game;
			size_t offset;
			int block_diff;
			double block_diff_log;
		    if (mode != dim-1) {
		  		kernel = libxsmm_dmmdispatch(right_size, 1, vector_size, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
		  	} else {
		  		kernel = libxsmm_dmmdispatch(1, result_size, vector_size, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
		  	}
			size_t el = 0;
			while (1) {
				// printf("Processing a single block!\n");
				if (mode != dim-1) {
					size_t next = 0;
					size_t next_result = 0;
					for (size_t i=0; i<left_mat_size; ++i) {
					    const double *const tensor_next = tensor_ptr + i*mat_size;
					    double *const result_next = result_ptr + i*right_size;
					    // printf("result_next el used:");
					    // print_to_console(result_next, right_size);
					    // printf("vector el used:");
					    // print_to_console(vector_ptr, vector_size);
					    kernel(tensor_next, vector_ptr, result_next);//, NULL, NULL, NULL);
					    // printf("Done, result is:");
					    // print_to_console(result_next, right_size);
					    #if (TEST_ENV == 1 && COUNT_ENABLED == 1)
						// #pragma omp critical
						// {
					    for (size_t tu=0; tu<mat_size; ++tu) {
					    	int ru = tu % right_size;
					    	int vu = tu % vector_size;
					    	// tensor_used[tensor_next + tu + tid*stripes*stripe_size - (tensor->lin.local_data[tid])] += 1; 
					    	tensor_used[tensor_next + tu - (tensor->lin.master_data)] += 1; 
					    	if (mode == 0) {
					    		vector_used[vector_ptr + vu + tid*tensor->layout2[mode] - (vector->local_data[tid])] += 1; 
					    		result_used[result_next + ru - (result_tensor->data)] += 1; 
					    	} else {
					    		vector_used[vector_ptr + vu - (vector->data)] += 1; 
					    		result_used[result_next + ru + tid*output_stripes*output_stripe_size - (result_tensor->local_data[tid])] += 1; 
					    	}
					    }
						// }
					    #endif
					}
				} else {
				    // printf("result_next el used:");
				    // print_to_console(result_ptr, tensor->layout2[mode]);
				    // printf("vector el used:");
				    // print_to_console(vector_ptr, vector_size);
					kernel(vector_ptr, tensor_ptr, result_ptr);//, NULL, NULL, NULL);
				    // printf("Done, result is:");
				    // print_to_console(result_ptr, tensor->layout2[mode]);
					#if (TEST_ENV == 1 && COUNT_ENABLED == 1)
					// #pragma omp critical
					// {
				    for (size_t tu=0; tu<block_size; ++tu) {
				    	int ru = tu % result_size;
				    	int vu = tu % vector_size;
				    	// tensor_used[tensor_ptr + tu + tid*stripes*stripe_size - (tensor->lin.local_data[tid])] += 1; 
				    	tensor_used[tensor_ptr + tu - (tensor->lin.master_data)] += 1; 
				    	vector_used[vector_ptr + vu - (vector->data)] += 1; 
				    	// result_used[result_ptr + ru - (result_tensor->data)] += 1; 
				    	result_used[result_ptr + ru + tid*output_stripes*output_stripe_size - (result_tensor->local_data[tid])] += 1; 
				    }
					// }
				    #endif
				}

				tensor_ptr += block_size;
				if (++el == blocks) { // TRICK: we must move the code to move the tensor to next block ABOVE (otherwise we do not actually MOVE it
					break;
				}
				// printf("more blocks?\n");
				old_global_vector = block_counter[mode];
				global_result += result_size;

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
					size_t block_diff_abs = block_counter_threshold[mode] - block_counter[mode];
					int block_diff_power = ceil(log2(block_diff_abs));
					if (block_diff_power < level) {
						level = block_diff_power;
					}
					for (int i=0; i<=level-1; ++i) {
						morton_block_indices[i] = global_result;
					}
				}
				result_ptr = base_result_ptr + global_result;
				// printf("We reset (within block algorithm) the result_ptr to base + %zu\n", global_result);
				global_vector = block_counter[mode] * tensor->block_layout[mode];
				vector_ptr = base_vector_ptr + global_vector;
			}
			free(morton_block_indices);
			free(block_counter);

			// tensor_ptr = tensor->lin.data + ((stripe+1)%stripes)*stripe_size + tid*stripes*stripe_size;
			#if (TEST_ENV == 1)
			// printf("We work with global data\n");
			tensor_ptr = tensor->lin.master_data + tid*stripes*stripe_size + ((stripe+1)%stripes)*stripe_size;
			// print_to_console(tensor_ptr, stripe_size);
			#else
			tensor_ptr = tensor->lin.local_data[tid] + ((stripe+1)%stripes)*stripe_size;
			#endif
			// printf("Thread %d moving to stripe %zu\n", tid, (stripe+1)%stripes);

			if (mode == 0) {
				// No change -> we work with the result global result allocated using interleaved allocation (!)
				base_result_ptr = result_tensor->data + ((stripe+1)%output_stripes)*output_stripe_size;
				// #pragma omp barrier

			} else {
				if ((stripe+1) % stripe_left == 0) {
					if (stripe+1 == stripes) {
						// base_result_ptr = result_tensor->data + tid*output_stripes*output_stripe_size;
						base_result_ptr = result_tensor->local_data[tid];
					} else {
						base_result_ptr += stripe_size/tensor->layout2[mode];
					}
					base_vector_ptr = vector->data;
				} else if ((stripe+1) % stripe_mode == 0) {	
					if (stripe+1 == stripes) {
					} else {
						base_result_ptr += stripe_size/tensor->layout2[mode]; // The whole vector_size i.e. layout[mode]
					}
					base_result_ptr -= stripe_mul[mode]*(stripe_size/tensor->layout2[mode]);
					if (stripe+1 == stripes) {
						base_vector_ptr = vector->data;
					} else {
						base_vector_ptr += tensor->layout2[mode];
					}
				} else {
					if (stripe+1 == stripes) {
						// base_result_ptr = result_tensor->data + tid*output_stripes*output_stripe_size;
						base_result_ptr = result_tensor->local_data[tid];
					} else {
						base_result_ptr += stripe_size/tensor->layout2[mode];
					}
				}
			}

			result_ptr = base_result_ptr;
			vector_ptr = base_vector_ptr; 

		}
		} // For the pragma critical (DO NOT FORGET TO DISABLE THE BARRIER)
	}
	free(block_counter_threshold);
	free(mul);
	
	#if (TEST_ENV == 1 && COUNT_ENABLED == 1)
	int result_count = result_used[0];
	int vector_count = vector_used[0];
	int tensor_count = tensor_used[0];
	// printf("INFO: Elements of tensor/result/vector were used correspondingly %d, %d and %d times\n", tensor_count, result_count, vector_count);
	// First verification
	for (size_t i=1; i<tensor->lin.size; ++i) {
		if (tensor_used[i] != tensor_count) {
			printf("ERROR: We used one of the elements DIFFERENT NUMBER OF TIMES (tensor[%zu], %d times), (tensor[%d], %d times)\n",
				i, tensor_used[i], 0, tensor_count);
			exit(-1);
		}
	}
	for (size_t i=1; i<result_tensor->size; ++i) {
		if (result_used[i] != result_count) {
			printf("ERROR: We used one of the elements DIFFERENT NUMBER OF TIMES (result[%zu], %d times), (result[%d], %d times)\n",
				i, result_used[i], 0, result_count);
			exit(-1);
		} 
	}
	for (size_t i=1; i<vector->size; ++i) {
		if (vector_used[i] != vector_count) {
			printf("ERROR: We used one of the elements DIFFERENT NUMBER OF TIMES (vector[%zu], %d times), (vector[%d], %d times)\n",
				i, vector_used[i], 0, vector_count);
			exit(-1);
		} 
	}
	assert(tensor_count == 1);
	assert(result_count == (int) (tensor->lin.size / result_tensor->size));
	assert(vector_count == (int) (tensor->lin.size / vector->size));
	free(vector_used);
	free(result_used);
	free(tensor_used);
	#endif

	free(stripei);
	free(stripe_mul);

}

void
tvm_ppower_sync_mkl(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

	// printf("tvm_ppower_sync\n");

	// BLAS call constants
	const double alpha = 1;
	const double beta = 1;
	const MKL_INT incx = 1;
	const MKL_INT incy = 1;

	#if (TEST_ENV == 1 && COUNT_ENABLED == 1)
	// Array of use-counts for each object
	int * const tensor_used = calloc(tensor->lin.size, sizeof(int));
	int * const result_used = calloc(result_tensor->size, sizeof(int));
	int * const vector_used = calloc(vector->size, sizeof(int));
	#endif

	// For now: shitty solution with modifying the structures (!) they were const before
	const size_t dim = tensor->dim;
	size_t * const stripe_mul = malloc(dim * sizeof(size_t));
	size_t * const stripei = malloc(dim * sizeof(size_t));
	stripe_mul[dim-1] = 1;
	size_t stripe_size = 1;
	size_t stripes = 1;
	size_t stripe_mode = 1;
	size_t stripe_left = 1;
	// Wow this is important, that this look is constructed by going downwards (because of the way I compute the stripe left and mode!)
	for (size_t i=dim-1; i<dim; --i) {
		stripe_size *= tensor->layout2[i];
		stripei[i] = tensor->layout[i] / tensor->layout2[i];
		stripes *= stripei[i]; // TRICK: make it think that its not striping over mode 0
		if (i > mode) {
			stripe_mode *= stripei[i];
		} else if (i == mode) {
			stripe_left = stripe_mode * stripei[i];
		}
		if (i!=0) {
			stripe_mul[i-1] = stripe_mul[i] * stripei[i];
		}
	}
	stripes = stripes / (tensor->layout[0] / tensor->layout2[0]);
	size_t output_stripes;
	if (mode == 0) {
		output_stripes = stripes;
	} else {
		output_stripes = stripes / (tensor->layout[mode] / tensor->layout2[mode]);
	}
	size_t output_stripe_size = stripe_size / tensor->layout2[mode];
	size_t vector_stripes = stripes / output_stripes;
	size_t * const block_counter_threshold = malloc(dim * sizeof(size_t));
	size_t * const mul = malloc(dim * sizeof(size_t));
	size_t mul_mode = 1;
	size_t mul_left = 1;
	size_t right_size = 1;
	size_t block_size = 1;
	size_t blocks = 1;
	size_t max_block = 0;
	mul[dim-1] = 1;
	for (size_t i=dim-1; i<dim; --i) {
		size_t temp = tensor->block_layout[tensor->layout_perm[i]];
		if (i > mode) {
			right_size *= temp;
		}
		block_size *= temp;
		block_counter_threshold[i] = (tensor->layout2[i] + temp -1) / temp;
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
	const size_t morton_block_levels = log2(max_block)+1; // round-up (take front size_teger and add 1)

	// printf("Number of parts p=%zu\n", stripes);

	int nthreads;
	#pragma omp parallel
	{
		nthreads = omp_get_num_threads();
	}
	if (nthreads > (int) stripei[0]) nthreads = stripei[0];
	// printf("We limit number of threads to max(threads=%d, stripes_at_mode_0=%d)\n", nthreads, stripei[0]);
	// check if p < p_sync in which case we can use at most p processors (and the rest is idle!)
	// same thing happens if we are not divisible by p, right?
	#pragma omp parallel num_threads(nthreads)
	{
		int tid = omp_get_thread_num();

		// #pragma omp critical
		// {

		// printf("Thread %d beginning at Tensor stripe %d\n", tid, tid);

	    #if (TEST_ENV == 1)
	    // printf("We work with the tensor master_data\n");
	    const double * tensor_ptr = tensor->lin.master_data + tid*stripes*stripe_size + tid*stripe_size;
	    // print_to_console(tensor_ptr, stripe_size);
		#else
	    const double * tensor_ptr = tensor->lin.local_data[tid] + tid*stripe_size;
		#endif

	    // const double * tensor_ptr = tensor->lin.data + tid*stripes*stripe_size + tid*stripe_size; // Navigate to partition, stripe
		double * base_result_ptr = result_tensor->data + tid*output_stripe_size; // TRICK: do not navigate toward partition, just the stripe(!) (for mode=0)

	    // const double * base_vector_ptr = vector->data + tid*tensor->layout2[mode];
	    const double * base_vector_ptr = vector->local_data[tid];

	    if (mode != 0) {
	    	size_t correct_output_stripe = 0;
			for (size_t i=dim-1; i<dim; --i) {
				if (i < mode) { // You only divide by mode if it's greater than mode (!)
					correct_output_stripe += ((tid / stripe_mul[i]) % stripei[i]) * (stripe_mul[i]/stripei[mode]);
				} else if (i == mode) {
					continue;
				} else {
					correct_output_stripe += ((tid / stripe_mul[i]) % stripei[i]) * stripe_mul[i];
				}
	    	}
		    base_vector_ptr = vector->data + ((tid/stripe_mul[mode])%stripei[mode])*tensor->layout2[mode];
		    // base_result_ptr = result_tensor->data + tid*output_stripes*output_stripe_size + correct_output_stripe*output_stripe_size;
		    base_result_ptr = result_tensor->local_data[tid] + correct_output_stripe*output_stripe_size;
		} else {
		}

	    double * result_ptr = base_result_ptr;
	    const double * vector_ptr = base_vector_ptr;

		for (size_t stripe=tid; stripe<stripes+tid; ++stripe) {
			// printf("Stripe=%zu\n", stripe);
			//////////////////////////////////////////////////////////////////////////////////

			// libxsmm_dmmfunction kernel;
			size_t * const block_counter = calloc(dim, sizeof(size_t));
			size_t * const morton_block_indices = calloc(morton_block_levels, sizeof(size_t));
			size_t global_tensor = 0;
			size_t global_result = 0;
			size_t global_vector = 0;
			size_t old_global_vector = 0;
			size_t mask;
			int level;
			size_t inc_game;
			size_t offset;
			int block_diff;
			double block_diff_log;
		   //  if (mode != dim-1) {
		  	// 	kernel = libxsmm_dmmdispatch(right_size, 1, vector_size, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
		  	// } else {
		  	// 	kernel = libxsmm_dmmdispatch(1, result_size, vector_size, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
		  	// }
			size_t el = 0;
			while (1) {
				// printf("Processing a single block!\n");
				if (mode != dim-1) {
					size_t next = 0;
					size_t next_result = 0;
					for (size_t i=0; i<left_mat_size; ++i) {
					    const double *const tensor_next = tensor_ptr + i*mat_size;
					    double *const result_next = result_ptr + i*right_size;
					    // printf("result_next el used:");
					    // print_to_console(result_next, right_size);
					    // printf("vector el used:");
					    // print_to_console(vector_ptr, vector_size);
					    // kernel(tensor_next, vector_ptr, result_next, NULL, NULL, NULL);
						cblas_dgemv(
							CblasRowMajor, // const CBLAS_LAYOUT
							CblasTrans, // const CBLAS_TRANSPOSE
							vector_size, right_size,
							alpha, // const double
							tensor_next, right_size, // const double*, const MKL_size_t
							vector_ptr, incx, // const double*, const MKL_size_t
							beta, // const float
							result_next, incy); // const double*, const MKL_size_t
					    // printf("Done, result is:");
					    // print_to_console(result_next, right_size);
					    #if (TEST_ENV == 1 && COUNT_ENABLED == 1)
						// #pragma omp critical
						// {
					    for (size_t tu=0; tu<mat_size; ++tu) {
					    	int ru = tu % right_size;
					    	int vu = tu % vector_size;
					    	// tensor_used[tensor_next + tu + tid*stripes*stripe_size - (tensor->lin.local_data[tid])] += 1; 
					    	tensor_used[tensor_next + tu - (tensor->lin.master_data)] += 1; 
					    	if (mode == 0) {
					    		vector_used[vector_ptr + vu + tid*tensor->layout2[mode] - (vector->local_data[tid])] += 1; 
					    		result_used[result_next + ru - (result_tensor->data)] += 1; 
					    	} else {
					    		vector_used[vector_ptr + vu - (vector->data)] += 1; 
					    		result_used[result_next + ru + tid*output_stripes*output_stripe_size - (result_tensor->local_data[tid])] += 1; 
					    	}
					    }
						// }
					    #endif
					}
				} else {
				    // printf("result_next el used:");
				    // print_to_console(result_ptr, tensor->layout2[mode]);
				    // printf("vector el used:");
				    // print_to_console(vector_ptr, vector_size);
					// kernel(vector_ptr, tensor_ptr, result_ptr, NULL, NULL, NULL);
					cblas_dgemv(
						CblasRowMajor, // const CBLAS_LAYOUT
						CblasNoTrans, // const CBLAS_TRANSPOSE
						result_size, vector_size, // const MKL_size_t (s)
						alpha, // const double
						tensor_ptr, vector_size, // const double*, const MKL_size_t
						vector_ptr, incx, // const double*, const MKL_size_t
						beta, // const float
						result_ptr, incy); // const double*, const MKL_size_t
				    // printf("Done, result is:");
				    // print_to_console(result_ptr, tensor->layout2[mode]);
					#if (TEST_ENV == 1 && COUNT_ENABLED == 1)
					// #pragma omp critical
					// {
				    for (size_t tu=0; tu<block_size; ++tu) {
				    	int ru = tu % result_size;
				    	int vu = tu % vector_size;
				    	// tensor_used[tensor_ptr + tu + tid*stripes*stripe_size - (tensor->lin.local_data[tid])] += 1; 
				    	tensor_used[tensor_ptr + tu - (tensor->lin.master_data)] += 1; 
				    	vector_used[vector_ptr + vu - (vector->data)] += 1; 
				    	// result_used[result_ptr + ru - (result_tensor->data)] += 1; 
				    	result_used[result_ptr + ru + tid*output_stripes*output_stripe_size - (result_tensor->local_data[tid])] += 1; 
				    }
					// }
				    #endif
				}

				tensor_ptr += block_size;
				if (++el == blocks) { // TRICK: we must move the code to move the tensor to next block ABOVE (otherwise we do not actually MOVE it
					break;
				}
				// printf("more blocks?\n");
				old_global_vector = block_counter[mode];
				global_result += result_size;

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
					size_t block_diff_abs = block_counter_threshold[mode] - block_counter[mode];
					int block_diff_power = ceil(log2(block_diff_abs));
					if (block_diff_power < level) {
						level = block_diff_power;
					}
					for (int i=0; i<=level-1; ++i) {
						morton_block_indices[i] = global_result;
					}
				}
				result_ptr = base_result_ptr + global_result;
				// printf("We reset (within block algorithm) the result_ptr to base + %zu\n", global_result);
				global_vector = block_counter[mode] * tensor->block_layout[mode];
				vector_ptr = base_vector_ptr + global_vector;
			}
			free(morton_block_indices);
			free(block_counter);

			// tensor_ptr = tensor->lin.data + ((stripe+1)%stripes)*stripe_size + tid*stripes*stripe_size;
			#if (TEST_ENV == 1)
			// printf("We work with global data\n");
			tensor_ptr = tensor->lin.master_data + tid*stripes*stripe_size + ((stripe+1)%stripes)*stripe_size;
			// print_to_console(tensor_ptr, stripe_size);
			#else
			tensor_ptr = tensor->lin.local_data[tid] + ((stripe+1)%stripes)*stripe_size;
			#endif
			// printf("Thread %d moving to stripe %zu\n", tid, (stripe+1)%stripes);

			if (mode == 0) {
				// No change -> we work with the result global result allocated using interleaved allocation (!)
				base_result_ptr = result_tensor->data + ((stripe+1)%output_stripes)*output_stripe_size;
				#pragma omp barrier

			} else {
				if ((stripe+1) % stripe_left == 0) {
					if (stripe+1 == stripes) {
						// base_result_ptr = result_tensor->data + tid*output_stripes*output_stripe_size;
						base_result_ptr = result_tensor->local_data[tid];
					} else {
						base_result_ptr += stripe_size/tensor->layout2[mode];
					}
					base_vector_ptr = vector->data;
				} else if ((stripe+1) % stripe_mode == 0) {	
					if (stripe+1 == stripes) {
					} else {
						base_result_ptr += stripe_size/tensor->layout2[mode]; // The whole vector_size i.e. layout[mode]
					}
					base_result_ptr -= stripe_mul[mode]*(stripe_size/tensor->layout2[mode]);
					if (stripe+1 == stripes) {
						base_vector_ptr = vector->data;
					} else {
						base_vector_ptr += tensor->layout2[mode];
					}
				} else {
					if (stripe+1 == stripes) {
						// base_result_ptr = result_tensor->data + tid*output_stripes*output_stripe_size;
						base_result_ptr = result_tensor->local_data[tid];
					} else {
						base_result_ptr += stripe_size/tensor->layout2[mode];
					}
				}
			}

			result_ptr = base_result_ptr;
			vector_ptr = base_vector_ptr; 

		}
		// } // For the pragma critical (DO NOT FORGET TO DISABLE THE BARRIER)
	}
	free(block_counter_threshold);
	free(mul);
	
	#if (TEST_ENV == 1 && COUNT_ENABLED == 1)
	int result_count = result_used[0];
	int vector_count = vector_used[0];
	int tensor_count = tensor_used[0];
	// printf("INFO: Elements of tensor/result/vector were used correspondingly %d, %d and %d times\n", tensor_count, result_count, vector_count);
	// First verification
	for (size_t i=1; i<tensor->lin.size; ++i) {
		if (tensor_used[i] != tensor_count) {
			printf("ERROR: We used one of the elements DIFFERENT NUMBER OF TIMES (tensor[%zu], %d times), (tensor[%d], %d times)\n",
				i, tensor_used[i], 0, tensor_count);
			exit(-1);
		}
	}
	for (size_t i=1; i<result_tensor->size; ++i) {
		if (result_used[i] != result_count) {
			printf("ERROR: We used one of the elements DIFFERENT NUMBER OF TIMES (result[%zu], %d times), (result[%d], %d times)\n",
				i, result_used[i], 0, result_count);
			exit(-1);
		} 
	}
	for (size_t i=1; i<vector->size; ++i) {
		if (vector_used[i] != vector_count) {
			printf("ERROR: We used one of the elements DIFFERENT NUMBER OF TIMES (vector[%zu], %d times), (vector[%d], %d times)\n",
				i, vector_used[i], 0, vector_count);
			exit(-1);
		} 
	}
	assert(tensor_count == 1);
	assert(result_count == (int) (tensor->lin.size / result_tensor->size));
	assert(vector_count == (int) (tensor->lin.size / vector->size));
	free(vector_used);
	free(result_used);
	free(tensor_used);
	#endif

	free(stripei);
	free(stripe_mul);

}

void
tvm_ppower_sync_mkl_p(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

	const size_t P = tensor->p; // This algorithm is parametrized to distribute along mode P

	// BLAS call constants
	const double alpha = 1;
	const double beta = 1;
	const MKL_INT incx = 1;
	const MKL_INT incy = 1;

	#if (TEST_ENV == 1 && COUNT_ENABLED == 1)
	// Array of use-counts for each object
	int * const tensor_used = calloc(tensor->lin.size, sizeof(int));
	int * const result_used = calloc(result_tensor->size, sizeof(int));
	int * const vector_used = calloc(vector->size, sizeof(int));
	#endif

	// For now: shitty solution with modifying the structures (!) they were const before
	const size_t dim = tensor->dim;
	size_t * const stripe_mul = malloc(dim * sizeof(size_t));
	size_t * const stripei = malloc(dim * sizeof(size_t));
	stripe_mul[dim-1] = 1;
	size_t stripe_size = 1;
	size_t stripes = 1;
	size_t stripe_mode = 1;
	size_t stripe_left = 1;
	// Wow this is important, that this look is constructed by going downwards (because of the way I compute the stripe left and mode!)
	for (size_t i=dim-1; i<dim; --i) {
		stripe_size *= tensor->layout2[i];

		// HOT FIX:
		if (i == P){
			stripei[i] = 1;
		} else {
			stripei[i] = tensor->layout[i] / tensor->layout2[i];
		}
		stripes *= stripei[i];
		// printf("stripei in this dimension=%zu\n", stripei[i]);

		// Number of stripes cannot be affected by above though
		// stripes *= tensor->layout[i] / tensor->layout2[i]; // TRICK: make it think that its not striping over mode 0

		if (i > mode) {
			stripe_mode *= stripei[i];
		} else if (i == mode) {
			stripe_left = stripe_mode * stripei[i];
		}
		if (i!=0) {
			// if (i==mode) {
			// 	stripe_mul[i-1] = stripe_mul[i] * 1;
			// } else {
				stripe_mul[i-1] = stripe_mul[i] * stripei[i];
			// }
		}
	}

	// stripes = stripes / (tensor->layout[P] / tensor->layout2[P]);
	size_t output_stripes;
	if (mode == P) {
		output_stripes = stripes;
	} else {
		output_stripes = stripes / (tensor->layout[mode] / tensor->layout2[mode]);
	}
	
	size_t output_stripe_size = stripe_size / tensor->layout2[mode];
	size_t vector_stripes = stripes / output_stripes;
	size_t * const block_counter_threshold = malloc(dim * sizeof(size_t));
	size_t * const mul = malloc(dim * sizeof(size_t));
	size_t mul_mode = 1;
	size_t mul_left = 1;
	size_t right_size = 1;
	size_t block_size = 1;
	size_t blocks = 1;
	size_t max_block = 0;
	mul[dim-1] = 1;
	for (size_t i=dim-1; i<dim; --i) {
		size_t temp = tensor->block_layout[tensor->layout_perm[i]];
		if (i > mode) {
			right_size *= temp;
		}
		block_size *= temp;
		block_counter_threshold[i] = (tensor->layout2[i] + temp -1) / temp;
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
	const size_t morton_block_levels = log2(max_block)+1; // round-up (take front size_teger and add 1)

	// printf("Number of parts p=%zu\n", stripes);

	int nthreads;
	#pragma omp parallel
	{
		nthreads = omp_get_num_threads();
	}
	// printf("We limit number of threads to max(threads=%d, stripes_at_mode_P=%d)\n", nthreads, tensor->layout[P] / tensor->layout2[P]);
	if (nthreads > (int) (tensor->layout[P] / tensor->layout2[P])) nthreads = tensor->layout[P] / tensor->layout2[P];
	// check if p < p_sync in which case we can use at most p processors (and the rest is idle!)
	// same thing happens if we are not divisible by p, right?
	#if (TEST_ENV == 1) 
	// printf("INFO: using %d threads\n", nthreads);
	#endif
	#pragma omp parallel num_threads(nthreads)
	{
		int tid = omp_get_thread_num();

		// #pragma omp critical
		// {

	    #if (TEST_ENV == 1)
	    // printf("We work with the tensor master_data (thread %d)\n", tid);
	    const double * tensor_ptr = tensor->lin.master_data + tid*stripes*stripe_size + tid*stripe_size;
	    // print_to_console(tensor_ptr, stripe_size);
		#else
	    const double * tensor_ptr = tensor->lin.local_data[tid] + tid*stripe_size;
		#endif

	    // const double * tensor_ptr = tensor->lin.data + tid*stripes*stripe_size + tid*stripe_size; // Navigate to partition, stripe
		double * base_result_ptr = result_tensor->data + tid*output_stripe_size; // TRICK: do not navigate toward partition, just the stripe(!) (for mode=0)

	    // const double * base_vector_ptr = vector->data + tid*tensor->layout2[mode];
	    const double * base_vector_ptr = vector->local_data[tid];

	    if (mode != P) {
	    	// printf("Using local result and global vector!\n");
	    	// printf("We are here.\n");
	    	size_t correct_output_stripe = 0;
			for (size_t i=dim-1; i<=dim-1; --i) {
				// stripe_offset += stripe_counter[d] * stripe_real_mul[d]  * tensor->layout2[d

				// Should we not just ignore the mode?
				if (i > mode) {
				 	// printf("%zu, %zu, %zu, %zu = %zu\n",
				 	// tid, stripe_mul[i], stripei[i], stripe_mul[i], ((tid/stripe_mul[i])%stripei[i]) * (stripe_mul[i]));
					correct_output_stripe += ((tid/stripe_mul[i])%stripei[i]) * (stripe_mul[i]);
				} else if (i < mode) {
				 	// printf("%zu, %zu, %zu, %zu div by %zu = %zu\n",
				 	// tid, stripe_mul[i], stripei[i], stripe_mul[i], stripei[mode], ((tid/stripe_mul[i])%stripei[i]) * (stripe_mul[i]/stripei[mode]));
					correct_output_stripe += ((tid/stripe_mul[i])%stripei[i]) * (stripe_mul[i]/stripei[mode]);
				}

				// 	printf("i=%zu, tid=%d\n", i, tid);
				// 	printf("tid / stripe_mul = %zu\n", tid/stripe_mul[i]); // stripe_mul is based on partition "striping" not real
				//  	printf("stripei is based on %zu\n", stripei[i]);
				//  	printf("then multiplied by %zu\n", stripe_mul[i]);
				// 	correct_output_stripe += ((tid / stripe_mul[i]) % stripei[i]) * stripe_mul[i];
				// 	printf("we added %zu\n", ((tid / stripe_mul[i]) % stripei[i]) * stripe_mul[i]);
				// } else if (i < mode) {
				// 	printf("i=%zu, tid=%d\n", i, tid);
				// 	printf("tid / stripe_mul = %zu\n", tid/stripe_mul[i]); // stripe_mul is based on partition "striping" not real
				//  	printf("stripei is based on %zu\n", stripei[i]);
				//  	printf("then multiplied by %zu\n", stripe_mul[i]);
				// 	correct_output_stripe += (((tid ) % stripei[i]/stripei[mode])) * stripe_mul[i];
				// 	printf("we added %zu\n", (((tid ) % stripei[i]/stripei[mode])) * stripe_mul[i]);
				// }

				// if (i!=P) {
				// 	printf("corr_out_stripe += (pos=%zu) * (mul=%zu) = %zu (perhaps div by %zu?)\n", ((tid/stripe_mul[i]%stripei[i])), stripe_mul[i], stripei[i]);
				// }
				// if (stripe_counter[0] != stripei[P]) {

				// if (i < P) { // You only divide by P if it's greater than P (!)
				// 	correct_output_stripe += ((tid / stripe_mul[i]) % stripei[i]) * (stripe_mul[i]/stripei[P]);
				// } else if (i == P) {
				// 	continue;
				// } else {
				// 	correct_output_stripe += ((tid / stripe_mul[i]) % stripei[i]) * stripe_mul[i];
				// }
	    	}
	    	// printf("correct_output_stripe = %zu\n", correct_output_stripe);
		    base_vector_ptr = vector->data + ((tid/stripe_mul[mode])%stripei[mode])*tensor->layout2[mode];
		    // base_result_ptr = result_tensor->data + tid*output_stripes*output_stripe_size + correct_output_stripe*output_stripe_size;
		    base_result_ptr = result_tensor->local_data[tid] + correct_output_stripe*output_stripe_size;
		    // printf("my part of the result tensor (we take the local data and the offset by %zu:\n", correct_output_stripe*output_stripe_size);
		    // print_to_console(result_tensor->local_data[tid], result_tensor->size/omp_get_num_threads());
		    // printf("my part of result ::::::\n");
		    // print_to_console(base_result_ptr, 4);
		    // print_to_console(base_result_ptr, result_tensor->lin.size/omp_get_num_threads());
		}
		    // printf("my part of result ::::::\n");
		    // print_to_console(base_result_ptr, 4);
	    double * result_ptr = base_result_ptr;
	    // printf("my part of result ::::::\n");
	    // print_to_console(result_ptr, 4);

	    const double * vector_ptr = base_vector_ptr;

		// printf("Thread %d beginning at Tensor stripe %d\n", tid, tid);
		// print_to_console(tensor_ptr, stripe_size);


		for (size_t stripe=tid; stripe<stripes+tid; ++stripe) {
			// printf("Stripe=%zu\n", stripe);
			//////////////////////////////////////////////////////////////////////////////////

			// libxsmm_dmmfunction kernel;
			size_t * const block_counter = calloc(dim, sizeof(size_t));
			size_t * const morton_block_indices = calloc(morton_block_levels, sizeof(size_t));
			size_t global_tensor = 0;
			size_t global_result = 0;
			size_t global_vector = 0;
			size_t old_global_vector = 0;
			size_t mask;
			int level;
			size_t inc_game;
			size_t offset;
			int block_diff;
			double block_diff_log;
		   //  if (mode != dim-1) {
		  	// 	kernel = libxsmm_dmmdispatch(right_size, 1, vector_size, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
		  	// } else {
		  	// 	kernel = libxsmm_dmmdispatch(1, result_size, vector_size, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
		  	// }
			size_t el = 0;
			while (1) {
				// printf("Processing a single block!\n");
				if (mode != dim-1) {
					size_t next = 0;
					size_t next_result = 0;
					for (size_t i=0; i<left_mat_size; ++i) {
					    const double *const tensor_next = tensor_ptr + i*mat_size;
					    double *const result_next = result_ptr + i*right_size;
					    // printf("tensor el used:");
					    // print_to_console(tensor_next, mat_size);
					    // printf("result_next el used:");
					    // print_to_console(result_next, right_size);
					    // printf("vector el used:");
					    // print_to_console(vector_ptr, vector_size);
						cblas_dgemv(
							CblasRowMajor, // const CBLAS_LAYOUT
							CblasTrans, // const CBLAS_TRANSPOSE
							vector_size, right_size,
							alpha, // const double
							tensor_next, right_size, // const double*, const MKL_size_t
							vector_ptr, incx, // const double*, const MKL_size_t
							beta, // const float
							result_next, incy); // const double*, const MKL_size_t
					    // printf("Done, result is:");
					    // print_to_console(result_next, right_size);
					    #if (TEST_ENV == 1 && COUNT_ENABLED == 1)
						// #pragma omp critical
						// {
					    for (size_t tu=0; tu<mat_size; ++tu) {
					    	int ru = tu % right_size;
					    	int vu = tu % vector_size;
					    	// tensor_used[tensor_next + tu + tid*stripes*stripe_size - (tensor->lin.local_data[tid])] += 1; 
					    	tensor_used[tensor_next + tu - (tensor->lin.master_data)] += 1; 
					    	if (mode == P) {
					    		vector_used[vector_ptr + vu + tid*tensor->layout2[mode] - (vector->local_data[tid])] += 1; 
					    		result_used[result_next + ru - (result_tensor->data)] += 1; 
					    	} else {
					    		vector_used[vector_ptr + vu - (vector->data)] += 1; 
					    		result_used[result_next + ru + tid*output_stripes*output_stripe_size - (result_tensor->local_data[tid])] += 1; 
					    	}
					    }
						// }
					    #endif
					}
				} else {
				    // printf("result_next el used:");
				    // print_to_console(result_ptr, tensor->layout2[mode]);
				    // printf("vector el used:");
				    // print_to_console(vector_ptr, vector_size);
					// kernel(vector_ptr, tensor_ptr, result_ptr, NULL, NULL, NULL);
					cblas_dgemv(
						CblasRowMajor, // const CBLAS_LAYOUT
						CblasNoTrans, // const CBLAS_TRANSPOSE
						result_size, vector_size, // const MKL_size_t (s)
						alpha, // const double
						tensor_ptr, vector_size, // const double*, const MKL_size_t
						vector_ptr, incx, // const double*, const MKL_size_t
						beta, // const float
						result_ptr, incy); // const double*, const MKL_size_t
				    // printf("Done, result is:");
				    // print_to_console(result_ptr, tensor->layout2[mode]);
					#if (TEST_ENV == 1 && COUNT_ENABLED == 1)
					// #pragma omp critical
					// {
				    for (size_t tu=0; tu<block_size; ++tu) {
				    	int ru = tu % result_size;
				    	int vu = tu % vector_size;
				    	// tensor_used[tensor_ptr + tu + tid*stripes*stripe_size - (tensor->lin.local_data[tid])] += 1; 
				    	tensor_used[tensor_ptr + tu - (tensor->lin.master_data)] += 1; 
				    	if (mode == P) {
							vector_used[vector_ptr + vu + tid*tensor->layout2[mode] - (vector->local_data[tid])] += 1; 
					    	result_used[result_ptr + ru - (result_tensor->data)] += 1; 	    		
				    	} else {
				    		vector_used[vector_ptr + vu - (vector->data)] += 1; 
				    		result_used[result_ptr + ru + tid*output_stripes*output_stripe_size - (result_tensor->local_data[tid])] += 1; 
				    	}
				    	// vector_used[vector_ptr + vu - (vector->data)] += 1; 
				    	// result_used[result_ptr + ru - (result_tensor->data)] += 1; 
				    	// result_used[result_ptr + ru + tid*output_stripes*output_stripe_size - (result_tensor->local_data[tid])] += 1; 
				    }
					// }
				    #endif
				}

				tensor_ptr += block_size;
				if (++el == blocks) { // TRICK: we must move the code to move the tensor to next block ABOVE (otherwise we do not actually MOVE it
					break;
				}
				// printf("more blocks?\n");
				old_global_vector = block_counter[mode];
				global_result += result_size;

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
					size_t block_diff_abs = block_counter_threshold[mode] - block_counter[mode];
					int block_diff_power = ceil(log2(block_diff_abs));
					if (block_diff_power < level) {
						level = block_diff_power;
					}
					for (int i=0; i<=level-1; ++i) {
						morton_block_indices[i] = global_result;
					}
				}
				result_ptr = base_result_ptr + global_result;
				// printf("We reset (within block algorithm) the result_ptr to base + %zu\n", global_result);
				global_vector = block_counter[mode] * tensor->block_layout[mode];
				vector_ptr = base_vector_ptr + global_vector;
			}
			free(morton_block_indices);
			free(block_counter);

			// tensor_ptr = tensor->lin.data + ((stripe+1)%stripes)*stripe_size + tid*stripes*stripe_size;
			#if (TEST_ENV == 1)
			// printf("We work with global data\n");
			tensor_ptr = tensor->lin.master_data + tid*stripes*stripe_size + ((stripe+1)%stripes)*stripe_size;
			// print_to_console(tensor_ptr, stripe_size);
			#else
			tensor_ptr = tensor->lin.local_data[tid] + ((stripe+1)%stripes)*stripe_size;
			#endif
			// printf("Thread %d moving to stripe %zu\n", tid, (stripe+1)%stripes);

			if (mode == P) {
				// No change -> we work with the result global result allocated using interleaved allocation (!)
				base_result_ptr = result_tensor->data + ((stripe+1)%output_stripes)*output_stripe_size;
				#pragma omp barrier
				// printf("sync point\n");

			} else {
				if ((stripe+1) % stripe_left == 0) {
					// printf("move left of mode because %zu mod %zu\n", stripe+1, stripe_left);
					if (stripe+1 == stripes) {
						// base_result_ptr = result_tensor->data + tid*output_stripes*output_stripe_size;
						base_result_ptr = result_tensor->local_data[tid];
					} else {
						base_result_ptr += stripe_size/tensor->layout2[mode];
					}
					base_vector_ptr = vector->data;
				} else if ((stripe+1) % stripe_mode == 0) {	
					// printf("move along of mode because %zu mod %zu\n", stripe+1, stripe_mode);
					if (stripe+1 == stripes) {
					} else {
						base_result_ptr += stripe_size/tensor->layout2[mode]; // The whole vector_size i.e. layout[mode]
					}
					base_result_ptr -= stripe_mul[mode]*(stripe_size/tensor->layout2[mode]);
					if (stripe+1 == stripes) {
						base_vector_ptr = vector->data;
					} else {
						base_vector_ptr += tensor->layout2[mode];
					}
				} else {
					// printf("move right of mode because %zu is neither div by %zu or %zu\n", stripe+1, stripe_left, stripe_mode); 
					if (stripe+1 == stripes) {
						// base_result_ptr = result_tensor->data + tid*output_stripes*output_stripe_size;
						base_result_ptr = result_tensor->local_data[tid];
					} else {
						base_result_ptr += stripe_size/tensor->layout2[mode];
					}
				}
			}

			result_ptr = base_result_ptr;
			vector_ptr = base_vector_ptr; 

		}
		// } // For the pragma critical (DO NOT FORGET TO DISABLE THE BARRIER)
	}
	free(block_counter_threshold);
	free(mul);
	
	#if (TEST_ENV == 1 && COUNT_ENABLED == 1)
	int result_count = result_used[0];
	int vector_count = vector_used[0];
	int tensor_count = tensor_used[0];
	// printf("INFO: Elements of tensor/result/vector were used correspondingly %d, %d and %d times\n", tensor_count, result_count, vector_count);
	// First verification
	for (size_t i=1; i<tensor->lin.size; ++i) {
		if (tensor_used[i] != tensor_count) {
			printf("ERROR: We used one of the elements DIFFERENT NUMBER OF TIMES (tensor[%zu], %d times), (tensor[%d], %d times)\n",
				i, tensor_used[i], 0, tensor_count);
			exit(-1);
		}
	}
	for (size_t i=1; i<result_tensor->size; ++i) {
		if (result_used[i] != result_count) {
			printf("ERROR: We used one of the elements DIFFERENT NUMBER OF TIMES (result[%zu], %d times), (result[%d], %d times)\n",
				i, result_used[i], 0, result_count);
			exit(-1);
		} 
	}
	for (size_t i=1; i<vector->size; ++i) {
		if (vector_used[i] != vector_count) {
			printf("ERROR: We used one of the elements DIFFERENT NUMBER OF TIMES (vector[%zu], %d times), (vector[%d], %d times)\n",
				i, vector_used[i], 0, vector_count);
			exit(-1);
		} 
	}
	assert(tensor_count == 1);
	assert(result_count == (int) (tensor->lin.size / result_tensor->size));
	assert(vector_count == (int) (tensor->lin.size / vector->size));
	free(vector_used);
	free(result_used);
	free(tensor_used);
	#endif

	free(stripei);
	free(stripe_mul);

}

void
tvm_power_sync_p(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

	const size_t P = tensor->p; // This algorithm is parametrized to distribute along mode P
	const size_t dim = tensor->dim;

	// Modify this algorithm such that each thread has a copy
	size_t stripes = 1;
	size_t stripe_p = 0;
	for (size_t i=dim-1; i<dim; --i) {
		// HOT FIX:
		if (i == P){
			stripe_p = tensor->layout[i] / tensor->layout2[i];
		} else {
			stripes *= tensor->layout[i] / tensor->layout2[i];
		}
	}

	int loop_count = 1;
	int nthreads = 0;
    #pragma omp parallel
    #pragma omp single
    	nthreads = omp_get_num_threads();

   
	// We generally limit to parallelization available at mode P due to distribution
    if (stripe_p < (size_t) nthreads) {
    	nthreads = stripe_p;
		#if (TEST_ENV == 1)
    	printf("INFO: Limit parallelization to dist at mode P (%d)\n", nthreads);
    	#endif
    }
    // However, if we happen to compute along the mode we have to synchronize over
	if (mode == P) {
		// Then we have to see what is there to synchronize over (stripes) and that's how much we can use 
		if (stripes < (size_t) nthreads) {
			nthreads = stripes;
			#if (TEST_ENV == 1)
			printf("INFO: Limit parallelization to number of parts (%d) as we have to synchronize\n", nthreads);
	    	#endif
		}
		// This triggers modification of the loop count
		loop_count = (stripe_p + stripes - 1) / stripes;
		// printf("INFO: Loop_count set to %d\n", loop_count);
	}

	//////////////////////////
	// print_to_console(tensor->lin.data, tensor->lin.size);

	// BLAS call constants
	const double alpha = 1;
	const double beta = 1;
	const MKL_INT incx = 1;
	const MKL_INT incy = 1;

	#if (TEST_ENV == 1 && COUNT_ENABLED == 1)
	// Array of use-counts for each object
	int * const tensor_used = calloc(tensor->lin.size, sizeof(int));
	int * const result_used = calloc(result_tensor->size, sizeof(int));
	int * const vector_used = calloc(vector->size, sizeof(int));
	// printf("we allocated sizes %zu, %zu and %zu\n", tensor->lin.size, result_tensor->size, vector->size);
	#endif

	// For now: shitty solution with modifying the structures (!) they were const before
	size_t * const stripe_mul = malloc(dim * sizeof(size_t));
	size_t * const stripei = malloc(dim * sizeof(size_t));
	stripe_mul[dim-1] = 1;
	size_t stripe_size = 1;
	// size_t stripes = 1;
	size_t stripe_mode = 1;
	size_t stripe_left = 1;
	// size_t stripe_p;
	// Wow this is important, that this look is constructed by going downwards (because of the way I compute the stripe left and mode!)
	for (size_t i=dim-1; i<dim; --i) {
		stripe_size *= tensor->layout2[i];

		// HOT FIX:
		if (i == P){
			stripei[i] = 1;
			// stripe_p = tensor->layout[i] / tensor->layout2[i];
		} else {
			stripei[i] = tensor->layout[i] / tensor->layout2[i];
		}
		// stripes *= stripei[i];
		// printf("stripei in this dimension=%zu\n", stripei[i]);

		// Number of stripes cannot be affected by above though
		// stripes *= tensor->layout[i] / tensor->layout2[i]; // TRICK: make it think that its not striping over mode 0

		if (i > mode) {
			stripe_mode *= stripei[i];
		} else if (i == mode) {
			stripe_left = stripe_mode * stripei[i];
		}
		if (i!=0) {
			// if (i==mode) {
			// 	stripe_mul[i-1] = stripe_mul[i] * 1;
			// } else {
				stripe_mul[i-1] = stripe_mul[i] * stripei[i];
			// }
		}
	}

	// stripes = stripes / (tensor->layout[P] / tensor->layout2[P]);
	size_t output_stripes;
	if (mode == P) {
		output_stripes = stripes;
	} else {
		output_stripes = stripes / (tensor->layout[mode] / tensor->layout2[mode]);
	}
	
	size_t output_stripe_size = stripe_size / tensor->layout2[mode];
	size_t vector_stripes = stripes / output_stripes;
	size_t * const block_counter_threshold = malloc(dim * sizeof(size_t));
	size_t * const mul = malloc(dim * sizeof(size_t));
	size_t mul_mode = 1;
	size_t mul_left = 1;
	size_t right_size = 1;
	size_t block_size = 1;
	size_t blocks = 1;
	size_t max_block = 0;
	mul[dim-1] = 1;
	for (size_t i=dim-1; i<dim; --i) {
		size_t temp = tensor->block_layout[tensor->layout_perm[i]];
		if (i > mode) {
			right_size *= temp;
		}
		block_size *= temp;
		block_counter_threshold[i] = (tensor->layout2[i] + temp -1) / temp;
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
	const size_t morton_block_levels = log2(max_block)+1; // round-up (take front size_teger and add 1)
	//////////////////////////

	#if (TEST_ENV == 1) 
	printf("INFO: using %d threads (loop_count=%d)\n", nthreads, loop_count);
	#endif
	// CAREFUL: as we introduce this loop it means the formula changes from tid to (tid+l*stripes) only if loop_count>1
	for (int l=0; l<loop_count; ++l) {
	// We may have to use even less threads for the last iteration (!)
		if ((mode == P) & (l == loop_count-1)) {
			if (stripe_p % stripes != 0) nthreads = stripe_p % stripes;
			// printf("INFO: last iteration -- using %d threads\n", nthreads);
		} 
	#pragma omp parallel num_threads(nthreads)
	{
		int tid = omp_get_thread_num();
		libxsmm_dmmfunction kernel;
		
		#ifdef INFO
		#pragma omp critical
		{
		#endif

		size_t * const block_counter = calloc(dim, sizeof(size_t));
		size_t * const morton_block_indices = calloc(morton_block_levels, sizeof(size_t));

		#ifdef INFO
		printf("TID = %d\n", tid);
		#endif
		
	    #if (TEST_ENV == 1)
			// Imagine 2 threads running at independent tensor parts, but there is only 1 stripe...
			// That means they should not start at different stripes, but they are forced to start at stripe 0
	    // printf("We work with the tensor master_data, thread %d and stripe %d!\n", (l*stripes)+tid, tid%stripes);
	    const double * restrict tensor_ptr = tensor->lin.master_data + ((l*stripes)+tid)*stripes*stripe_size + (tid%stripes)*stripe_size;
	    if (mode != P) {
	    	tensor_ptr = tensor->lin.master_data + ((l*stripes)+tid)*stripes*stripe_size;
	    	// We are not aligned 1 to 1 with tid == stripe -> we start at stripe 0 always (?)
	    }
	    // print_to_console(tensor_ptr, stripe_size);
		#else
	    const double * restrict tensor_ptr = tensor->lin.local_data[((l*stripes)+tid)] + (tid%stripes)*stripe_size;
	    if (mode != P) {
	    	tensor_ptr = tensor->lin.local_data[((l*stripes)+tid)];
	    }
		#endif

		// printf("We navigate to stripe %d of result!!! IS TAHT OKAY?\n", tid);
	    // const double * tensor_ptr = tensor->lin.data + tid*stripes*stripe_size + tid*stripe_size; // Navigate to partition, stripe
		double * restrict base_result_ptr = result_tensor->data + tid*output_stripe_size; // TRICK: do not navigate toward partition, just the stripe(!) (for mode=0)
		
		#ifdef SYNCHONREMOTE
		double * restrict base_result_ptr2;
		#endif

	    // const double * base_vector_ptr = vector->data + tid*tensor->layout2[mode];
	    const double * restrict base_vector_ptr = vector->local_data[((l*stripes)+tid)];

	    if (mode != P) {
	    	// printf("Using local result and global vector!\n");
	    	// printf("We are here.\n");
	    	size_t correct_output_stripe = 0;
			for (size_t i=dim-1; i<=dim-1; --i) {
				// int tid_adjusted = tid % output_stripes;
				// stripe_offset += stripe_counter[d] * stripe_real_mul[d]  * tensor->layout2[d

				// Should we not just ignore the mode?
				if (i > mode) {
				 	// printf("%zu, %zu, %zu, %zu = %zu\n",
				 	// tid, stripe_mul[i], stripei[i], stripe_mul[i], ((tid/stripe_mul[i])%stripei[i]) * (stripe_mul[i]));
					correct_output_stripe += ((tid/stripe_mul[i])%stripei[i]) * (stripe_mul[i]);
				} else if (i < mode) {
				 	// printf("%zu, %zu, %zu, %zu div by %zu = %zu\n",
				 	// tid, stripe_mul[i], stripei[i], stripe_mul[i], stripei[mode], ((tid/stripe_mul[i])%stripei[i]) * (stripe_mul[i]/stripei[mode]));
					correct_output_stripe += ((tid/stripe_mul[i])%stripei[i]) * (stripe_mul[i]/stripei[mode]);
				}

				// 	printf("i=%zu, tid=%d\n", i, tid);
				// 	printf("tid / stripe_mul = %zu\n", tid/stripe_mul[i]); // stripe_mul is based on partition "striping" not real
				//  	printf("stripei is based on %zu\n", stripei[i]);
				//  	printf("then multiplied by %zu\n", stripe_mul[i]);
				// 	correct_output_stripe += ((tid / stripe_mul[i]) % stripei[i]) * stripe_mul[i];
				// 	printf("we added %zu\n", ((tid / stripe_mul[i]) % stripei[i]) * stripe_mul[i]);
				// } else if (i < mode) {
				// 	printf("i=%zu, tid=%d\n", i, tid);
				// 	printf("tid / stripe_mul = %zu\n", tid/stripe_mul[i]); // stripe_mul is based on partition "striping" not real
				//  	printf("stripei is based on %zu\n", stripei[i]);
				//  	printf("then multiplied by %zu\n", stripe_mul[i]);
				// 	correct_output_stripe += (((tid ) % stripei[i]/stripei[mode])) * stripe_mul[i];
				// 	printf("we added %zu\n", (((tid ) % stripei[i]/stripei[mode])) * stripe_mul[i]);
				// }

				// if (i!=P) {
				// 	printf("corr_out_stripe += (pos=%zu) * (mul=%zu) = %zu (perhaps div by %zu?)\n", ((tid/stripe_mul[i]%stripei[i])), stripe_mul[i], stripei[i]);
				// }
				// if (stripe_counter[0] != stripei[P]) {

				// if (i < P) { // You only divide by P if it's greater than P (!)
				// 	correct_output_stripe += ((tid / stripe_mul[i]) % stripei[i]) * (stripe_mul[i]/stripei[P]);
				// } else if (i == P) {
				// 	continue;
				// } else {
				// 	correct_output_stripe += ((tid / stripe_mul[i]) % stripei[i]) * stripe_mul[i];
				// }
	    	}
	    	// printf("correct_output_stripe = %zu\n", correct_output_stripe);
		    // base_result_ptr = result_tensor->data + tid*output_stripes*output_stripe_size + correct_output_stripe*output_stripe_size;
		    // printf("my part of the result tensor (we take the local data and the offset by %zu:\n", correct_output_stripe*output_stripe_size);
		    // print_to_console(result_tensor->local_data[tid], result_tensor->size/omp_get_num_threads());
		    // printf("my part of result ::::::\n");
		    // print_to_console(base_result_ptr, 4);
		    // print_to_console(base_result_ptr, result_tensor->lin.size/omp_get_num_threads());

		    // base_vector_ptr = vector->data + ((tid/stripe_mul[mode])%stripei[mode])*tensor->layout2[mode];
		    // base_result_ptr = result_tensor->local_data[tid] + correct_output_stripe*output_stripe_size;
		    // We change the game: simply navigate to beginning of the partition (for simplicity!)
	    	base_result_ptr = result_tensor->local_data[tid];
	    	base_vector_ptr = vector->data;
		} else {
		#ifdef SYNCHONREMOTE
			// mode is P, but we still want to use local results(!)
			base_result_ptr2 = result_tensor->local_data[tid];
		#endif
		}
		    // printf("my part of result ::::::\n");
		    // print_to_console(base_result_ptr, 4);
	    double * restrict result_ptr = base_result_ptr;

		#ifdef SYNCHONREMOTE
	    double * restrict result_ptr2 = base_result_ptr2;
	    #endif
	    // printf("my part of result ::::::\n");
	    // print_to_console(result_ptr, 4);

	    const double * restrict vector_ptr = base_vector_ptr;

		// printf("Thread %d beginning at Tensor stripe %d\n", tid, tid);
		// print_to_console(tensor_ptr, stripe_size);

	    int start_tid = tid;
	    if (mode != P) {
	    	// This is very special -- if we do not care about striping (i.e. we are in the case of mode other than P)
	    	// We can simply take any stripe, even stripe=0 or roll over (stripe=(tid%stripes))
	    	start_tid = 0;
	    }

		for (size_t stripe=start_tid; stripe<stripes+start_tid; ++stripe) {
			// printf("Stripe=%zu, stripes+tid=%d\n", stripe, stripes+start_tid);
			//////////////////////////////////////////////////////////////////////////////////

			// libxsmm_dmmfunction kernel;
			reset_array_sizet(block_counter, dim, (size_t) 0);
			reset_array_sizet(morton_block_indices, morton_block_levels, (size_t) 0);

			size_t global_tensor = 0;
			size_t global_result = 0;
			size_t global_vector = 0;
			size_t old_global_vector = 0;
			size_t mask;
			int level;
			size_t inc_game;
			size_t offset;
			int block_diff;
			double block_diff_log;
		    if (mode != dim-1) {
		  		kernel = libxsmm_dmmdispatch(right_size, 1, vector_size, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
		  	} else {
		  		kernel = libxsmm_dmmdispatch(1, result_size, vector_size, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
		  	}
			size_t el = 0;
			while (1) {
				// printf("Processing a single block!\n");
				if (mode != dim-1) {
					size_t next = 0;
					size_t next_result = 0;
					for (size_t i=0; i<left_mat_size; ++i) {
					    const double * restrict const tensor_next = tensor_ptr + i*mat_size;
					    double * restrict const result_next = result_ptr + i*right_size;

						#ifdef SYNCHONREMOTE
					    double * restrict const result_next2 = result_ptr2 + i*right_size;
					    #endif
					    // printf("tensor el used:");
					    // print_to_console(tensor_next, mat_size);
					    // printf("result_next el used:");
					    // print_to_console(result_next, right_size);
					    // printf("vector el used:");
					    // print_to_console(vector_ptr, vector_size);
					    kernel(tensor_next, vector_ptr, result_next);//, NULL, NULL, NULL);

						#ifdef SYNCHONREMOTE
					    if (mode == P) {
					    	kernel(tensor_next, vector_ptr, result_next2, NULL, NULL, NULL);
					    }
					    #endif
					    // kernel(tensor_next, vector_ptr, result_next2, NULL, NULL, NULL);
						// cblas_dgemv(
						// 	CblasRowMajor, // const CBLAS_LAYOUT
						// 	CblasTrans, // const CBLAS_TRANSPOSE
						// 	vector_size, right_size,
						// 	alpha, // const double
						// 	tensor_next, right_size, // const double*, const MKL_size_t
						// 	vector_ptr, incx, // const double*, const MKL_size_t
						// 	beta, // const float
						// 	result_next, incy); // const double*, const MKL_size_t
					    // printf("Done, result is:");
					    // print_to_console(result_next, right_size);
					    #if (TEST_ENV == 1 && COUNT_ENABLED == 1)
					    #ifndef INFO
						#pragma omp critical
						{
						#endif
					    for (size_t tu=0; tu<mat_size; ++tu) {
					    	int ru = tu % right_size;
					    	int vu = tu % vector_size;
					    	// if (mode == P) {
					    	// printf("mode=P, we access element %zu, %zu and %zu\n",
					    	// 	tensor_next + tu - (tensor->lin.master_data),
					    	// 	vector_ptr + vu + ((l*stripes)+tid)*tensor->layout2[mode] - (vector->local_data[((l*stripes)+tid)]),
					    	// 	result_next + ru - (result_tensor->data));
					    	// } else {
					    	// printf("mode!=P, we access element %zu, %zu and %zu\n",
					    	// 	tensor_next + tu - (tensor->lin.master_data),
					    	// 	vector_ptr + vu - (vector->data),
					    	// 	result_next + ru + tid*output_stripes*output_stripe_size - (result_tensor->local_data[tid]));
					    	// }
					    	// tensor_used[tensor_next + tu + tid*stripes*stripe_size - (tensor->lin.local_data[tid])] += 1; 
					    	tensor_used[tensor_next + tu - (tensor->lin.master_data)] += 1; 
					    	if (mode == P) {
					    		vector_used[vector_ptr + vu + ((l*stripes)+tid)*tensor->layout2[mode] - (vector->local_data[((l*stripes)+tid)])] += 1; 
					    		result_used[result_next + ru - (result_tensor->data)] += 1; 
					    	} else {
					    		vector_used[vector_ptr + vu - (vector->data)] += 1; 
					    		result_used[result_next + ru + tid*output_stripes*output_stripe_size - (result_tensor->local_data[tid])] += 1; 
					    	}
					    	// result_used2[result_next2 + ru -(result_tensor->local_data[tid])];
					    #ifndef INFO
					    }
					    #endif
						}
					    #endif
					}
				} else {
				    // printf("tensor el used:");
				    // print_to_console(tensor_ptr, mat_size);
				    // printf("result_next el used:");
				    // print_to_console(result_ptr, tensor->layout2[mode]);
				    // printf("vector el used:");
				    // print_to_console(vector_ptr, vector_size);
					kernel(vector_ptr, tensor_ptr, result_ptr);//, NULL, NULL, NULL);

					#ifdef SYNCHONREMOTE
					if (mode == P) {
						kernel(vector_ptr, tensor_ptr, result_ptr2, NULL, NULL, NULL);
					}
					#endif
					// kernel(vector_ptr, tensor_ptr, result_ptr2, NULL, NULL, NULL);
					// cblas_dgemv(
					// 	CblasRowMajor, // const CBLAS_LAYOUT
					// 	CblasNoTrans, // const CBLAS_TRANSPOSE
					// 	result_size, vector_size, // const MKL_size_t (s)
					// 	alpha, // const double
					// 	tensor_ptr, vector_size, // const double*, const MKL_size_t
					// 	vector_ptr, incx, // const double*, const MKL_size_t
					// 	beta, // const float
					// 	result_ptr, incy); // const double*, const MKL_size_t
				    // printf("Done, result is:");
				    // print_to_console(result_ptr, tensor->layout2[mode]);
					#if (TEST_ENV == 1 && COUNT_ENABLED == 1)
					#ifndef INFO
					#pragma omp critical
					{
					#endif
				    for (size_t tu=0; tu<block_size; ++tu) {
				    	int ru = tu % result_size;
				    	int vu = tu % vector_size;
				    	// if (mode == P) {
				    	// printf("mode=P, we access element %zu, %zu and %zu\n",
				    	// 	tensor_ptr + tu - (tensor->lin.master_data),
				    	// 	vector_ptr + vu + ((l*stripes)+tid)*tensor->layout2[mode] - (vector->local_data[((l*stripes)+tid)]),
				    	// 	result_ptr + ru - (result_tensor->data));
				    	// } else {
				    	// printf("mode!=P, we access element %zu, %zu and %zu\n",
				    	// 	tensor_ptr + tu - (tensor->lin.master_data),
				    	// 	vector_ptr + vu - (vector->data),
				    	// 	result_ptr + ru + tid*output_stripes*output_stripe_size - (result_tensor->local_data[tid]));
				    	// }
				    	// tensor_used[tensor_ptr + tu + tid*stripes*stripe_size - (tensor->lin.local_data[tid])] += 1; 
				    	tensor_used[tensor_ptr + tu - (tensor->lin.master_data)] += 1; 
				    	if (mode == P) {
							vector_used[vector_ptr + vu + ((l*stripes)+tid)*tensor->layout2[mode] - (vector->local_data[(l*stripes)+tid])] += 1; 
					    	result_used[result_ptr + ru - (result_tensor->data)] += 1; 	    		
				    	} else {
				    		vector_used[vector_ptr + vu - (vector->data)] += 1; 
				    		result_used[result_ptr + ru + tid*output_stripes*output_stripe_size - (result_tensor->local_data[tid])] += 1; 
				    	}
				    	// vector_used[vector_ptr + vu - (vector->data)] += 1; 
				    	// result_used[result_ptr + ru - (result_tensor->data)] += 1; 
				    	// result_used[result_ptr + ru + tid*output_stripes*output_stripe_size - (result_tensor->local_data[tid])] += 1; 
				    }
				    #ifndef INFO
					}
					#endif
				    #endif
				}

				tensor_ptr += block_size;
				if (++el == blocks) { // TRICK: we must move the code to move the tensor to next block ABOVE (otherwise we do not actually MOVE it
					break;
				}
				// printf("more blocks?\n");
				old_global_vector = block_counter[mode];
				global_result += result_size;

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
				// printf("are we stuck aboev?\n");
				block_counter[offset] |= mask;
				if (offset == mode) {
					size_t temp = global_result;
					global_result = morton_block_indices[level];
					morton_block_indices[level] = temp;
					size_t block_diff_abs = block_counter_threshold[mode] - block_counter[mode];
					int block_diff_power = ceil(log2(block_diff_abs));
					if (block_diff_power < level) {
						level = block_diff_power;
					}
					for (int i=0; i<=level-1; ++i) {
						morton_block_indices[i] = global_result;
					}
				}
				result_ptr = base_result_ptr + global_result;
				// printf("We reset (within block algorithm) the result_ptr to base + %zu\n", global_result);
				global_vector = block_counter[mode] * tensor->block_layout[mode];
				vector_ptr = base_vector_ptr + global_vector;
			}

			// printf("WE ARE HERE!\n");

			// tensor_ptr = tensor->lin.data + ((stripe+1)%stripes)*stripe_size + tid*stripes*stripe_size;
			#if (TEST_ENV == 1)
			// printf("We work with global data, thread %d, stripe %d\n", tid, ((stripe+1)%stripes));
			tensor_ptr = tensor->lin.master_data + (l*stripes+tid)*stripes*stripe_size + ((stripe+1)%stripes)*stripe_size;
			// print_to_console(tensor_ptr, stripe_size);
			#else
			tensor_ptr = tensor->lin.local_data[(l*stripes+tid)] + ((stripe+1)%stripes)*stripe_size;
			#endif

			// printf("Thread %d moving to stripe %zu\n", tid, (stripe+1)%stripes);

			if (mode == P) {
				// printf("WE ENTER THE BARRIER (from thread %d)\n", tid);
				// No change -> we work with the result global result allocated using interleaved allocation (!)
				base_result_ptr = result_tensor->data + ((stripe+1)%output_stripes)*output_stripe_size;
				#ifndef INFO
				#pragma omp barrier
				#endif
				// printf("sync point\n");

			} else {
				if ((stripe+1) % stripe_left == 0) {
					// printf("move left of mode because %zu mod %zu\n", stripe+1, stripe_left);
					if (stripe+1 == stripes) {
						// base_result_ptr = result_tensor->data + tid*output_stripes*output_stripe_size;
						base_result_ptr = result_tensor->local_data[tid];
						// printf("RES ROZKMINA move left of mode (!): reset the result!\n");
					} else {
						base_result_ptr += stripe_size/tensor->layout2[mode];
						// printf("RES ROZKMINA: move left of mode (!): increment\n");
					}
					base_vector_ptr = vector->data;
				} else if ((stripe+1) % stripe_mode == 0) {	
					// printf("move along of mode because %zu mod %zu\n", stripe+1, stripe_mode);
					if (stripe+1 == stripes) {
					} else {
						base_result_ptr += stripe_size/tensor->layout2[mode]; // The whole vector_size i.e. layout[mode]
					}
					base_result_ptr -= stripe_mul[mode]*(stripe_size/tensor->layout2[mode]);
					if (stripe+1 == stripes) {
						base_vector_ptr = vector->data;
					} else {
						base_vector_ptr += tensor->layout2[mode];
					}
				} else {
					// printf("move right of mode because %zu is neither div by %zu or %zu\n", stripe+1, stripe_left, stripe_mode); 
					if (stripe+1 == stripes) {
						// base_result_ptr = result_tensor->data + tid*output_stripes*output_stripe_size;
						base_result_ptr = result_tensor->local_data[tid];
						// printf("RES ROZKMINA move right of mode(!): reset the result!\n");
					} else {
						base_result_ptr += stripe_size/tensor->layout2[mode];
						// printf("RES ROZKMINA: move right of mode (!): increment\n");
					}
				}
			}

			result_ptr = base_result_ptr;
			vector_ptr = base_vector_ptr; 

		}
		free(morton_block_indices);
		free(block_counter);

		#ifdef INFO
		} // For the pragma critical (DO NOT FORGET TO DISABLE THE BARRIER)
		#endif

	}
	}
	free(mul);
	free(block_counter_threshold);

	#if (TEST_ENV == 1 && COUNT_ENABLED == 1)
	int result_count = result_used[0];
	int vector_count = vector_used[0];
	int tensor_count = tensor_used[0];
	printf("INFO: Elements of tensor/result/vector were used correspondingly %d, %d and %d times\n", tensor_count, result_count, vector_count);
	// First verification
	for (size_t i=1; i<tensor->lin.size; ++i) {
		if (tensor_used[i] != tensor_count) {
			printf("ERROR: We used one of the elements DIFFERENT NUMBER OF TIMES (tensor[%zu], %d times), (tensor[%d], %d times)\n",
				i, tensor_used[i], 0, tensor_count);
			exit(-1);
		}
	}
	for (size_t i=1; i<result_tensor->size; ++i) {
		if (result_used[i] != result_count) {
			printf("ERROR: We used one of the elements DIFFERENT NUMBER OF TIMES (result[%zu], %d times), (result[%d], %d times)\n",
				i, result_used[i], 0, result_count);
			exit(-1);
		} 
	}
	for (size_t i=1; i<vector->size; ++i) {
		if (vector_used[i] != vector_count) {
			printf("ERROR: We used one of the elements DIFFERENT NUMBER OF TIMES (vector[%zu], %d times), (vector[%d], %d times)\n",
				i, vector_used[i], 0, vector_count);
			exit(-1);
		} 
	}
	assert(tensor_count == 1);
	assert(result_count == (int) (tensor->lin.size / result_tensor->size));
	assert(vector_count == (int) (tensor->lin.size / vector->size));
	free(vector_used);
	free(result_used);
	free(tensor_used);
	#endif

	free(stripei);
	free(stripe_mul);

}

void
tvm_power_sync_mkl_p(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

	const size_t P = tensor->p; // This algorithm is parametrized to distribute along mode P
	const size_t dim = tensor->dim;

	// Modify this algorithm such that each thread has a copy
	size_t stripes = 1;
	size_t stripe_p = 0;
	for (size_t i=dim-1; i<dim; --i) {
		// HOT FIX:
		if (i == P){
			stripe_p = tensor->layout[i] / tensor->layout2[i];
		} else {
			stripes *= tensor->layout[i] / tensor->layout2[i];
		}
	}

	int loop_count = 1;
	int nthreads = 0;
    #pragma omp parallel
    #pragma omp single
    	nthreads = omp_get_num_threads();

	// We generally limit to parallelization available at mode P due to distribution
    if (stripe_p < (size_t) nthreads) {
    	nthreads = stripe_p;
    	// printf("INFO: Limit parallelization to dist at mode P (%d)\n", nthreads);
    }
    // However, if we happen to compute along the mode we have to synchronize over
	if (mode == P) {
		// Then we have to see what is there to synchronize over (stripes) and that's how much we can use 
		if (stripes < (size_t) nthreads) {
			nthreads = stripes;
			// printf("INFO: Limit parallelization to number of parts (%d) as we have to synchronize\n", nthreads);
		}
		// This triggers modification of the loop count
		loop_count = (stripe_p + stripes - 1) / stripes;
		// printf("INFO: Loop_count set to %d\n", loop_count);
	}

	//////////////////////////
	// print_to_console(tensor->lin.data, tensor->lin.size);

	// BLAS call constants
	const double alpha = 1;
	const double beta = 1;
	const MKL_INT incx = 1;
	const MKL_INT incy = 1;

	#if (TEST_ENV == 1 && COUNT_ENABLED == 1)
	// Array of use-counts for each object
	int * const tensor_used = calloc(tensor->lin.size, sizeof(int));
	int * const result_used = calloc(result_tensor->size, sizeof(int));
	int * const vector_used = calloc(vector->size, sizeof(int));
	// printf("we allocated sizes %zu, %zu and %zu\n", tensor->lin.size, result_tensor->size, vector->size);
	#endif

	// For now: shitty solution with modifying the structures (!) they were const before
	size_t * const stripe_mul = malloc(dim * sizeof(size_t));
	size_t * const stripei = malloc(dim * sizeof(size_t));
	stripe_mul[dim-1] = 1;
	size_t stripe_size = 1;
	// size_t stripes = 1;
	size_t stripe_mode = 1;
	size_t stripe_left = 1;
	// size_t stripe_p;
	// Wow this is important, that this look is constructed by going downwards (because of the way I compute the stripe left and mode!)
	for (size_t i=dim-1; i<dim; --i) {
		stripe_size *= tensor->layout2[i];

		// HOT FIX:
		if (i == P){
			stripei[i] = 1;
			// stripe_p = tensor->layout[i] / tensor->layout2[i];
		} else {
			stripei[i] = tensor->layout[i] / tensor->layout2[i];
		}
		// stripes *= stripei[i];
		// printf("stripei in this dimension=%zu\n", stripei[i]);

		// Number of stripes cannot be affected by above though
		// stripes *= tensor->layout[i] / tensor->layout2[i]; // TRICK: make it think that its not striping over mode 0

		if (i > mode) {
			stripe_mode *= stripei[i];
		} else if (i == mode) {
			stripe_left = stripe_mode * stripei[i];
		}
		if (i!=0) {
			// if (i==mode) {
			// 	stripe_mul[i-1] = stripe_mul[i] * 1;
			// } else {
				stripe_mul[i-1] = stripe_mul[i] * stripei[i];
			// }
		}
	}

	// stripes = stripes / (tensor->layout[P] / tensor->layout2[P]);
	size_t output_stripes;
	if (mode == P) {
		output_stripes = stripes;
	} else {
		output_stripes = stripes / (tensor->layout[mode] / tensor->layout2[mode]);
	}
	
	size_t output_stripe_size = stripe_size / tensor->layout2[mode];
	size_t vector_stripes = stripes / output_stripes;
	size_t * const block_counter_threshold = malloc(dim * sizeof(size_t));
	size_t * const mul = malloc(dim * sizeof(size_t));
	size_t mul_mode = 1;
	size_t mul_left = 1;
	size_t right_size = 1;
	size_t block_size = 1;
	size_t blocks = 1;
	size_t max_block = 0;
	mul[dim-1] = 1;
	for (size_t i=dim-1; i<dim; --i) {
		size_t temp = tensor->block_layout[tensor->layout_perm[i]];
		if (i > mode) {
			right_size *= temp;
		}
		block_size *= temp;
		block_counter_threshold[i] = (tensor->layout2[i] + temp -1) / temp;
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
	const size_t morton_block_levels = log2(max_block)+1; // round-up (take front size_teger and add 1)
	//////////////////////////

	#if (TEST_ENV == 1) 
	// printf("INFO: using %d threads (loop_count=%d)\n", nthreads, loop_count);
	#endif
	// CAREFUL: as we introduce this loop it means the formula changes from tid to (tid+l*stripes) only if loop_count>1
	for (int l=0; l<loop_count; ++l) {
	// We may have to use even less threads for the last iteration (!)
		if ((mode == P) & (l == loop_count-1)) {
			if (stripe_p % stripes != 0) nthreads = stripe_p % stripes;
			// printf("INFO: last iteration -- using %d threads\n", nthreads);
		} 
	#pragma omp parallel num_threads(nthreads)
	{
		int tid = omp_get_thread_num();
		libxsmm_dmmfunction kernel;

		// #pragma omp critical
		// {
		size_t * const block_counter = calloc(dim, sizeof(size_t));
		size_t * const morton_block_indices = calloc(morton_block_levels, sizeof(size_t));
		// printf("TID = %d\n", tid);

	    #if (TEST_ENV == 1)
			// Imagine 2 threads running at independent tensor parts, but there is only 1 stripe...
			// That means they should not start at different stripes, but they are forced to start at stripe 0
	    // printf("We work with the tensor master_data, thread %d and stripe %d!\n", (l*stripes)+tid, tid%stripes);
	    const double * tensor_ptr = tensor->lin.master_data + ((l*stripes)+tid)*stripes*stripe_size + (tid%stripes)*stripe_size;
	    if (mode != P) {
	    	tensor_ptr = tensor->lin.master_data + ((l*stripes)+tid)*stripes*stripe_size;
	    	// We are not aligned 1 to 1 with tid == stripe -> we start at stripe 0 always (?)
	    }
	    // print_to_console(tensor_ptr, stripe_size);
		#else
	    const double * tensor_ptr = tensor->lin.local_data[((l*stripes)+tid)] + (tid%stripes)*stripe_size;
	    if (mode != P) {
	    	tensor_ptr = tensor->lin.local_data[((l*stripes)+tid)];
	    }
		#endif

		// printf("We navigate to stripe %d of result!!! IS TAHT OKAY?\n", tid);
	    // const double * tensor_ptr = tensor->lin.data + tid*stripes*stripe_size + tid*stripe_size; // Navigate to partition, stripe
		double * base_result_ptr = result_tensor->data + tid*output_stripe_size; // TRICK: do not navigate toward partition, just the stripe(!) (for mode=0)

	    // const double * base_vector_ptr = vector->data + tid*tensor->layout2[mode];
	    const double * base_vector_ptr = vector->local_data[((l*stripes)+tid)];

	    if (mode != P) {
	    	// printf("Using local result and global vector!\n");
	    	// printf("We are here.\n");
	    	size_t correct_output_stripe = 0;
			for (size_t i=dim-1; i<=dim-1; --i) {
				// int tid_adjusted = tid % output_stripes;
				// stripe_offset += stripe_counter[d] * stripe_real_mul[d]  * tensor->layout2[d

				// Should we not just ignore the mode?
				if (i > mode) {
				 	// printf("%zu, %zu, %zu, %zu = %zu\n",
				 	// tid, stripe_mul[i], stripei[i], stripe_mul[i], ((tid/stripe_mul[i])%stripei[i]) * (stripe_mul[i]));
					correct_output_stripe += ((tid/stripe_mul[i])%stripei[i]) * (stripe_mul[i]);
				} else if (i < mode) {
				 	// printf("%zu, %zu, %zu, %zu div by %zu = %zu\n",
				 	// tid, stripe_mul[i], stripei[i], stripe_mul[i], stripei[mode], ((tid/stripe_mul[i])%stripei[i]) * (stripe_mul[i]/stripei[mode]));
					correct_output_stripe += ((tid/stripe_mul[i])%stripei[i]) * (stripe_mul[i]/stripei[mode]);
				}

				// 	printf("i=%zu, tid=%d\n", i, tid);
				// 	printf("tid / stripe_mul = %zu\n", tid/stripe_mul[i]); // stripe_mul is based on partition "striping" not real
				//  	printf("stripei is based on %zu\n", stripei[i]);
				//  	printf("then multiplied by %zu\n", stripe_mul[i]);
				// 	correct_output_stripe += ((tid / stripe_mul[i]) % stripei[i]) * stripe_mul[i];
				// 	printf("we added %zu\n", ((tid / stripe_mul[i]) % stripei[i]) * stripe_mul[i]);
				// } else if (i < mode) {
				// 	printf("i=%zu, tid=%d\n", i, tid);
				// 	printf("tid / stripe_mul = %zu\n", tid/stripe_mul[i]); // stripe_mul is based on partition "striping" not real
				//  	printf("stripei is based on %zu\n", stripei[i]);
				//  	printf("then multiplied by %zu\n", stripe_mul[i]);
				// 	correct_output_stripe += (((tid ) % stripei[i]/stripei[mode])) * stripe_mul[i];
				// 	printf("we added %zu\n", (((tid ) % stripei[i]/stripei[mode])) * stripe_mul[i]);
				// }

				// if (i!=P) {
				// 	printf("corr_out_stripe += (pos=%zu) * (mul=%zu) = %zu (perhaps div by %zu?)\n", ((tid/stripe_mul[i]%stripei[i])), stripe_mul[i], stripei[i]);
				// }
				// if (stripe_counter[0] != stripei[P]) {

				// if (i < P) { // You only divide by P if it's greater than P (!)
				// 	correct_output_stripe += ((tid / stripe_mul[i]) % stripei[i]) * (stripe_mul[i]/stripei[P]);
				// } else if (i == P) {
				// 	continue;
				// } else {
				// 	correct_output_stripe += ((tid / stripe_mul[i]) % stripei[i]) * stripe_mul[i];
				// }
	    	}
	    	// printf("correct_output_stripe = %zu\n", correct_output_stripe);
		    // base_result_ptr = result_tensor->data + tid*output_stripes*output_stripe_size + correct_output_stripe*output_stripe_size;
		    // printf("my part of the result tensor (we take the local data and the offset by %zu:\n", correct_output_stripe*output_stripe_size);
		    // print_to_console(result_tensor->local_data[tid], result_tensor->size/omp_get_num_threads());
		    // printf("my part of result ::::::\n");
		    // print_to_console(base_result_ptr, 4);
		    // print_to_console(base_result_ptr, result_tensor->lin.size/omp_get_num_threads());

		    // base_vector_ptr = vector->data + ((tid/stripe_mul[mode])%stripei[mode])*tensor->layout2[mode];
		    // base_result_ptr = result_tensor->local_data[tid] + correct_output_stripe*output_stripe_size;
		    // We change the game: simply navigate to beginning of the partition (for simplicity!)
	    	base_result_ptr = result_tensor->local_data[tid];
	    	base_vector_ptr = vector->data;
		}
		    // printf("my part of result ::::::\n");
		    // print_to_console(base_result_ptr, 4);
	    double * result_ptr = base_result_ptr;
	    // printf("my part of result ::::::\n");
	    // print_to_console(result_ptr, 4);

	    const double * vector_ptr = base_vector_ptr;

		// printf("Thread %d beginning at Tensor stripe %d\n", tid, tid);
		// print_to_console(tensor_ptr, stripe_size);

	    int start_tid = tid;
	    if (mode != P) {
	    	// This is very special -- if we do not care about striping (i.e. we are in the case of mode other than P)
	    	// We can simply take any stripe, even stripe=0 or roll over (stripe=(tid%stripes))
	    	start_tid = 0;
	    }

		for (size_t stripe=start_tid; stripe<stripes+start_tid; ++stripe) {
			// printf("Stripe=%zu, stripes+tid=%d\n", stripe, stripes+start_tid);
			//////////////////////////////////////////////////////////////////////////////////

			// libxsmm_dmmfunction kernel;
			reset_array_sizet(block_counter, dim, (size_t) 0);
			reset_array_sizet(morton_block_indices, morton_block_levels, (size_t) 0);

			size_t global_tensor = 0;
			size_t global_result = 0;
			size_t global_vector = 0;
			size_t old_global_vector = 0;
			size_t mask;
			int level;
			size_t inc_game;
			size_t offset;
			int block_diff;
			double block_diff_log;
		    if (mode != dim-1) {
		  		kernel = libxsmm_dmmdispatch(right_size, 1, vector_size, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
		  	} else {
		  		kernel = libxsmm_dmmdispatch(1, result_size, vector_size, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
		  	}
			size_t el = 0;
			while (1) {
				// printf("Processing a single block!\n");
				if (mode != dim-1) {
					size_t next = 0;
					size_t next_result = 0;
					for (size_t i=0; i<left_mat_size; ++i) {
					    const double *const tensor_next = tensor_ptr + i*mat_size;
					    double *const result_next = result_ptr + i*right_size;
					    // printf("tensor el used:");
					    // print_to_console(tensor_next, mat_size);
					    // printf("result_next el used:");
					    // print_to_console(result_next, right_size);
					    // printf("vector el used:");
					    // print_to_console(vector_ptr, vector_size);
					    // kernel(tensor_next, vector_ptr, result_next, NULL, NULL, NULL);
						cblas_dgemv(
							CblasRowMajor, // const CBLAS_LAYOUT
							CblasTrans, // const CBLAS_TRANSPOSE
							vector_size, right_size,
							alpha, // const double
							tensor_next, right_size, // const double*, const MKL_size_t
							vector_ptr, incx, // const double*, const MKL_size_t
							beta, // const float
							result_next, incy); // const double*, const MKL_size_t
					    // printf("Done, result is:");
					    // print_to_console(result_next, right_size);
					    #if (TEST_ENV == 1 && COUNT_ENABLED == 1)
						#pragma omp critical
						{
					    for (size_t tu=0; tu<mat_size; ++tu) {
					    	int ru = tu % right_size;
					    	int vu = tu % vector_size;
					    	// if (mode == P) {
					    	// printf("mode=P, we access element %zu, %zu and %zu\n",
					    	// 	tensor_next + tu - (tensor->lin.master_data),
					    	// 	vector_ptr + vu + ((l*stripes)+tid)*tensor->layout2[mode] - (vector->local_data[((l*stripes)+tid)]),
					    	// 	result_next + ru - (result_tensor->data));
					    	// } else {
					    	// printf("mode!=P, we access element %zu, %zu and %zu\n",
					    	// 	tensor_next + tu - (tensor->lin.master_data),
					    	// 	vector_ptr + vu - (vector->data),
					    	// 	result_next + ru + tid*output_stripes*output_stripe_size - (result_tensor->local_data[tid]));
					    	// }
					    	// tensor_used[tensor_next + tu + tid*stripes*stripe_size - (tensor->lin.local_data[tid])] += 1; 
					    	tensor_used[tensor_next + tu - (tensor->lin.master_data)] += 1; 
					    	if (mode == P) {
					    		vector_used[vector_ptr + vu + ((l*stripes)+tid)*tensor->layout2[mode] - (vector->local_data[((l*stripes)+tid)])] += 1; 
					    		result_used[result_next + ru - (result_tensor->data)] += 1; 
					    	} else {
					    		vector_used[vector_ptr + vu - (vector->data)] += 1; 
					    		result_used[result_next + ru + tid*output_stripes*output_stripe_size - (result_tensor->local_data[tid])] += 1; 
					    	}
					    }
						}
					    #endif
					}
				} else {
				    // printf("tensor el used:");
				    // print_to_console(tensor_ptr, mat_size);
				    // printf("result_next el used:");
				    // print_to_console(result_ptr, tensor->layout2[mode]);
				    // printf("vector el used:");
				    // print_to_console(vector_ptr, vector_size);
					// kernel(vector_ptr, tensor_ptr, result_ptr, NULL, NULL, NULL);
					cblas_dgemv(
						CblasRowMajor, // const CBLAS_LAYOUT
						CblasNoTrans, // const CBLAS_TRANSPOSE
						result_size, vector_size, // const MKL_size_t (s)
						alpha, // const double
						tensor_ptr, vector_size, // const double*, const MKL_size_t
						vector_ptr, incx, // const double*, const MKL_size_t
						beta, // const float
						result_ptr, incy); // const double*, const MKL_size_t
				    // printf("Done, result is:");
				    // print_to_console(result_ptr, tensor->layout2[mode]);
					#if (TEST_ENV == 1 && COUNT_ENABLED == 1)
					#pragma omp critical
					{
				    for (size_t tu=0; tu<block_size; ++tu) {
				    	int ru = tu % result_size;
				    	int vu = tu % vector_size;
				    	// if (mode == P) {
				    	// printf("mode=P, we access element %zu, %zu and %zu\n",
				    	// 	tensor_ptr + tu - (tensor->lin.master_data),
				    	// 	vector_ptr + vu + ((l*stripes)+tid)*tensor->layout2[mode] - (vector->local_data[((l*stripes)+tid)]),
				    	// 	result_ptr + ru - (result_tensor->data));
				    	// } else {
				    	// printf("mode!=P, we access element %zu, %zu and %zu\n",
				    	// 	tensor_ptr + tu - (tensor->lin.master_data),
				    	// 	vector_ptr + vu - (vector->data),
				    	// 	result_ptr + ru + tid*output_stripes*output_stripe_size - (result_tensor->local_data[tid]));
				    	// }
				    	// tensor_used[tensor_ptr + tu + tid*stripes*stripe_size - (tensor->lin.local_data[tid])] += 1; 
				    	tensor_used[tensor_ptr + tu - (tensor->lin.master_data)] += 1; 
				    	if (mode == P) {
							vector_used[vector_ptr + vu + ((l*stripes)+tid)*tensor->layout2[mode] - (vector->local_data[(l*stripes)+tid])] += 1; 
					    	result_used[result_ptr + ru - (result_tensor->data)] += 1; 	    		
				    	} else {
				    		vector_used[vector_ptr + vu - (vector->data)] += 1; 
				    		result_used[result_ptr + ru + tid*output_stripes*output_stripe_size - (result_tensor->local_data[tid])] += 1; 
				    	}
				    	// vector_used[vector_ptr + vu - (vector->data)] += 1; 
				    	// result_used[result_ptr + ru - (result_tensor->data)] += 1; 
				    	// result_used[result_ptr + ru + tid*output_stripes*output_stripe_size - (result_tensor->local_data[tid])] += 1; 
				    }
					}
				    #endif
				}

				tensor_ptr += block_size;
				if (++el == blocks) { // TRICK: we must move the code to move the tensor to next block ABOVE (otherwise we do not actually MOVE it
					break;
				}
				// printf("more blocks?\n");
				old_global_vector = block_counter[mode];
				global_result += result_size;

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
				// printf("are we stuck aboev?\n");
				block_counter[offset] |= mask;
				if (offset == mode) {
					size_t temp = global_result;
					global_result = morton_block_indices[level];
					morton_block_indices[level] = temp;
					size_t block_diff_abs = block_counter_threshold[mode] - block_counter[mode];
					int block_diff_power = ceil(log2(block_diff_abs));
					if (block_diff_power < level) {
						level = block_diff_power;
					}
					for (int i=0; i<=level-1; ++i) {
						morton_block_indices[i] = global_result;
					}
				}
				result_ptr = base_result_ptr + global_result;
				// printf("We reset (within block algorithm) the result_ptr to base + %zu\n", global_result);
				global_vector = block_counter[mode] * tensor->block_layout[mode];
				vector_ptr = base_vector_ptr + global_vector;
			}

			// printf("WE ARE HERE!\n");

			// tensor_ptr = tensor->lin.data + ((stripe+1)%stripes)*stripe_size + tid*stripes*stripe_size;
			#if (TEST_ENV == 1)
			// printf("We work with global data, thread %d, stripe %d\n", tid, ((stripe+1)%stripes));
			tensor_ptr = tensor->lin.master_data + (l*stripes+tid)*stripes*stripe_size + ((stripe+1)%stripes)*stripe_size;
			// print_to_console(tensor_ptr, stripe_size);
			#else
			tensor_ptr = tensor->lin.local_data[(l*stripes+tid)] + ((stripe+1)%stripes)*stripe_size;
			#endif

			// printf("Thread %d moving to stripe %zu\n", tid, (stripe+1)%stripes);

			if (mode == P) {
				// printf("WE ENTER THE BARRIER (from thread %d)\n", tid);
				// No change -> we work with the result global result allocated using interleaved allocation (!)
				base_result_ptr = result_tensor->data + ((stripe+1)%output_stripes)*output_stripe_size;
				#pragma omp barrier
				// printf("sync point\n");

			} else {
				if ((stripe+1) % stripe_left == 0) {
					// printf("move left of mode because %zu mod %zu\n", stripe+1, stripe_left);
					if (stripe+1 == stripes) {
						// base_result_ptr = result_tensor->data + tid*output_stripes*output_stripe_size;
						base_result_ptr = result_tensor->local_data[tid];
						// printf("RES ROZKMINA move left of mode (!): reset the result!\n");
					} else {
						base_result_ptr += stripe_size/tensor->layout2[mode];
						// printf("RES ROZKMINA: move left of mode (!): increment\n");
					}
					base_vector_ptr = vector->data;
				} else if ((stripe+1) % stripe_mode == 0) {	
					// printf("move along of mode because %zu mod %zu\n", stripe+1, stripe_mode);
					if (stripe+1 == stripes) {
					} else {
						base_result_ptr += stripe_size/tensor->layout2[mode]; // The whole vector_size i.e. layout[mode]
					}
					base_result_ptr -= stripe_mul[mode]*(stripe_size/tensor->layout2[mode]);
					if (stripe+1 == stripes) {
						base_vector_ptr = vector->data;
					} else {
						base_vector_ptr += tensor->layout2[mode];
					}
				} else {
					// printf("move right of mode because %zu is neither div by %zu or %zu\n", stripe+1, stripe_left, stripe_mode); 
					if (stripe+1 == stripes) {
						// base_result_ptr = result_tensor->data + tid*output_stripes*output_stripe_size;
						base_result_ptr = result_tensor->local_data[tid];
						// printf("RES ROZKMINA move right of mode(!): reset the result!\n");
					} else {
						base_result_ptr += stripe_size/tensor->layout2[mode];
						// printf("RES ROZKMINA: move right of mode (!): increment\n");
					}
				}
			}

			result_ptr = base_result_ptr;
			vector_ptr = base_vector_ptr; 

		}
		free(morton_block_indices);
		free(block_counter);
		// } // For the pragma critical (DO NOT FORGET TO DISABLE THE BARRIER)
	}
	}
	free(mul);
	free(block_counter_threshold);

	#if (TEST_ENV == 1 && COUNT_ENABLED == 1)
	int result_count = result_used[0];
	int vector_count = vector_used[0];
	int tensor_count = tensor_used[0];
	printf("INFO: Elements of tensor/result/vector were used correspondingly %d, %d and %d times\n", tensor_count, result_count, vector_count);
	// First verification
	for (size_t i=1; i<tensor->lin.size; ++i) {
		if (tensor_used[i] != tensor_count) {
			printf("ERROR: We used one of the elements DIFFERENT NUMBER OF TIMES (tensor[%zu], %d times), (tensor[%d], %d times)\n",
				i, tensor_used[i], 0, tensor_count);
			exit(-1);
		}
	}
	for (size_t i=1; i<result_tensor->size; ++i) {
		if (result_used[i] != result_count) {
			printf("ERROR: We used one of the elements DIFFERENT NUMBER OF TIMES (result[%zu], %d times), (result[%d], %d times)\n",
				i, result_used[i], 0, result_count);
			exit(-1);
		} 
	}
	for (size_t i=1; i<vector->size; ++i) {
		if (vector_used[i] != vector_count) {
			printf("ERROR: We used one of the elements DIFFERENT NUMBER OF TIMES (vector[%zu], %d times), (vector[%d], %d times)\n",
				i, vector_used[i], 0, vector_count);
			exit(-1);
		} 
	}
	assert(tensor_count == 1);
	assert(result_count == (int) (tensor->lin.size / result_tensor->size));
	assert(vector_count == (int) (tensor->lin.size / vector->size));
	free(vector_used);
	free(result_used);
	free(tensor_used);
	#endif

	free(stripei);
	free(stripe_mul);

}

void
tvMortonMulticoreMkl(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

	// printf("tvMortonMulticoreMKL!\n");

	#pragma omp parallel
	{

	int my_rank = omp_get_thread_num();
	int world_size = omp_get_num_threads();
    int output_index = 0;

    // printf("Hello from the parallel region! I am thread %d in the world of %d threads!\n", my_rank, world_size);

	// initialize LIBXSMM
	// int prefetch = LIBXSMM_PREFETCH_AUTO;
	libxsmm_dmmfunction kernel;
	// libxsmm_xmmfunction  is just a weak type for both: dmm and smm
	
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
	// printf("blocks:%zu; output_blocks:%zu\n", blocks, blocks/block_counter_threshold[mode]);

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
	int level;
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

	size_t el = 0;
	int first_run = 0;

	while (1) {

		if (output_index == my_rank) {

			if (mode != dim-1) {
				next = 0;
				next_result = 0;
				for (size_t i=0; i<left_mat_size; ++i) {

				    const double *const tensor_next = tensor_ptr + i*mat_size;
				    double *const result_next = result_ptr + i*right_size;
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
		}

		if (++el == blocks) {
			break;
		}

		old_global_vector = block_counter[mode];
		global_result += result_size;
		tensor_ptr += block_size;

		int inc_count = -1;
		mask = 1;
		level = 0;
		inc_game = 1;
		offset = dim-1;
		while (inc_game) {
			inc_count++;
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
		// print_to_console_sizet(block_counter, dim);
		// printf("level=%zu\n", level);
		// printf("inc took place=%d\n", inc_count);

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
				for (int i=0; i<=block_diff-1; ++i) {
					morton_block_indices[i] = global_result;
				}
				}
			} else {
				if (level > 0) {
				for (int i=0; i<=level-1; ++i) {
					morton_block_indices[i] = global_result;
				}
				}
			}
		}

		result_ptr = base_result_ptr + global_result;
		output_index = (global_result/result_size) % world_size;

		// VECTOR HAS TO CHANGE???
		global_vector = block_counter[mode] * tensor->block_layout[mode];
		vector_ptr = base_vector_ptr + global_vector;

	}

	free(morton_block_indices);
	free(block_counter);
	free(block_counter_threshold);
	free(mul);

	}

}

void
tvMortonMulticore(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {
	
	// printf("tvMortonMulticore!\n");
	#pragma omp parallel
	{

	int my_rank = omp_get_thread_num();
	int world_size = omp_get_num_threads();
    int output_index = 0;

    // printf("Hello from the parallel region! I am thread %d in the world of %d threads!\n", my_rank, world_size);

	// initialize LIBXSMM
	// int prefetch = LIBXSMM_PREFETCH_AUTO;
	libxsmm_dmmfunction kernel;
	// libxsmm_xmmfunction  is just a weak type for both: dmm and smm
	
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
	// printf("blocks:%zu; output_blocks:%zu\n", blocks, blocks/block_counter_threshold[mode]);

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
	int level;
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

	size_t el = 0;
	while (1) {

		if (output_index == my_rank) {
			// printf("current_block=%d, I am thread %d, and I work on output block %d!\n", el, my_rank, output_index);
			if (mode != dim-1) {
				next = 0;
				next_result = 0;
				for (size_t i=0; i<left_mat_size; ++i) {
				    const double *const tensor_next = tensor_ptr + i*mat_size;
				    double *const result_next = result_ptr + i*right_size;

				    kernel(tensor_next, vector_ptr, result_next);//, NULL, NULL, NULL);

					/** Batched matrix multiplications (explicit data representation). */
					// libxsmm_mmbatch(kernel, 0, 0, 
					//   0, 0, 0,
					//   tensor_next, vector_ptr, result_next,
					//   // const void* a, const void* b, void* c,
					//   1,
					//   0, 1);
				}
			} else {
				kernel(vector_ptr, tensor_ptr, result_ptr);//, NULL, NULL, NULL);
			}
		}

		if (++el == blocks) {
			break;
		}

		old_global_vector = block_counter[mode];
		global_result += result_size;
		tensor_ptr += block_size;

		int inc_count = -1;
		mask = 1;
		level = 0;
		inc_game = 1;
		offset = dim-1;
		while (inc_game) {
			inc_count++;
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
		// print_to_console_sizet(block_counter, dim);
		// printf("level=%zu\n", level);
		// printf("inc took place=%d\n", inc_count);

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
				for (int i=0; i<=block_diff-1; ++i) {
					morton_block_indices[i] = global_result;
				}
				}
			} else {
				if (level > 0) {
				for (int i=0; i<=level-1; ++i) {
					morton_block_indices[i] = global_result;
				}
				}
			}
		}

		result_ptr = base_result_ptr + global_result;
		output_index = (global_result/result_size) % world_size;

		// VECTOR HAS TO CHANGE???
		global_vector = block_counter[mode] * tensor->block_layout[mode];
		vector_ptr = base_vector_ptr + global_vector;

	}

	free(morton_block_indices);
	free(block_counter);
	free(block_counter_threshold);
	free(mul);

	}

}

void
block_morton_block_unfold(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

	int unblock = 0;
	size_t next = 0;

	size_t last_known_counter = 0; // FIXING FOR P

	size_t stripe_offset = 0;

	// For now: shitty solution with modifying the structures (!) they were const before
	const size_t dim = tensor->dim;
	size_t * stripe_counter = calloc(dim, sizeof(size_t));
	size_t * const stripe_mul = malloc(dim * sizeof(size_t));
	size_t * const stripe_real_mul = malloc(dim * sizeof(size_t));
	size_t * const stripei = malloc(dim * sizeof(size_t));
	size_t * const stripep = malloc(dim * sizeof(size_t));
	stripe_mul[dim-1] = 1;
	stripe_real_mul[dim-1] = 1;
	size_t stripe_size = 1;
	size_t stripes = 1;
	size_t stripe_mode = 1;
	size_t stripe_left = 1;
	// Wow this is important, that this look is constructed by going downwards (because of the way I compute the stripe left and mode!)
	for (size_t i=dim-1; i<dim; --i) {
		stripe_size *= tensor->layout2[i];
		stripei[i] = tensor->layout[i] / tensor->layout2[i];
		
		if (i == mode) { // FIXING FOR P
			stripep[i] = 1;
		} else {
			stripep[i] = tensor->layout[i] / tensor->layout2[i];
		}

		stripes *= stripei[i]; // TRICK: make it think that its not striping over mode 0
		if (i > mode) {
			stripe_mode *= stripei[i];
		} else if (i == mode) {
			stripe_left = stripe_mode * stripei[i];
		}
		if (i!=0) {
			stripe_mul[i-1] = stripe_mul[i] * stripei[i];
			stripe_real_mul[i-1] = stripe_real_mul[i] * stripei[i] * tensor->layout2[i];
		}
	}

	// FIXING FOR P (tag to search for when loking for important changes)
	// printf("CHECKPOINT!============================================================ %zu %zu\n", tensor->layout[mode], tensor->layout2[mode]);
	stripes = stripes / (tensor->layout[mode] / tensor->layout2[mode]);

	size_t output_stripes;
	if (mode == 0) {
		output_stripes = stripes;
	} else {
		// printf("before div1\n");
		output_stripes = stripes / (tensor->layout[mode] / tensor->layout2[mode]);
		// printf("after div1\n");
	}
	size_t output_stripe_size = stripe_size / tensor->layout2[mode];
	size_t vector_stripes = stripes / output_stripes;
	// printf("after div2\n");
	size_t * const block_counter_threshold = malloc(dim * sizeof(size_t));
	size_t * const mul = malloc(dim * sizeof(size_t));
	size_t * const tensor_mul = malloc(dim * sizeof(size_t));

	size_t mul_mode = 1;
	size_t mul_left = 1;
	size_t right_size = 1;
	size_t block_size = 1;
	size_t blocks = 1;
	size_t max_block = 0;
	mul[dim-1] = 1;
	tensor_mul[dim-1] = 1;
	for (size_t i=dim-1; i<dim; --i) {
		size_t temp = tensor->block_layout[tensor->layout_perm[i]];
		// printf("temp=%zu\n", temp);
		if (i > mode) {
			right_size *= temp;
		}
		block_size *= temp;
		block_counter_threshold[i] = (tensor->layout2[i] + temp -1) / temp;
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
			tensor_mul[i-1] = tensor_mul[i] * tensor->layout[i];
		}
	}
	const size_t vector_size = tensor->block_layout[mode];
	const size_t mat_size = right_size * vector_size;
	// Commented this out as for 1D it is 0 so we have a weird error
	// printf("before div 3, vector_size and mat_size = %zu and %zu\n", vector_size, mat_size);
	// const size_t result_size = block_size / vector_size;
	// const size_t left_mat_size = block_size / mat_size;
	const size_t morton_block_levels = log2(max_block)+1; // round-up (take front size_teger and add 1)

	// printf("Kolejno, mul, tensor_mul, stripe_mul, stripei, layout2, stripe_real_mul:");
	// print_to_console_sizet(mul, dim);
	// print_to_console_sizet(tensor_mul, dim);
	// print_to_console_sizet(stripe_mul, dim);
	// print_to_console_sizet(stripei, dim);
	// print_to_console_sizet(tensor->layout2, dim);
	// print_to_console_sizet(stripe_real_mul, dim);
	// printf("Number of parts p=%zu\n", stripes);

	size_t blockblock_base = 0;
	size_t * outer_counter = calloc(dim, sizeof(size_t));

	int nthreads;
	#pragma omp parallel
	{
		nthreads = omp_get_num_threads();
	}
	if (nthreads > (int) stripei[0]) nthreads = stripei[0];
	// printf("We limit number of threads to max(threads=%d, stripes_at_mode_0=%d)\n", nthreads, stripei[0]);
	// check if p < p_sync in which case we can use at most p processors (and the rest is idle!)
	// same thing happens if we are not divisible by p, right?

	/////////////// FUTURE: Make a single loop over these (!) rather than like right now (i.e. parallel loop over a loop) i.e. loop going over ALL stripes in the tensor!
	// For now make this parallel critical and scheduled

	size_t absolute = 0;
	size_t * const block_counter = calloc(dim, sizeof(size_t));

	for (int tid=0; tid<(int) stripei[mode]; ++tid) {

		// printf("tid=%d (moving on the P dimension)\n", tid);
// 
		// #pragma omp parallel
		// {
		// int tid = omp_get_thread_num();
		// int nthreads = omp_get_num_threads();
		// #pragma omp for ordered schedule(static,1)
		// for (int t=0; t<nthreads; ++t)
		// {
		// assert( t==tid );
		// #pragma omp ordered
		// {

		// printf("Thread %d beginning at Tensor stripe %d\n", tid, tid);

	    const double * tensor_ptr = tensor->lin.data + tid*stripes*stripe_size + tid*stripe_size; // Navigate to partition, stripe
	    // const double * tensor_ptr = tensor->lin.local_data[tid] + tid*stripe_size;
	    // printf("printing these elements  of tensor:\n");
		// print_to_console_double(tensor_ptr, 10);
// 
		double * base_result_ptr = result_tensor->data + tid*output_stripe_size; // TRICK: do not navigate toward partition, just the stripe(!) (for mode=0)
		// printf("Base result pointer, printint elements:\n");
		// print_to_console_double(base_result_ptr, 10);

	    // const double * base_vector_ptr = vector->data + tid*tensor->layout2[mode];
	    const double * base_vector_ptr = vector->local_data[tid];
	    // printf("vector:\n");
	    // print_to_console_double(base_vector_ptr, 10);

	    if (mode != 0) {
	    	size_t correct_output_stripe = 0;
			for (size_t i=dim-1; i<dim; --i) {
				if (i < mode) { // You only divide by mode if it's greater than mode (!)
					correct_output_stripe += ((tid / stripe_mul[i]) % stripei[i]) * (stripe_mul[i]/stripei[mode]);
				} else if (i == mode) {
					continue;
				} else {
					correct_output_stripe += ((tid / stripe_mul[i]) % stripei[i]) * stripe_mul[i];
				}
	    	}
		    base_vector_ptr = vector->data + ((tid/stripe_mul[mode])%stripei[mode])*tensor->layout2[mode];
		    // base_result_ptr = result_tensor->data + tid*output_stripes*output_stripe_size + correct_output_stripe*output_stripe_size;
		    base_result_ptr = result_tensor->local_data[tid] + correct_output_stripe*output_stripe_size;
		} else {
		}

	    double * result_ptr = base_result_ptr;
	    const double * vector_ptr = base_vector_ptr;

	    size_t block_offset = 0;
	    size_t morton_offset = 0;
	    size_t * counter = calloc(dim, sizeof(size_t));

	    for(size_t stripe=0; stripe<stripes; ++stripe) {
		// for (size_t stripe=tid; stripe<stripes+tid; ++stripe) {

			// printf("stripe=%zu\n", stripe);
			// printf("stripe_counter="); print_to_console_sizet(stripe_counter, dim);

			//////////////////////////////////////////////////////////////////////////////////

			libxsmm_dmmfunction kernel;
			// IMPROVE: We do not need to allocate for each stripe (!)
			size_t * const morton_block_indices = calloc(morton_block_levels, sizeof(size_t));
			size_t global_tensor = 0;
			size_t global_result = 0;
			size_t global_vector = 0;
			size_t old_global_vector = 0;
			size_t mask;
			int level;
			size_t inc_game;
			size_t offset;
			int block_diff;
			double block_diff_log;
		   //  if (mode != dim-1) {
		  	// 	kernel = libxsmm_dmmdispatch(right_size, 1, vector_size, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
		  	// } else {
		  	// 	kernel = libxsmm_dmmdispatch(1, result_size, vector_size, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
		  	// }
			size_t el = 0;

			reset_array_sizet(block_counter, dim, (size_t) 0);

			morton_offset = 0;
			while (1) {

				// printf("block=%zu\n", el);
				// printf("block_counter="); print_to_console_sizet(block_counter, dim);

				// Go over the block
				reset_array_sizet(counter, dim, (size_t) 0);

				for (size_t t=0; t<block_size; ++t) {
					// printf("	element=%zu refers to original element stripe_offset(%zu) + morton_offset(%zu) + block_offset(%zu) = %zu\n"
					// 	, absolute, stripe_offset, morton_offset, block_offset, stripe_offset+morton_offset+block_offset);
					// printf("	element_counter=");
					// print_to_console_sizet(counter, dim);

					// if it is test_scenario, and it always is...
					// then we block the input tensor, but unblock the 
					if (unblock == 0) {
						// output tensor is 
						// printf("element %zu = element %zu\n", next, stripe_offset+morton_offset+block_offset);
						result_tensor->master_data[next++] = tensor->lin.data[stripe_offset+morton_offset+block_offset];
					} else {
						result_tensor->data[stripe_offset+morton_offset+block_offset] = tensor->lin.data[next++];
						// out_tensor->data[stripe_offset+morton_offset+block_offset] = tensor->lin.data[next++];
					}

					block_offset = 0;
					// should be protected 
					++counter[dim-1];
					for (size_t d=dim-1; d!=0; --d) {
						// two conditions to tick the counter
						// 1) it simply reaches the threshold
						// 2) it reaches the limit for this part dimension
						if (counter[d] == tensor->block_layout[d]) {
							if (d!=0) {
								++counter[d-1];
							}
							counter[d] = 0;
						}
						// block_offset += counter[d] * mul[d] * stripe_mul[d];
						block_offset += counter[d] * tensor_mul[d];
						// printf("	block_offset += counter(%zu) * tensor_mul(%zu) = %zu\n", 
						// 	counter[d], tensor_mul[d], counter[d]*tensor_mul[d]);
					}
					// handle the 0 case
					if (counter[0] == tensor->block_layout[0]) {
						break;
					}
					// block_offset += counter[0] * mul[0] * stripe_mul[0];
					block_offset += counter[0] * tensor_mul[0];
					// printf("	block_offset += counter(%zu) * tensor_mul(%zu) = %zu\n", 
					// 	counter[0], tensor_mul[0], counter[0]*tensor_mul[0]);
					++absolute;
				}

				// printf("Processing a single block!\n");
				// if (mode != dim-1) {
				// 	size_t next = 0;
				// 	size_t next_result = 0;
				// 	for (size_t i=0; i<left_mat_size; ++i) {
				// 	    const double *const tensor_next = tensor_ptr + i*mat_size;
				// 	    double *const result_next = result_ptr + i*right_size;
				// 	    kernel(tensor_next, vector_ptr, result_next, NULL, NULL, NULL);
				// 	}
				// } else {
				// 	kernel(vector_ptr, tensor_ptr, result_ptr, NULL, NULL, NULL);
				// }

				tensor_ptr += block_size;
				if (++el == blocks) { // TRICK: we must move the code to move the tensor to next block ABOVE (otherwise we do not actually MOVE it
					break;
				}
				old_global_vector = block_counter[mode];
				// global_result += result_size;

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

				morton_offset = 0;
				// Using counters, we calculate the actual position (i.e. where we are in the morton-curve)
				// printf("===== WE ARE HERE ======== (the hit is %d)\n", dim-1<=dim-1);
				for (size_t d=dim-1; d<=dim-1; --d) {
					// printf("morton weird summation: d=%zu\n", d);
					morton_offset += block_counter[d] * tensor_mul[d] * tensor->block_layout[d];
					// printf("	morton_offset += %zu * %zu * %zu = %zu\n",
					// 	block_counter[d], tensor_mul[d], tensor->block_layout[d]);
				}

				if (offset == mode) {
					size_t temp = global_result;
					global_result = morton_block_indices[level];
					morton_block_indices[level] = temp;
					size_t block_diff_abs = block_counter_threshold[mode] - block_counter[mode];
					int block_diff_power = ceil(log2(block_diff_abs));
					if (block_diff_power < level) {
						level = block_diff_power;
					}
					for (int i=0; i<=level-1; ++i) {
						morton_block_indices[i] = global_result;
					}
				}
				result_ptr = base_result_ptr + global_result;
				global_vector = block_counter[mode] * tensor->block_layout[mode];
				vector_ptr = base_vector_ptr + global_vector;
			}

			free(morton_block_indices);
			// free(block_counter);

			tensor_ptr = tensor->lin.data + ((stripe+1)%stripes)*stripe_size + tid*stripes*stripe_size;
			// tensor_ptr = tensor->lin.local_data[tid] + ((stripe+1)%stripes)*stripe_size;
			// printf("moving to next stripe in mode P which is stripe no %zu\n", tid);

			if (mode == 0) {
				// No change -> we work with the result global result allocated using interleaved allocation (!)
				base_result_ptr = result_tensor->data + ((stripe+1)%output_stripes)*output_stripe_size;
				// #pragma omp barrier

			} else {
				if ((stripe+1) % stripe_left == 0) {
					if (stripe+1 == stripes) {
						// base_result_ptr = result_tensor->data + tid*output_stripes*output_stripe_size;
						base_result_ptr = result_tensor->local_data[tid];
					} else {
						base_result_ptr += stripe_size/tensor->layout2[mode];
					}
					base_vector_ptr = vector->data;
				} else if ((stripe+1) % stripe_mode == 0) {	
					if (stripe+1 == stripes) {
					} else {
						base_result_ptr += stripe_size/tensor->layout2[mode]; // The whole vector_size i.e. layout[mode]
					}
					base_result_ptr -= stripe_mul[mode]*(stripe_size/tensor->layout2[mode]);
					if (stripe+1 == stripes) {
						base_vector_ptr = vector->data;
					} else {
						base_vector_ptr += tensor->layout2[mode];
					}
				} else {
					if (stripe+1 == stripes) {
						// base_result_ptr = result_tensor->data + tid*output_stripes*output_stripe_size;
						base_result_ptr = result_tensor->local_data[tid];
					} else {
						base_result_ptr += stripe_size/tensor->layout2[mode];
					}
				}
			}

			result_ptr = base_result_ptr;
			vector_ptr = base_vector_ptr;

			// printf("stripe_limits=");
			// print_to_console_sizet(stripei, dim);
			// stripe_offset is calcualted knowing that moving a stripe implies moving many blocks (stripe_i times stripe_layout_aka_length)
			++stripe_counter[dim-1];
			stripe_offset = 0;
			// printf("stripe_oredr:");
			// print_to_console_sizet(stripe_counter, dim);
			for (size_t d=dim-1; d!=0; --d) {
				// printf("d=%zu\n", d);

				if (d == mode) {
					if (stripe_counter[d] == (tid+1)*stripep[d]) {
						// printf("We reached threshold for mode %zu\n", d);
						if (d!=0) {
							// printf("increment counter %zu\n", d-1);
							++stripe_counter[d-1];
						} else {
							++stripe_counter[mode];
						}
					}
					stripe_counter[d] = last_known_counter;
				} else {
					if ((stripe_counter[d] == stripep[d])) { // FIXING FOR P
						// printf("We reached threshold for mode %zu\n", d);
						if (d!=0) {
							// printf("increment counter %zu\n", d-1);
							++stripe_counter[d-1];
						} else {
							++stripe_counter[mode];
						}
						stripe_counter[d] = 0;
					}
				}

				// block_offset += stripe_counter[d] * mul[d] * stripe_mul[d];
				// stripe_offset += stripe_counter[d] * stripe_real_mul[d]  * tensor->layout2[d];
				// printf("	stripe_offset += %zu * %zu *%zu = %zu\n", 
				// 	stripe_counter[d], stripe_mul[d], tensor->layout2[d], stripe_counter[d]*stripe_mul[d]*tensor->layout2[d]);
				// printf("new stripe_offset is %zu\n", stripe_offset);
			}

			// printf("stripe counters 1:");
			// print_to_console_sizet(stripe_counter, dim);

			// Add the 0 case (!)
			if (stripe_counter[0] == stripep[0]) { // FIXING FOR P
				// printf("we reached max on 0 mode, so we must increment P mode (Unless 0 is P  in which point... what?\n");
				// if (mode == 0) {
				// 	last_known_counter = stripe_counter[0];
				// } else {
				if (mode != 0) {
					stripe_counter[0] = 0;
					++stripe_counter[mode];
					last_known_counter = stripe_counter[mode];		
					// stripe_offset += stripe_counter[mode] * stripe_real_mul[mode]  * tensor->layout2[mode];
					// printf("stripe_offset increased by %zu times %zu times %zu = %zu\n", stripe_counter[mode], stripe_real_mul[mode], tensor->layout2[mode], stripe_counter[mode] * stripe_real_mul[mode]  * tensor->layout2[mode]);
					// printf("new stripe_offset is%zu\n", stripe_offset);
				} else {

				}
				// }
			}
			// stripe_offset += stripe_counter[0] * stripe_real_mul[0]  * tensor->layout2[0];

			// printf("stripe counters 2:");
			// print_to_console_sizet(stripe_counter, dim);
			
			// FIX FOR P: do the actual computation at the end because shit may happen above
			for (size_t d=0;d<dim;++d) {
				stripe_offset += stripe_counter[d] * stripe_real_mul[d]  * tensor->layout2[d];
				// printf("	stripe_offset += %zu * %zu *%zu = %zu\n", 
				// 	stripe_counter[d], stripe_mul[d], tensor->layout2[d], stripe_counter[d]*stripe_mul[d]*tensor->layout2[d]);
				// printf("new stripe_offset is %zu\n", stripe_offset);
			}

			if (stripe_counter[mode] == stripei[mode]) {
				// printf("THIS HAS BEEN THE LAST STRIPE; OVERFLOPW\n");
				break;
			}

		}
		// printf("finished with this\n");
		free(counter);

	}
	
	free(block_counter);

	free(block_counter_threshold);
	free(mul);
	free(outer_counter);
	free(tensor_mul);
	free(stripe_counter);
	free(stripe_real_mul);
	free(stripep);
	free(stripei);
	free(stripe_mul);
}


// void
// tvm_ppower_sync(struct tensor_storage * restrict tensor, struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

// 	#if (TEST_ENV == 1)
// 	// Array of use-counts for each object
// 	int * const tensor_used = calloc(tensor->lin.size, sizeof(int));
// 	int * const result_used = calloc(result_tensor->size, sizeof(int));
// 	int * const vector_used = calloc(vector->size, sizeof(int));
// 	#endif

// 	// For now: shitty solution with modifying the structures (!) they were const before
// 	const size_t dim = tensor->dim;

// 	size_t * const stripe_mul = malloc(dim * sizeof(size_t));
// 	size_t * const stripei = malloc(dim * sizeof(size_t));

// 	stripe_mul[dim-1] = 1;

// 	size_t stripe_size = 1;
// 	size_t stripes = 1;

// 	size_t stripe_mode = 1;
// 	size_t stripe_left = 1;

// 	// Wow this is important, that this look is constructed by going downwards (because of the way I compute the stripe left and mode!)
// 	for (size_t i=dim-1; i<dim; --i) {
// 		stripe_size *= tensor->layout2[i];
// 		stripei[i] = tensor->layout[i] / tensor->layout2[i];
// 		// printf("%zu div by %zu\n", tensor->layout[i], tensor->layout2[i]);
// 		stripes *= stripei[i]; // TRICK: make it think that its not striping over mode 0
// 		if (i > mode) {
// 			stripe_mode *= stripei[i];
// 		} else if (i == mode) {
// 			stripe_left = stripe_mode * stripei[i];
// 			// printf("We multiplied, stripe_left = %zu times (%zu / %zu)\n", stripe_mode, tensor->layout[i], tensor->layout2[i]);
// 		}
// 		if (i!=0) {
// 			stripe_mul[i-1] = stripe_mul[i] * stripei[i];
// 		}
// 	}
	
// 	// printf("UWAGAAAAAAA: stripe-Left=%zu, stripe-mode=%zu\n", stripe_left, stripe_mode);

// 	// Output stripes is actually divided by mode, so for mode=0 output_stripes = stripes 
// 	stripes = stripes / (tensor->layout[0] / tensor->layout2[0]);

// 	// Careful: output stripe means output stripes in a single mode0 partition(!!!!)
// 	// Therfore, we first divide the stripes (line above) and only THEN compute number of output stripes
// 	size_t output_stripes;
// 	if (mode == 0) {
// 		output_stripes = stripes;
// 	} else {
// 		output_stripes = stripes / (tensor->layout[mode] / tensor->layout2[mode]);
// 	}

// 	// CAREFUL: for output of mode=0 we will unnecessarily divide twice (!) 

// 	size_t output_stripe_size = stripe_size / tensor->layout2[mode];

// 	size_t vector_stripes = stripes / output_stripes;

// 	// the following is block code (!) -> must be based off information in the stripe NOT the whole tensor: TRICK 
// 	size_t * const block_counter_threshold = malloc(dim * sizeof(size_t));
// 	size_t * const mul = malloc(dim * sizeof(size_t));

// 	size_t mul_mode = 1;
// 	size_t mul_left = 1;
// 	size_t right_size = 1;
// 	size_t block_size = 1;
// 	size_t blocks = 1;
// 	size_t max_block = 0;
// 	mul[dim-1] = 1;
// 	for (size_t i=dim-1; i<dim; --i) {
// 		size_t temp = tensor->block_layout[tensor->layout_perm[i]];
// 		if (i > mode) {
// 			right_size *= temp;
// 		}
// 		block_size *= temp;
// 		block_counter_threshold[i] = (tensor->layout2[i] + temp -1) / temp;
// 		blocks *= block_counter_threshold[i];
// 		if (block_counter_threshold[i] > max_block) {
// 			max_block = block_counter_threshold[i];
// 		}
// 		if (i > mode) {
// 			mul_mode *= block_counter_threshold[i];
// 		} else if (i == mode) {
// 			mul_left = mul_mode * block_counter_threshold[i];
// 		}
// 		if (i!=0) {
// 			mul[i-1] = mul[i] * block_counter_threshold[i];
// 		}
// 	}
// 	// printf("Block counter thresholds:\n");
// 	// print_to_console_sizet(block_counter_threshold, dim);
// 	// printf("Mul:\n");
// 	// print_to_console_sizet(mul, dim);

// 	const size_t vector_size = tensor->block_layout[mode];
// 	const size_t result_size = block_size / vector_size;
// 	const size_t mat_size = right_size * vector_size;
// 	const size_t left_mat_size = block_size / mat_size;
// 	const size_t morton_block_levels = log2(max_block)+1; // round-up (take front size_teger and add 1)

// 	// printf("FULL VECTOR ELEMENTS:\n");
// 	// print_to_console(vector->data, vector->size);

// 	int nthreads;
// 	#pragma omp parallel
// 	{
// 		nthreads = omp_get_num_threads();
// 	}
// 	int limit_nthreads = nthreads;
	
// 	if (stripei[0] < (size_t) nthreads) {
// 		limit_nthreads = stripei[0];
// 		// printf("We limit to fit [%d] threads, as tensor has [%zu] partitions along mode 0.\n", nthreads, stripei[0]);
// 	} else {
// 		// printf("We assume the number of cores equals the number of partitions ([%d] threads, [%zu] partitions along mode 0).\n", nthreads, stripei[0]);
// 	}

// 	////// Start the parallel region
// 	#pragma omp parallel num_threads(limit_nthreads)
// 	{
// 		int tid = omp_get_thread_num();
// 		#pragma omp critical 
// 		{

// 		// Idea: feed tensor and result_tensor as before, but maybe modify their memory indices;
// 		// Set to correct starting points
	    
// 	    const double * tensor_ptr = tensor->lin.data + tid*stripes*stripe_size + tid*stripe_size; // Navigate to partition, stripe
// 	    double * base_result_ptr = result_tensor->data + tid*output_stripe_size; // TRICK: do not navigate toward partition, just the stripe(!) (for mode=0)
// 	    const double * base_vector_ptr = vector->data + tid*tensor->layout2[mode];

// 	    if (mode != 0) {

// 			// printf("STRIPEI:\n");
// 			// print_to_console_sizet(stripei, tensor->dim);
// 			// printf("STRIPE_MUL: \n");
// 			// print_to_console_sizet(stripe_mul, tensor->dim);

// 	    	size_t correct_output_stripe = 0;
// 			for (size_t i=dim-1; i<dim; --i) {
// 				// printf("current_mode = %zu (mode=%zu)\n", i, mode);
// 				if (i < mode) { // You only divide by mode if it's greater than mode (!)
// 					correct_output_stripe += ((tid / stripe_mul[i]) % stripei[i]) * (stripe_mul[i]/stripei[mode]);
// 				// printf("+= (%zu / %zu) mod %zu) * (%zu / %zu) (so we inc by %zu)\n", tid, stripe_mul[i], stripei[i], stripe_mul[i], stripei[mode],
// 					// ((tid / stripe_mul[i]) % stripei[i]) * (stripe_mul[i]/stripei[mode]));
// 				// printf("correct_output_stripe = %zu\n", correct_output_stripe);
// 				} else if (i == mode) {
// 					continue;
// 				} else {
// 					correct_output_stripe += ((tid / stripe_mul[i]) % stripei[i]) * stripe_mul[i];
// 					// printf("integer division (tid div stripe_mul)= %d, that modulo stripei is %d\n", tid/(int)stripe_mul[i], (tid/(int)stripe_mul[i])%stripei[i]);;
// 					// printf("+= (%zu / %zu) mod %zu) * (%zu) (so we inc by %zu)\n", tid, stripe_mul[i], stripei[i], stripe_mul[i]),
// 					// (((tid / stripe_mul[i]) % stripei[i]) * stripe_mul[i]);
// 					// printf("correct_output_stripe = %zu\n", correct_output_stripe);
// 				}
// 	    	}
// 	    	// printf("================== Correct output stripe is %zu for tid %d\n", correct_output_stripe, tid);

// 	    	// we must set the vector to go to the right stirpe
// 	    	// AT this point, stripe = tid
// 		    // for modes other than 0 we navigate to the right stripe, while there is no distribution(!)
// 		    // but in our case number of vector_stripes is exactly p, because our distribution is m=p
// 		    base_vector_ptr = vector->data + ((tid/stripe_mul[mode])%stripei[mode])*tensor->layout2[mode];
// 		    // we need to divide here because multiple (numbers next to each other) map to one (!) for instance 0,1->0, 2,3->1 
// 		    // normally we can modulo because series maps to a series (case for output?)
// 		   	// okay this is more tricky.... we have the following situation:
// 		   	// for intance, we have STRIPE Ai,j and we have the absolute position (tid) -> AND we want to get the actual i, j depending on the mode (!)
// 		   	// this is something that I did not need before (!)


// 		    // We now work on disjoint outputs, hence, we navigate only otwards the right partition

// 		    // For example, we may need to navigate to the second stripe of the second partition, wzorek?
// 		    // partition decided by core (out of p cores), but okay we need to add the stripe(!)
// 		    // stripe = (tid/output_stripes)*output_stripe_size
// 		    base_result_ptr = result_tensor->data + tid*output_stripes*output_stripe_size + correct_output_stripe*output_stripe_size;
// 			// printf("\n\n(core %d) We start at offset %zu of vector, %zu of result\n", 
// 			// 	tid,
// 			// 	((tid/stripe_mul[mode])%stripei[mode])*tensor->layout2[mode],
// 			// 	tid*output_stripes*output_stripe_size + correct_output_stripe*output_stripe_size);
// 		} else {
// 			// printf("\n\n(core %d) We start at offset %zu of vector, %zu of result (mode 0)\n",
// 			// 	tid, 
// 			// 	tid*tensor->layout2[mode], 
// 			// 	tid*output_stripe_size);
// 		}

// 	    double * result_ptr = base_result_ptr;
// 	    const double * vector_ptr = base_vector_ptr;

// 		for (size_t stripe=tid; stripe<stripes+tid; ++stripe) {

// 			// printf("(core %d) stripe=%d\n", tid, stripe);
// 		 //    printf("Tensor data in the stripe:\n");
// 		 //    print_to_console(tensor_ptr, stripe_size);
// 		 //    printf("Result data in the stripe:\n");
// 		 //    print_to_console(result_ptr, output_stripe_size);
// 		 //    printf("Vector data in the stripe:\n");
// 		 //    print_to_console(vector_ptr, tensor->layout2[mode]);

// 		 //    printf("TOTAL RESULTS IN THE TENSOR OUTPUT:\n");
// 		 //    print_to_console(result_tensor->data, result_tensor->size);
// // 
// 			//////////////////////////////////////////////////////////////////////////////////

// 			libxsmm_dmmfunction kernel;
// 			size_t * const block_counter = calloc(dim, sizeof(size_t));
// 			size_t * const morton_block_indices = calloc(morton_block_levels, sizeof(size_t));
// 			size_t global_tensor = 0;
// 			size_t global_result = 0;
// 			size_t global_vector = 0;
// 			size_t old_global_vector = 0;
// 			size_t mask;
// 			int level;
// 			size_t inc_game;
// 			size_t offset;
// 			int block_diff;
// 			double block_diff_log;
// 		    if (mode != dim-1) {
// 		  		kernel = libxsmm_dmmdispatch(right_size, 1, vector_size, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
// 		  	} else {
// 		  		kernel = libxsmm_dmmdispatch(1, result_size, vector_size, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
// 		  	}
// 			size_t el = 0;
// 			while (1) {

// 				// printf("Processing a single block!\n");

// 				if (mode != dim-1) {
// 					size_t next = 0;
// 					size_t next_result = 0;
// 					for (size_t i=0; i<left_mat_size; ++i) {
// 					    const double *const tensor_next = tensor_ptr + i*mat_size;
// 					    double *const result_next = result_ptr + i*right_size;
// 					    kernel(tensor_next, vector_ptr, result_next, NULL, NULL, NULL);

// 					    #if (TEST_ENV == 1)
// 					    for (size_t tu=0; tu<mat_size; ++tu) {
// 					    	int ru = tu % right_size;
// 					    	int vu = tu % vector_size;
// 					    	tensor_used[tensor_next + tu - (tensor->lin.data)] += 1; 
// 					    	vector_used[vector_ptr + vu - (vector->data)] += 1; 
// 					    	result_used[result_next + ru - (result_tensor->data)] += 1; 
// 					    	// printf("tensor[%zu], vector[%zu], result[%zu]\n", 
// 					    	// 	tensor_next - (tensor->lin.data),
// 					    	// 	vector_ptr - (vector->data),
// 					    	// 	result_next - (result_tensor->data));
// 					    }
// 					    #endif

// 					}
// 				} else {
// 					kernel(vector_ptr, tensor_ptr, result_ptr, NULL, NULL, NULL);

// 					#if (TEST_ENV == 1)
// 				    for (size_t tu=0; tu<block_size; ++tu) {
// 				    	int ru = tu % result_size;
// 				    	int vu = tu % vector_size;
// 				    	tensor_used[tensor_ptr + tu - (tensor->lin.data)] += 1; 
// 				    	vector_used[vector_ptr + vu - (vector->data)] += 1; 
// 				    	result_used[result_ptr + ru - (result_tensor->data)] += 1; 
// 				    	// printf("tensor[%zu], vector[%zu], result[%zu]\n", 
// 				    	// 	tensor_ptr - (tensor->lin.data),
// 				    	// 	vector_ptr - (vector->data),
// 				    	// 	result_ptr - (result_tensor->data));
// 				    }
// 				    #endif

// 				}

// 				tensor_ptr += block_size;
// 				if (++el == blocks) { // TRICK: we must move the code to move the tensor to next block ABOVE (otherwise we do not actually MOVE it
// 					break;
// 				}
// 				old_global_vector = block_counter[mode];
// 				global_result += result_size;

// 				mask = 1;
// 				level = 0;
// 				inc_game = 1;
// 				offset = dim-1;
// 				while (inc_game) {
// 					if (block_counter[offset] & mask) {
// 						block_counter[offset] &= ~mask;
// 						if (offset == 0) {
// 							mask <<= 1;
// 							level += 1;
// 							offset = dim-1;
// 						} else {
// 							offset -= 1;
// 						}
// 					} else {
// 						if ((block_counter[offset] | mask) >= block_counter_threshold[offset]) {
// 							if (offset == 0) {
// 								mask <<= 1;
// 								level += 1;
// 								offset = dim-1;
// 							} else {
// 								offset -= 1;
// 							}
// 						} else {
// 							inc_game = 0;
// 						}
// 					}
// 				}
// 				block_counter[offset] |= mask;
// 				if (offset == mode) {
// 					size_t temp = global_result;
// 					global_result = morton_block_indices[level];
// 					morton_block_indices[level] = temp;
// 					size_t block_diff_abs = block_counter_threshold[mode] - block_counter[mode];
// 					int block_diff_power = ceil(log2(block_diff_abs));
// 					if (block_diff_power < level) {
// 						level = block_diff_power;
// 					}
// 					for (int i=0; i<=level-1; ++i) {
// 						morton_block_indices[i] = global_result;
// 					}
// 				}
// 				result_ptr = base_result_ptr + global_result;
// 				// printf("We reset (within block algorithm) the result_ptr to base + %zu\n", global_result);
// 				global_vector = block_counter[mode] * tensor->block_layout[mode];
// 				vector_ptr = base_vector_ptr + global_vector;
// 			}
// 			free(morton_block_indices);
// 			free(block_counter);

// 			//////////////////////////////////////////////////////////////////////////////////

// 			// OBSERVATION: for all modes the tensor_ptr is the same (!)
// 			// Get partition right? = (stripes*stripe_size) = partition_size * (tid)
// 			tensor_ptr = tensor->lin.data + ((stripe+1)%stripes)*stripe_size + tid*stripes*stripe_size;

// 			if (mode == 0) {
// 				// Let's have a modulo solution here (!) especially as I am writing single lined code
// 				// If mode 0, we just care about the stripe (not the partition)
// 				base_result_ptr = result_tensor->data + ((stripe+1)%output_stripes)*output_stripe_size;

// 				// EXAMPLE WHY:
// 				// We do mode=0, when we move between stripes we reuse the same base_vector_ptr,
// 				// but vector_Ptr is modified within morton-block algorithm
// 				// thus, reset it

// 				// I DISABLED THE FOLLOWING (nice, smart, optimized code) IN THE INTEREST OF CORRECTNESS (using modulo is less error-prone than handling cases of subcases...)
// 				// DO NOT DELETE (there are some valuable tricks inside)
// 				// if (stripe+1 == stripes) {
// 				// 	printf("Rolling over (for the next stripe) by adding %zu to beginning of tensor, the elements are:\n", tid*stripes*stripe_size);
// 				// 	// reset the tensor
// 				//     tensor_ptr = tensor->lin.data + tid*stripes*stripe_size; // TRICK: rolling over means going to right partition NOT stripe
// 				//     print_to_console(tensor_ptr, stripe_size);
// 				//     if ()
// 				//     base_result_ptr = result_tensor->data; // TRICK: rolloing over means going to right parititon NOT stripe (single partition for mode = 0)
// 				//     result_ptr = base_result_ptr;
// 				// printf("the elements are:\n");
// 			 //    print_to_console(tensor_ptr, stripe_size);
// 				// printf("the elements are:\n");
// 			 //    print_to_console(tensor->lin.data + tid*stripes*stripe_size, stripe_size);
// 				// } else {
// 				// 	printf("Tensor moved; Reset the result in case it's been messed up!\n");
// 				// 	// tensor_ptr has been already moved above (block by block)
// 				// 	// move the result_ptr ahead (but only base_result_ptr has been untouched)
// 				// 	base_result_ptr += output_stripe_size;
// 				// 	result_ptr = base_result_ptr;
// 				// }
// 				// printf("the elements are:\n");
// 			 //    print_to_console(tensor_ptr, stripe_size);
// 				// printf("the elements are:\n");
// 			 //    print_to_console(tensor->lin.data + tid*stripes*stripe_size, stripe_size);
// 				// printf("the elements are:\n");
// 			 //    print_to_console(tensor_ptr, stripe_size);
// 				// printf("the elements are:\n");
// 			 //    print_to_console(tensor->lin.data + tid*stripes*stripe_size, stripe_size);


// 			} else {

// 				// printf("NOTE1: If I was to move result, I'd move by %zu + %zu elements\n", tid*output_stripes*output_stripe_size, ((stripe+1)%output_stripes)*output_stripe_size);
// 				// printf("NOTE2: result_size * mul_mode; stripe_Mul [mode] = %zu, output_stripe_size=%zu", stripe_mul[mode], output_stripe_size);

// 				// with mode zero its just correct partition, now its only the stripe (!)
// 				// Because we are not striping along the mode 0
// 				// now with all other vectors we are not partitioning them, so 
// 				// we just find the correct stripe which is modulo of current stripe?
// 				// base_result_ptr = result_tensor->data + ((stripe+1)%nthreads)*output_stripe_size;
// 				if ((stripe+1) % stripe_left == 0) {
// 					// printf("Weeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee moved left of mode\n");
// 					// Partition AND the stripe
// 					// base_result_ptr = result_tensor->data + tid*output_stripes*output_stripe_size + ((stripe+1)%output_stripes)*output_stripe_size;
// 					if (stripe+1 == stripes) {
// 						base_result_ptr = result_tensor->data + tid*output_stripes*output_stripe_size;
// 						// printf("reset the result to beginning\n");
// 					} else {
// 						base_result_ptr += stripe_size/tensor->layout2[mode];
// 						// printf("incrementing the result by stripe_size(%zu) =%zu\n",stripe_size, stripe_size/tensor->layout2[mode]);

// 						// IMPORTANT REALISATION
// 						// base_result_ptr should be incremented by partitionSize / vectorPartitionSize;
// 						// for all modes other than 0 vectorPartitionSize is the whole vector_size i.e. layout[mode]
// 						// partitionSize = stripes*stripe_size/layout[mode] 
// 					}

// 					base_vector_ptr = vector->data;
// 					// printf("reset the vector to beginning!\n");



// 					// printf("tid(%zu) times output_stripes(%zu) times output_stripe_size(%zu) + (nextStripeId=%zu) mod %zu(output_stripes) times output_stripe_size(%zu)\n",
// 					// 	tid, output_stripes, output_stripe_size, stripe+1, output_stripes, output_stripe_size);
// 				 //    base_vector_ptr = vector->data + ((stripe+1)%nthreads)*tensor->layout2[mode];
// 				 //    printf("We start at offset %zu of vector, %zu of result)\n", ((stripe+1)%nthreads)*tensor->layout2[mode], tid*output_stripes*output_stripe_size + ((stripe+1)%output_stripes)*output_stripe_size);
// 				} else if ((stripe+1) % stripe_mode == 0) {	
// 					// printf("Weeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee moved along the mode\n");

// 					// Final idea: we just reset; so reset the result_ptr to beginning of the partition?
// 					// False: for mode-2, so if you do the fastest changing mode, then you always reset, which makes no sense, right?
// 					// base_result_ptr = result_tensor->data + tid*output_stripes*output_stripe_size;
// 					if (stripe+1 == stripes) {
// 						// base_result_ptr = result_tensor->data + tid*output_stripes*output_stripe_size;
// 						// printf("reset the result to beginning\n");
// 					} else {
// 						base_result_ptr += stripe_size/tensor->layout2[mode];
// 						// printf("incrementing the result by stripe_size(%zu) =%zu\n",stripe_size, stripe_size/tensor->layout2[mode]);

// 						// IMPORTANT REALISATION
// 						// base_result_ptr should be incremented by partitionSize / vectorPartitionSize;
// 						// for all modes other than 0 vectorPartitionSize is the whole vector_size i.e. layout[mode]
// 						// partitionSize = stripes*stripe_size/layout[mode] 
// 					}
// 					base_result_ptr -= stripe_mul[mode]*(stripe_size/tensor->layout2[mode]);
// 					// printf("Redued the result ptr by %zu times (%zu div %zu)\n", stripe_mul[mode], stripe_size, tensor->layout2[mode]);
// 					// printf("reduced by ")

// 					// And we must increment the vector if we moved along the mode

// 					// Same thing with moving the vector -> first you need to know the stripe you are working on next (stripe+1)
// 					// 



// 					if (stripe+1 == stripes) {
// 						base_vector_ptr = vector->data;
// 						// base_result_ptr = result_tensor->data + tid*output_stripes*output_stripe_size;
// 						// printf("reset the vector\n");
// 					} else {
// 						base_vector_ptr += tensor->layout2[mode];
// 						// printf("incrementing the vector by %zu\n", tensor->layout2[mode]);

// 						// IMPORTANT REALISATION
// 						// base_result_ptr should be incremented by partitionSize / vectorPartitionSize;
// 						// for all modes other than 0 vectorPartitionSize is the whole vector_size i.e. layout[mode]
// 						// partitionSize = stripes*stripe_size/layout[mode] 
// 					}

// 					// Calculation is
// 					// i_k = (i / stripe_mul[mode]) % stripei[mode]
// 					// base_vector_ptr = vector->data + (((stripe+1)/stripe_mul[mode])%stripei[mode])*tensor->layout2[mode];
// 					// printf("size of stripe along the mode is %zu, stripe_mul[mode]is %zu\n", tensor->layout2[mode], stripe_mul[mode]);

// 					// New start; when you move along the mode, you always reset to beginning, don't you?
// 					// base_result_ptr = result_tensor->data + tid*output_stripes*output_stripe_size - stripe_mul[mode]*output_stripe_size;
// 					// printf("We start at offset +%zu -%zu of result\n", tid*output_stripes*output_stripe_size, stripe_mul[mode]*output_stripe_size);

// 				    // base_vector_ptr = vector->data + ((stripe+1)%nthreads)*tensor->layout2[mode];
// 				    // printf("We start at offset %zu of vector (%zu div %zu mod %zu times %zu) and %zu of result\n", ((stripe+1)/stripe_mul[mode])*tensor->layout2[mode], 
// 				    // 	(stripe+1), stripe_mul[mode], stripei[mode], tensor->layout2[mode], tid*output_stripes*output_stripe_size);
// 				} else {
// 					// printf("Weeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee moved right of mode\n");
// 					// base_result_ptr = result_tensor->data + tid*output_stripes*output_stripe_size + ((stripe+1)%output_stripes)*output_stripe_size;
// 					if (stripe+1 == stripes) {
// 						base_result_ptr = result_tensor->data + tid*output_stripes*output_stripe_size;
// 						// printf("reset the result to beginning\n");
// 					} else {
// 						base_result_ptr += stripe_size/tensor->layout2[mode];
// 						// printf("incrementing the result by stripe_size(%zu) =%zu\n",stripe_size, stripe_size/tensor->layout2[mode]);

// 						// IMPORTANT REALISATION
// 						// base_result_ptr should be incremented by partitionSize / vectorPartitionSize;
// 						// for all modes other than 0 vectorPartitionSize is the whole vector_size i.e. layout[mode]
// 						// partitionSize = stripes*stripe_size/layout[mode] 
// 					}

// 					// printf("tid(%zu) times output_stripes(%zu) times output_stripe_size(%zu) + (nextStripeId=%zu) mod %zu(output_stripes) times output_stripe_size(%zu)\n",
// 						// tid, output_stripes, output_stripe_size, stripe+1, output_stripes, output_stripe_size);
// 					// Even though the offset maybe the same, we might 
// 				    // printf("We start at same offset of vector, %zu + %zu of result\n", tid*output_stripes*output_stripe_size, ((stripe+1)%output_stripes)*output_stripe_size);
// 				}
// 				// For all other modes, you navigate both in partition and stripe
// 				// global_result += result_size;
// 				// global_tensor += block_size;
// 				// The difference here is that now we care about the partition (!) as well as the stripe (!)
// 				// base_result_ptr = result_tensor->data + ((stripe+1)%output_stripes)*output_stripe_size;
// 				// result_ptr = base_result_ptr;
// 			}

// 			result_ptr = base_result_ptr;
// 			vector_ptr = base_vector_ptr; 

// 			// if (mode == 0) {
// 			// 	#pragma omp barrier
// 			// }

// 		    // printf("TOTAL RESULTS IN THE TENSOR OUTPUT:\n");
// 		    // print_to_console(result_tensor->data, result_tensor->size+8);
// 			// printf("---------------------------------------------------------\n\n");
// 		}
// 		} // For the pragma critical (DO NOT FORGET TO DISABLE THE BARRIER)
// 	}
// 	free(block_counter_threshold);
// 	free(mul);
	
// 	// #pragma omp barrier
// 	// We need this to ensure they all finished summing up 
// 	// This is done by a single processor so we are fine (+ I think there is a barrier at the end of the parallel section above, no?)

// 	#if (TEST_ENV == 1)
// 	int result_count = result_used[0];
// 	int vector_count = vector_used[0];
// 	int tensor_count = tensor_used[0];
// 	printf("Elements of tensor/result/vector were used correspondingly %d, %d and %d times\n", tensor_count, result_count, vector_count);
// 	// First verification
// 	for (size_t i=1; i<tensor->lin.size; ++i) {
// 		if (tensor_used[i] != tensor_count) {
// 			printf("ERROR: We used one of the elements DIFFERENT NUMBER OF TIMES (tensor[%zu], %d times), (tensor[%d], %d times)\n",
// 				i, tensor_used[i], 0, tensor_count);
// 			exit(-1);
// 		}
// 	}
// 	for (size_t i=1; i<result_tensor->size; ++i) {
// 		if (result_used[i] != result_count) {
// 			printf("ERROR: We used one of the elements DIFFERENT NUMBER OF TIMES (result[%zu], %d times), (result[%d], %d times)\n",
// 				i, result_used[i], 0, result_count);
// 			exit(-1);
// 		}
// 	}
// 	for (size_t i=1; i<vector->size; ++i) {
// 		if (vector_used[i] != vector_count) {
// 			printf("ERROR: We used one of the elements DIFFERENT NUMBER OF TIMES (vector[%zu], %d times), (vector[%d], %d times)\n",
// 				i, vector_used[i], 0, vector_count);
// 			exit(-1);
// 		}
// 	}
// 	// Second verification in the following way(!)
// 	assert(tensor_count == 1);
// 	assert(result_count == (int) (tensor->lin.size / result_tensor->size));
// 	assert(vector_count == (int) (tensor->lin.size / vector->size));
// 	free(vector_used);
// 	free(result_used);
// 	free(tensor_used);
// 	#endif

// 	free(stripei);
// 	free(stripe_mul);

// }
