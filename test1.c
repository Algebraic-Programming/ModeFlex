#include <algorithms.h>
#include <gen_utils.h> // for reset_array_sizet
#include <gen_data.h> // get_vector, gen_vector, gen_block_tensor
#include <file_utils.h> // for save_to_file
#include <test.h> // for inline functions
#include <stdlib.h> // for free
#include <string.h>
#include <numa.h>
#include <libxsmm.h>
#include <omp.h>

int compare( const void* _a, const void* _b)
{
     DTYPE a = * ( (DTYPE*) _a );
     DTYPE b = * ( (DTYPE*) _b );

     if ( a == b ) return 0;
     else if ( a < b ) return -1;
     else return 1;
}

int test_algorithms(void (**ttv)(),
		const int begin,
		const int num, struct lin_storage * const result_lin,
		const DTYPE * const model_result_lin,
		const struct tensor_storage * const tensor,
		const struct lin_storage * const vector,
		const size_t mode,
		char * const filename,
		char * const filename2,
		const size_t dim,
		const size_t n,
		const size_t block_n,
		const int out_algo_in,
		DTYPE * const restrict unfold_storage,
		buffer_t * const buffer,
		libxsmm_dmmfunction * const kernel
		) {

	size_t pos = 0;
	int out_algo = -1;

	for (int algo=begin; algo<begin+num; ++algo) {

		reset_array(result_lin->data, result_lin->size, 0); 

		// if ( 	ttv[algo] == tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold ||
		// 		ttv[algo] == tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_v2_unfold ||
		// 		ttv[algo] == tvm_vector_major_BLAS_col_benchmarkable ||
 		// 		// ttv[algo] == tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_2_unfold ||
 		// 		// ttv[algo] == tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold ||
 		// 		ttv[algo] == tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_mode1 || 
 		// 		// ttv[algo] == tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_mode1 ||
 		// 		// ttv[algo] == tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal || 
 		// 		ttv[algo] == tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_partcopy ||
 		// 		ttv[algo] == tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_partcopy_stream ||
 		// 		ttv[algo] == tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_intorow ||
 		// 		ttv[algo] == tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_intorow2 ||
 		// 		ttv[algo] == tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_intocol_linin ||
 		// 		ttv[algo] == tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_intocol_linin2 ||
 		// 		ttv[algo] == tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_intocol_linin3 ||
  		// 		ttv[algo] == tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_intocol_linout) {

		// 	ttv[algo] (tensor, vector, result_lin, mode, unfold_storage);

		// } else if ( 
		// 		ttv[algo] == tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_consumer || 
		// 		ttv[algo] == tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_consumer_multicore) {

		// 	ttv[algo] (tensor, vector, result_lin, mode, NULL, buffer);
		
		// } else if (
				// ttv[algo] == tvm_vector_major_BLAS_col_GEMM_libx) {
				// ttv[algo] == tvm_vector_major_BLAS_col_mode_libx) {

			// ttv[algo] (tensor, vector, result_lin, mode, kernel);
// 

		// } else {
			ttv[algo] (tensor, vector, result_lin, mode);
		// }

		// // Do it only for blockmode_blockmode
		// if (	ttv[algo] == tvm_blockmode_major_input_aligned_output_aligned ||
		// 		ttv[algo] == tvm_blockmode_major_BLAS_input_aligned_output_aligned ||
		// 		ttv[algo] == tvm_hilbert_POWERS_libx ||
		// 		ttv[algo] == tvm_hilbert_POWERS_mkl || 
		// 		// below are obviously the TransD versions
		// 		// ttv[algo] == tvm_output_major_BLAS_row_onecall_unfold || 
		// 		ttv[algo] == tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_v2_unfold ||
		// 		ttv[algo] == tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_v2_unfold) {
		// 		// ttv[algo] == tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold) {
		// 	qsort(result_lin->data, result_lin->size, sizeof(DTYPE), compare);
		// }

		// Lenient test (against epsilon)
		pos = 0;
		while ( fequal(model_result_lin[pos], result_lin->data[pos]) ) {
			if (++pos == result_lin->size) {
				break;
			}
		}

		if (pos != result_lin->size) {
			// for (int i=0; i<result_lin->size; ++i) {
			// 	if ( fequal(model_result_lin[i], result_lin->data[i]) ) {
			// 		printf("They are equal\n");
			// 	} else {
			// 		printf("They are NOT equal: %f, %f\n", result_lin->data[i], model_result_lin[i]);
			// 	}
			// }
			// if (pos != result_lin->size || ttv[algo]==tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_intorow) {
			out_algo = algo;
			snprintf(filename, BUFSIZE, "%zu %zu %zu %zu %d", dim, mode, n, block_n, algo);
			SAVE_2(result_lin);
			printf("Algorithm %d incorrect.\n", algo);
			for (int i=0; i<result_lin->size; ++i) {
				if ( fequal(model_result_lin[i], result_lin->data[i]) ) {
					printf("They are equal\n");
				} else {
					printf("They are NOT equal: %f, %f\n", result_lin->data[i], model_result_lin[i]);
				}
			}
			exit(-1);
		} else {
			printf("Algorithm %d correct.\n", algo);
		}

	}
	if (out_algo_in == -1) {
		return out_algo;
	} else {
		// carry on with the broken algorithm from before
		return out_algo_in;
	}
}

int test1(int argc, char ** argv) {

	// initialize LIBXSMM
	libxsmm_dmmfunction a_kernel;

	// int lda,ldb,ldc;
 //  	int n,m,k;
 //  	n = 10;
 //  	m = 10;
 //  	k = 20;
	// double* a;
	// double* b;
	// double* c1;
	// double* c2;
 //    libxsmm_gemm_prefetch_type l_prefetch_op = LIBXSMM_PREFETCH_NONE;
	// lda = m;
	// ldb = k;
	// ldc = m;
	// a  = (double*)_mm_malloc(lda*k*sizeof(double), 64);
	// b  = (double*)_mm_malloc(ldb*n*sizeof(double), 64);
	// c1 = (double*)_mm_malloc(ldc*n*sizeof(double), 64);
	// c2 = (double*)_mm_malloc(ldc*n*sizeof(double), 64);

	int dim_min, dim_max, n_min, n_max;
	int mode_min, mode_max;
	int block_n_min, block_n_max;

	// we must provide default arguments
	dim_min = 2;
	dim_max = 10;
	mode_min = 0;
	mode_max = dim_max-1;
	n_min = 1;
	n_max = 24;
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

	// Set bounds for each parameter for testing (expressed as a loop)
	// Params: dim, mode, n, block_n
	// for now: do not test blockmode

	// model algorithm
	TVM model_algorithm = tvm_tensor_major;
	
	// improvement: could include the numbered versions for completeness
	TVM unfold_unfold_algorithms[] = {

		// tvm_vector_major_BLIS_col_mode,
		tvm_test_dgemm,
		// for now exclude dgemm (becuase it messes up test!)

		// tvm_vector_major,
		// tvm_output_major,
		// tvm_tensor_major,
		// tvm_tensor_major_mine,
		// tvm_block_major,
		// tvm_vector_major_BLAS_col, // this computes an unfold
		// // tvm_output_major_BLAS_row_unfold, // this computes an unfold
		
		// tvm_vector_major_BLAS_col_mode,
		// tvm_vector_major_BLAS_col_mode_multicore,
		// tvm_vector_major_BLAS_col_mode_multicore2,
		// tvm_vector_major_BLAS_col_mode_multicore3,
		// // tvm_vector_major_BLAS_col_mode2,

		// // tvm_vector_major_BLAS_col_benchmarkable,

		// tvm_vector_major_BLAS_col_mode_libx,
		// // tvm_taco,

		// // require k_leftmost
		// tvm_vector_major_input_aligned,
		// tvm_vector_major_BLAS_col_BLAS,
		// tvm_vector_major_BLAS_col_GEMM,
		// tvm_vector_major_BLAS_col_BLAS_trans,
		// // tvm_BLIS_col,

		// // WHY DO I HAVE TO EXCLUDE THIS??? ;/
		// // PLEASE FIX
		// // tvm_vector_major_BLAS_col_GEMM_libx,

		// // require k_rightmost
		// tvm_output_major_input_aligned,
		// // algorithm 16 finishes here(!)
		// tvm_output_major_BLAS_row_BLAS, // this computes an unfold
		// tvm_output_major_BLAS_row_BLAS_trans, // this computes an unfold
		// tvm_output_major_BLAS_row_libx,
		// // tvm_output_major_BLAS_row_GEMM,
		// // tvm_BLIS_row,
		// tvm_output_major_input_aligned_vectorized,

		// // block
		// tvm_block_major_input_aligned,
		// tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_v3_mine,
		// // tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_v3_old,
		
		// // openmp distirbution
		// // tvm_ppower_sync,

		// // morton block
		// tvm_morton_block_major_input_aligned,
		// tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_libx,
		// tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_mklgemm,
		// tvMortonMulticore,
		// tvMortonMulticoreMkl,
		// // tvm_morton_block_major_input_aligned_output_aligned,

		// // destructive call (destructive to the result only)
		// tvm_output_major_BLAS_row_onecall_unfold,

		// // blockmode
		// tvm_hilbert_POWERS_libx,
		// tvm_hilbert_POWERS_mkl,
		// tvm_blockmode_major_input_aligned,
		// tvm_blockmode_major_input_aligned_output_aligned
	};
	//TVM outin_unfold_unfold_output_major_algorithm = tvm_output_major_input_aligned;
	//TVM inout_unfold_unfold_output_major_algorithm = tvm_vector_major_input_aligned;

	// Inject your own tensor layout here
	//size_t tensor_layout_2[4] = {2, 3, 4, 5};
		
	for (size_t dim=(size_t) dim_min; dim<=(size_t) dim_max; ++dim) {
		printf("dim=%zu:\n", dim);

		size_t block_layout[dim];
		size_t tensor_layout[dim];

		size_t temp_mode_max;

		if (dim-1 < (size_t) mode_max) {
			temp_mode_max = dim-1;
		} else {
			temp_mode_max = mode_max;
		}
		for (size_t mode=(size_t) mode_min; mode<=temp_mode_max; ++mode) {
			printf("    mode=%zu:\n", mode);

			for (size_t n=(size_t) n_min; n<=(size_t) n_max; n*=2) {			
				printf("        n=%zu:\n", n);
				// put n as each element of tensor_layout
				reset_array_sizet(tensor_layout, dim, n);	
				// tensor_layout[0] = 8;

				// for (size_t block_n=(size_t) 2; block_n<=2; block_n*=2) {
				for (size_t block_n=(size_t) 2; block_n<=2 && block_n<=(size_t) block_n_max && block_n<=n; block_n*=2) {

					// The following code is to determine what will happen later (at blocked blocked tensor initiation)
					// size_t stripe_n = -1;
					// #pragma omp parallel
					// {
					// 	int tid = omp_get_thread_num();
					// 	int nthreads = omp_get_num_threads();
					// 	if (tid == 0) {
					// 		for (int i=0; i<dim; ++i) {
					// 			// if (block_n > tensor_layout[i]/nthreads) {
					// 			// 	// tensor->layout2[i] = tensor->block_layout[i];
					// 			// 	stripe_n = block_n;
					// 			// } else {
					// 				// HOTFIX: We must have parallelization of exactly CORES over mode 0(!)
					// 				size_t temp_value = tensor_layout[i]/block_n;
					// 				while (temp_value % nthreads != 0) {
					// 					--nthreads;
					// 				}
					// 				stripe_n = block_n * (temp_value / nthreads);
					// 			// }
					// 			break;
					// 		}
					// 	}
					// }

					// printf("            stripe_n=%zu:\n", stripe_n);
					printf("                block_n=%zu:\n", block_n);
					// put block_n as each element of block_layout
					reset_array_sizet(block_layout, dim, block_n);
					// allocate tensor,vector,result on the stack

					size_t block_size = 1;
					for (size_t d=0; d<dim; ++d) {
						block_size *= block_layout[d];
					}

					DTYPE * const restrict unfold = get_aligned_memory(sizeof(DTYPE) * block_size, ALIGNMENT_BLOCK);
	
					struct tensor_storage *tensor = gen_block_tensor(dim, tensor_layout, block_layout);
					// printf("Tensor layout:\n"), print_to_console_sizet(tensor->layout, tensor->dim);
					// printf("Block layout:\n"), print_to_console_sizet(tensor->block_layout, tensor->dim);

					DTYPE * const restrict unfold_large = get_aligned_memory(sizeof(DTYPE) * tensor->lin.size, ALIGNMENT_BLOCK);

					struct lin_storage *vector = gen_vector_interleaved(tensor->layout[mode]);
					struct tensor_storage  *result = get_block_result_tensor(tensor, mode);
					
					struct tensor_storage *model_result = get_block_result_tensor(tensor, mode);		
					model_algorithm(tensor, vector, &model_result->lin, mode);

					int out_algo = -1;
					int algo_counter = 0;

					//////////////////////////////////////////////////////////////////// UNFOLD

					out_algo = test_algorithms(unfold_unfold_algorithms, algo_counter, 10, &result->lin, model_result->lin.data, tensor, vector, mode,
							filename, filename2, dim, n, block_n, out_algo, unfold, NULL, NULL);
					algo_counter += 11;
					
					// printf("Tensor elements:\n");
					// print_to_console(tensor->lin.data, tensor->lin.size);
					
					// ignore failure of that guy
					// out_algo = -1;

					// out_algo = test_algorithms(unfold_unfold_algorithms, algo_counter, 1, &result->lin, model_result->lin.data, tensor, vector, mode,
					// 		filename, filename2, dim, n, block_n, out_algo, unfold_large, NULL, NULL);
					// algo_counter += 1;

					// out_algo = test_algorithms(unfold_unfold_algorithms, algo_counter, 1, &result->lin, model_result->lin.data, tensor, vector, mode,
					// 		filename, filename2, dim, n, block_n, out_algo, unfold, NULL, &a_kernel);
					// algo_counter += 1;

					// 	out_algo = test_algorithms(unfold_unfold_algorithms, algo_counter, 1, &result->lin, model_result->lin.data, tensor, vector, mode,
					// 			filename, filename2, dim, n, block_n, out_algo, unfold, NULL, NULL);
					// algo_counter += 1;

                    // struct tensor_storage *tensor_k_leftmost = get_in_out_unfold(tensor, 0, mode);
					// out_algo = test_algorithms(unfold_unfold_algorithms, algo_counter, 3, &result->lin, model_result->lin.data, tensor_k_leftmost, vector, mode,
					// 		filename, filename2, dim, n, block_n, out_algo, unfold, NULL, NULL);
					// algo_counter += 4;
                    // free_tensor_storage(tensor_k_leftmost);

                    // struct tensor_storage *tensor_k_rightmost = get_in_out_unfold(tensor, 1, mode);
					// out_algo = test_algorithms(unfold_unfold_algorithms, algo_counter, 5, &result->lin, model_result->lin.data, tensor_k_rightmost, vector, mode,
					// 		filename, filename2, dim, n, block_n, out_algo, unfold, NULL, NULL);
					// algo_counter += 5;
					// free_tensor_storage(tensor_k_rightmost);
					
					// // //////////////////////////////////////////////////////////////////// BLOCK

					// // algo_counter += 2;
					// struct tensor_storage *blocked_tensor = get_block_tensor(tensor, 0, 0);

					// out_algo = test_algorithms(unfold_unfold_algorithms, algo_counter, 1, &result->lin, model_result->lin.data, blocked_tensor, vector, mode,
					// 		filename, filename2, dim, n, block_n, out_algo, unfold, NULL, NULL);
					// algo_counter += 1;
					// struct tensor_storage *unblocked_result = get_block_tensor(model_result, 0, 0);
					// out_algo = test_algorithms(unfold_unfold_algorithms, algo_counter, 1, &result->lin, unblocked_result->lin.data, blocked_tensor, vector, mode,
					// 		filename, filename2, dim, n, block_n, out_algo, unfold, NULL, NULL);
					// algo_counter += 1;

					// free_tensor_storage(unblocked_result);
					// free_tensor_storage(blocked_tensor);

					// // printf("MOVING TO MORTON\n");

					// // //////////////////////////////////////////////////////////////////// MORTON

					// // printf("Moving on to Morton\n");

					// struct tensor_storage *morton_blocked_tensor = get_block_tensor(tensor, 0, 1);
					// out_algo = test_algorithms(unfold_unfold_algorithms, algo_counter, 1, &result->lin, model_result->lin.data, morton_blocked_tensor, vector, mode,
					// 		filename, filename2, dim, n, block_n, out_algo, unfold, NULL, NULL);
					// algo_counter += 1;
					// struct tensor_storage *morton_unblocked_result = get_block_tensor(model_result, 0, 1);
					// out_algo = test_algorithms(unfold_unfold_algorithms, algo_counter, 4, &result->lin, morton_unblocked_result->lin.data, morton_blocked_tensor, vector, mode,
					// 		filename, filename2, dim, n, block_n, out_algo, unfold, NULL, NULL);
					// algo_counter += 4;
					// free_tensor_storage(morton_unblocked_result);
					// free_tensor_storage(morton_blocked_tensor);
 
 					break;
					//////////////////////////////////////////////////////////////////// DESTRUCTIVE
					
					/// OKAY PROBLEM SOLVED: WHAT IM PRINTING OUT IS SORTED OUTPUT (DESTROYED) HENCE THE ERROR...

					// qsort(model_result->lin.data, model_result->lin.size, sizeof(DTYPE), compare);
					// out_algo = test_algorithms(unfold_unfold_algorithms, algo_counter, 1, &result->lin, model_result->lin.data, tensor, vector, mode,
					// 		filename, filename2, dim, n, block_n, out_algo, unfold,NULL, NULL);
					// algo_counter += 1;

					//////////////////////////////////////////////////////////////////// BLOCKMODE
					// BLOCKMODE -> destructive for the model_result, hence commented out
					// These don't work because: block_n >= n(!)
					// More importantly: vv==last_vv AND then we do %vv when vv is 0 (that why we get Floating point exception!)

					// struct tensor_storage *blockmode_tensor = get_blockmode_tensor(tensor, mode, 0); // block 
					// out_algo = test_algorithms(unfold_unfold_algorithms, algo_counter, 1, &result->lin, model_result->lin.data, blockmode_tensor, vector, mode,
					// 		filename, filename2, dim, n, block_n, out_algo, unfold, NULL);
					// algo_counter += 1;
					
					// compare to unblocked result (unblock the result before comparison)
					//int result_mode;
					//if (mode == 0) {
						//result_mode = 0;
					//} else {
						//result_mode = mode-1;
					//}

					// There is a problem with the result of blockmode_blockmode -> just resort to using qsort...
					// qsort(model_result->lin.data, model_result->lin.size, sizeof(DTYPE), compare);
					// out_algo = test_algorithms(unfold_unfold_algorithms, algo_counter, 1, &result->lin, model_result->lin.data, blockmode_tensor, vector, mode,
					// 		filename, filename2, dim, n, block_n, out_algo, unfold, NULL, NULL);
					// algo_counter += 1;
					// free_tensor_storage(blockmode_tensor);

					//////////////////////////////////////////////////////////////////// HILBERT CURVE 

					// struct tensor_storage *blocked_tensor = get_block_tensor(tensor, 0, 0);
					// struct tensor_storage *unblocked_result = get_block_tensor(model_result, 0, 0);
					// algo_counter += 28;
					// printf("tensor:\n");
					// print_to_console(tensor->lin.data, tensor->lin.size);

					// struct tensor_storage *hilbert_blocked_tensor = get_block_tensor(tensor, 0, 2);
					// printf("hilbert tensor:\n");
					// print_to_console(hilbert_blocked_tensor->lin.data, hilbert_blocked_tensor->lin.size);

					// printf("and the vector is:\n");
					// print_to_console(vector->data, vector->size);

					algo_counter = 24;
					// qsort(unblocked_result->lin.data, unblocked_result->lin.size, sizeof(DTYPE), compare);
					// out_algo = test_algorithms(unfold_unfold_algorithms, algo_counter, 2, &result->lin, unblocked_result->lin.data, hilbert_blocked_tensor, vector, mode,
					// 		filename, filename2, dim, n, block_n, out_algo, unfold, NULL, NULL);
					// free_tensor_storage(unblocked_result);
					// free_tensor_storage(blocked_tensor);
					// free_tensor_storage(hilbert_blocked_tensor);

					//////////////////////////////////////////////////////////////////// BLOCKMODE RESULTMAJOR
#if 0
					struct tensor_storage *blockmode_tensor = get_blockmode_tensor(tensor, mode, 0); // block 
					out_algo = test_algorithms(unfold_unfold_algorithms, algo_counter, 1, &result->lin, model_result->lin.data, blockmode_tensor, vector, mode,
							filename, filename2, dim, n, block_n, out_algo, unfold, NULL);
					algo_counter += 1;

					// compare to unblocked result (unblock the result before comparison)
					//int result_mode;
					//if (mode == 0) {
						//result_mode = 0;
					//} else {
						//result_mode = mode-1;
					//}

					// There is a problem with the result of blockmode_blockmode -> just resort to using qsort...
					qsort(model_result->lin.data, model_result->lin.size, sizeof(DTYPE), compare);
					out_algo = test_algorithms(unfold_unfold_algorithms, algo_counter, 1, &result->lin, &model_result->lin.data, blockmode_tensor, vector, mode,
							filename, filename2, dim, n, block_n, out_algo, unfold, NULL);
					algo_counter += 1;
					free_tensor_storage(blockmode_tensor);
#endif
					////////////////////////////////////////////////////////////////////

					if ((out_algo != -1)) {
						snprintf(filename, BUFSIZE, "%zu %zu %zu %zu %d", dim, mode, n, block_n, -1);
						SAVE(model_result->lin);
						exit(-1);
					}

					free(unfold_large);
					free(unfold);
					free_tensor_storage(model_result);
					free_tensor_storage(result); // quick fix so this could cause errors(!)
					// free_result_tensor_storage(result, mode);
					free_lin_storage(vector);
					free_tensor_storage(tensor);

				// finished with blocks
				}

			// finished with ns
			}

			// mode level loop
			if (mode == dim-1) {
				break;
			}
		// finished with mode
		}
	
	// finished with dim
	}

	return 0;
}

