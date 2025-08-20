#include <time_meas.h>
#include <time.h>
#include <unistd.h> // for sleep
#include <algorithms.h>
#include <math.h>
#include <stdlib.h>
#include <gen_utils.h>
#include <assert.h>
#include <string.h>
#include <file_utils.h>
#include <omp.h>
#include <mkl.h>
// #include <valgrind/callgrind.h>

#define _GNU_SOURCE
#define MILLION 1000000L
#define BILLION 1000000000L
#define HEADER "timestamp,elapsed,response,success,dim,mode,n,block_n,algo_id,std,mworst,mbest,perf,mbuf,mkl,omp,operation\n"
#define HEADERTMM "timestamp,elapsed,response,success,dim,mode,n,block_n,algo_id,std,mworst,mbest,perf,mbuf,mkl,omp,operation,l_dim,l_block_dim\n"
#define HEADERPM "timestamp,elapsed,response,success,dim,mode,n,block_n,algo_id,std,mworst,mbest,perf,mbuf,mkl,omp,operation\n"

#define FASTTEST

#ifdef FASTTEST
	#define TIMES 5
#else 
	#define TIMES 10
#endif

//type of a anonymous function pointer
//(does not care about signature) 
typedef void (*TVM)();

/** Translates a function pointer to a tensor vector multiplication code to a string. */
char * toString(TVM x) {
	if ( x == tvm_tensor_major ) {
		return "unfoldBaseline";
	// } else if ( x == tvm_tensor_major_mine ) {
	// 	return "unfoldBaselinePerf";
	// } else if ( x == tvm_vector_major_BLAS_col_benchmarkable ) {
	// 	return "tvmUNFOLD";
	// } else if ( x == tvm_taco ) {
	// 	return "tvmTaco";
	// } else if ( x == tvm_vector_major_input_aligned ) {
	// 	return "vector";
	// } else if ( x == tvm_output_major_input_aligned ) {
	// 	return "output";
	// } else if ( x == tvm_vector_major_BLAS_col ) {
	// 	return "blas_col";
	// } else if ( x == tvm_vector_major_BLAS_col_BLAS ) {
	// 	return "MKL_colmajor";
	// } else if ( x == tvm_vector_major_BLAS_col_GEMM_libx ) {
	// 	return "LIBX_colmajor";
	// } else if ( x == tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_v4 ) {
	// 	return "blockColNoTrans";
	// // } else if ( x == tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_4 ) {
	// 	// return "mortonColNoTrans";
	// // } else if ( x == tvm_vector_major_BLAS_col_mode ) {
	// // 	return "unfoldColNoTrans";
	// // } else if ( x == tvm_vector_major_BLAS_col_mode_libx ) {
	// // 	return "unfoldColNoTransLibx";
	// } else if ( x == tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_v2 ) {
	// 	return "blockRowNoTransD";
	// } else if ( x == tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_v2_unfold ) {
	// 	return "blockRowNoTransDUnfold";
	// } else if ( x == tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_intorow ) {
	// 	return "blockRowNoTransNontemporalintorow";
	// } else if ( x == tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_intocol_linin3 ) {
	// 	return "blockRowNoTransNontemporalintocol_linOUT";
	// } else if ( x == tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_intocol_linout ) {
	// 	return "blockRowNoTransNontemporalintocol_MKL";
	// } else if ( x == tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_partcopy ) {
	// 	return "blockRowNoTransPartCopy";
	// } else if ( x == tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_partcopy_stream ) {
	// 	return "CopySTREAM";
	// // } else if ( x == tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal ) {
	// // 	return "mortonRowNoTransNontemporal";
	// } else if ( x == tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_mode1 ) {
	// 	return "blockRowNoTransNontemporalMode";
	// } else if ( x == tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_mode1 ) {
	// // 	return "mortonRowNoTransNontemporalMode";
	// } else if ( x == tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_consumer ) {
	// 	return "blockRowNoTransNontemporalHT";
	// } else if ( x == tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_consumer_multicore ) {
	// 	return "blockRowNoTransNontemporalMULTICORE";
	// // } else if ( x == tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS ) {
	// // 	return "mortonRowNoTransD";
	// // } else if ( x == tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold ) {
	// // 	return "mortonRowNoTransDUnfold";
	// // } else if ( x == tvm_output_major_BLAS_row_onecall ) {
	// // 	return "unfoldRowNoTransD";
	// // } else if ( x == tvm_output_major_BLAS_row_onecall_unfold ) {
	// // 	return "unfoldRowNoTransDUnfold";
	// } else if ( x == tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_v3 ) {
	// 	return "blockRowTransPerf";
	// } else if ( x == tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_v3_libx ) {
	// 	return "blockRowTransPerfLibx";
	// } else if ( x == tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_v3_mine ) {
	// 	return "blockRowTransPerfMine";
	// // } else if ( x == tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3 ) {
	// // 	return "mortonRowTransPerf";
	// // } else if ( x == tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_libx ) {
	// 	return "mortonRowTransLibx";
	// } else if ( x == tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_mklgemm ) {
	// 	return "mortonRowTransMKLgemm";
	// } else if ( x == tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_mklgemminlined ) {
	// 	return "mortonRowTransMKLgemmINLINED";
	// } else if ( x == tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_libx_tensor ) {
	// 	return "mortonRowTransLibxTensor";
	// } else if ( x == tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_libx_result ) {
	// 	return "mortonRowTransLibxResult";
	// } else if ( x == tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_libx_vector ) {
	// 	return "mortonRowTransLibxVector";

	// } else if ( x == tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_libx_tensor2 ) {
	// 	return "mortonRowTransLibxTensor2";
	// } else if ( x == tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_libx_result2 ) {
	// 	return "mortonRowTransLibxResult2";
	// } else if ( x == tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_libx_vector2 ) {
	// 	return "mortonRowTransLibxVector2";

	// } else if ( x == tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_libx_tensor3 ) {
	// 	return "mortonRowTransLibxTensor3";
	// } else if ( x == tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_libx_vector3 ) {
	// 	return "mortonRowTransLibxVector3";
	
	// } else if ( x == tvm_ppower_sync_mkl_p ) {
	// 	return "tvPsyncMKL";

	// } else if ( x == tvm_power_sync_mkl_p ) {
	// 	return "tvMKL_";
	// } else if ( x == tvm_power_sync_p ) {
	// 	return "tvLIBX_";

	// } else if ( x == tvm_vector_major_BLAS_col_mode_multicore ) {
	// 	return "tvBaseline";
	// } else if ( x == tvm_vector_major_BLAS_col_mode_multicore2 ) {
	// 	return "tvBaseline2";
	// } else if ( x == tvm_vector_major_BLAS_col_mode_multicore3 ) {
	// 	return "tvBaseline3";
	// } else if ( x == tvm_ppower_sync ) {
	// 	return "tvPsyncMorton";
	// } else if ( x == tvm_ppower_sync_mkl ) {
	// 	return "tvPsyncMortonMkl";



	// } else if ( x == tvm_hilbert_POWERS_libx ) {
	// 	return "hilbert_tvm_libx";
	// } else if ( x == tvm_hilbert_POWERS_mkl ) {
	// 	return "hilbert_tvm_mkl";
	// // } else if ( x == pmTaco10 ) {
	// 	return "pmTaco10";
	// } else if ( x == pmTaco5 ) {
	// 	return "pmTaco5";

	// } else if ( x == tmm_looped_mkl ) {
	// 	return "tmm_looped_mkl";
	// } else if ( x == tmm_blocked_mkl ) {
	// 	return "tmm_blocked_mkl";
	// } else if ( x == tmm_blocked_libx ) {
	// 	return "tmm_doubly_blocked_mkl";
	// } else if ( x == tmm_mortonblocked_mkl ) {
	// 	return "tmm_mortonblocked_mkl";
	// } else if ( x == tmm_blocked_trans ) {
	// 	return "tmm_blocked_trans";
	// } else if ( x == tmm_mortonblocked_trans ) {
	// 	return "tmm_mortonblocked_trans";
		
  	// } else if ( x == pmLooped ) {
	// 	return "unfold_BLAS_looped";
	// } else if ( x == pmLoopedSingle ) {
	// 	return "unfold_BLAS_single_left";
	// } else if ( x == pmBlock ) {
	// 	return "block_BLAS_looped";
	// } else if ( x == pmMorton ) {
	// 	return "morton_BLAS_looped";
	// } else if ( x == pmBlockLibx ) {
	// 	return "block_LIBX_looped";
	// } else if ( x == pmMortonLibx ) {
	// 	return "morton_LIBX_looped_onebuffer"; // 		return "morton_LIBX_looped";
	// } else if ( x == pmMortonSingle ) {
	// 	return "morton_BLAS_single_left";
	// } else if ( x == pmMortonLibxSingle ) {
	// 	return "morton_LIBX_single_left"; // 		return "morton_LIBX_single_left_onebuffer";
	// } else if ( x == pmMortonSingleMvs ) {
	// 	return "morton_BLAS_single_right";
	// } else if ( x == pmMortonLibxSingleMvs ) {
	// 	return "morton_LIBX_single_right"; // 		return "morton_LIBX_single_right_onebuffer";
	// } else if ( x == pmMortonLibxVms ) {
	// 	return "pmMortonLibxVms";
	// } else if ( x == pmLoopedSingleMvs ) {
	// 	return "unfold_BLAS_single_right";

  	// } else if ( x == pmBlockSingleMvs ) {
	// 	return "block_BLAS_single_right";

	// } else if ( x == pmMortonMyself ) {
	// 	return "pmMortonNoblas";

	// } else if ( x == tvMortonMulticore ) {
	// 	return "tvMortonMulticore";
	// } else if ( x == tvMortonMulticoreMkl ) {
	// 	return "tvMortonMulticoreMkl";

	// // } else if ( x == tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_mine ) {
	// // 	return "mortonRowTransPerfMine";
	// // } else if ( x == tvm_vector_major_BLIS_col_mode ) {
	// // 	return "unfoldRowTransPerf";
	// } else if ( x == tvm_block_major_input_aligned_output_aligned_BLAS_POWERS ) {
	// 	return "blockRowNoTrans";
	// } else if ( x == tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold ) {
	// 	return "blockRowNoTransUnfold";
	// // } else if ( x == tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_2 ) {
	// // 	return "mortonRowNoTrans";
	// // } else if ( x == tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_2_unfold ) {
	// // 	return "mortonRowNoTransUnfold";
	// // } else if ( x == tvm_output_major_BLAS_row ) {
	// // 	return "unfoldRowNoTrans";
	// // } else if ( x == tvm_output_major_BLAS_row_unfold ) {
	// // 	return "unfoldRowNoTransUnfold";
	// // } else if ( x == tvm_morton_block_major_input_aligned_output_aligned ) {
	// // 	return "morton";
	// // } else if ( x == tvm_morton_block_major_input_aligned_output_aligned_POWERS ) {
	// // 	return "morton_powers";
	// // } else if ( x == tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_bench ) {
	// // 	return "morton_BLAS_powers_bench";
	// // } else if ( x == tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_bench_2 ) {
	// // 	return "morton_BLAS_powers_bench_2";
	// // } else if ( x == tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_bench_3 ) {
	// // 	return "morton_BLAS_powers_bench_3";
	// } else if ( x == tvm_block_major_input_aligned_output_aligned ) {
	// 	return "block";
	// } else if ( x == tvm_block_major_input_aligned_output_aligned_POWERS ) {
	// 	return "block_powers";
	// } else if ( x == tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_bench ) {
	// 	return "block_BLAS_powers_bench";
	// // } else if ( x == tvm_morton_block_major_input_aligned_output_aligned_POWERS_aux ) {
	// // 	return "morton_aux";
	// } else if ( x == tvm_block_major_input_aligned_output_aligned_aux_v1 ) {
	// 	return "block_aux1";
	// } else if ( x == tvm_block_major_input_aligned_output_aligned_aux_v2 ) {
	// 	return "block_aux2";
	// } else if ( x == tvm_block_major_input_aligned_output_aligned_aux_v3 ) {
	// 	return "block_aux3";
	// // } else if ( x == tvm_output_major_BLAS_row ) {
	// // 	return "unfold";
	// // } else if ( x == tvm_output_major_BLAS_row_BLAS ) {
	// // 	return "MKL_rowmajor";
	// // } else if ( x == tvm_output_major_BLAS_row_libx ) {
	// // 	return "LIBX_rowmajor";
	// // } else if ( x == tvm_output_major_BLAS_row_UNFOLD ) {
	// // 	return "unfold_UNFOLD";
	// } else if ( x == tvm_block_major_input_aligned_output_aligned_BLAS_POWERS ) {
	// 	return "block_BLAS";
	// } else if ( x == tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_UNFOLD) {
	// 	return "block_BLAS_UNFOLD";
	// } else if ( x == tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_BLAS) {
	// 	return "block_BLAS_BLAS";
	// } else if ( x == tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS ) {
	// 	return "morton_BLAS";
	// } else if ( x == tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_UNFOLD) {
	// 	return "morton_BLAS_UNFOLD";
	// } else if ( x == tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_BLAS) {
	// 	return "morton_BLAS_BLAS";
	} else if ( x == block_array_int ) {
		return "block_array_int";
	} else {
		return "ERROR";
	}
}

// opens the file and writes down a line of results
// time(NULL) - epoch time,
// timespec_to_microseconds - elapsed time of a function in microseconds,
// algo - the integer ID of the algorithm
void
write_perf_result(FILE * const file_handle, 
		const size_t * const layout,
		const double elapsed_time, const double standard_deviation,
		const size_t dim, const size_t mode, const size_t n, const size_t block_n, const size_t block_size, char * algo_name, char * operation, const size_t size_ten, const size_t size_vec, const size_t size_res)  {
	
	(void) block_size;
	
	// printf("Printing the tensor dimensions as considered in calculation of performance:\n");
	// print_to_console_sizet(layout, dim);

	// REDO ALL THIS(!!!)
	size_t tensor_size = 1;
	size_t vector_size = 1;
	size_t result_size = 1;
	for (size_t d=0; d<=dim-1; ++d) {
		tensor_size *= layout[d];
		if (d != mode) {
			result_size *=layout[d];
		} else {
			vector_size = layout[d];
		}
	}
	assert(tensor_size == size_ten);
	assert(vector_size == size_vec);
	assert(result_size == size_res);

	// printf("ten=%zu, vec=%zu, res=%zu\n", tensor_size, vector_size, result_size);
	// printf("COMPARE: tensor_size=%zu, pow(n,dim)=%zu\n", tensor_size, (size_t) pow(n,dim));

	// calculate mbest,mworst,perf
	double tensor_size_in_bytes = tensor_size * sizeof(DTYPE);
	double vector_size_in_bytes = vector_size * sizeof(DTYPE);
	double result_size_in_bytes = result_size * sizeof(DTYPE);
	double time_in_nanoseconds = elapsed_time;
	// Important: performance is not referring to bytes(!)
	double perf = ((tensor_size * 2) * 1000000000.0) / (time_in_nanoseconds * 1073741824.0);
	// double block_size_in_bytes = block_size * sizeof(DTYPE);
	double mworst = ((tensor_size_in_bytes * 3) * 1000000000.0) / (time_in_nanoseconds * 1073741824.0);
	double mbest = ((tensor_size_in_bytes + vector_size_in_bytes + result_size_in_bytes) * 1000000000.0) / (time_in_nanoseconds * 1073741824.0);
	double mbuf = ((tensor_size_in_bytes * 2) * 1000000000.0) / (time_in_nanoseconds * 1073741824.0);
	struct timespec time;
	int rc = clock_gettime(CLOCK_MONOTONIC, &time);
	if( rc != 0 ) {
		fprintf( stderr, "Error while getting current time. Attempting to continue anyway...\n" );
	}
	// printf("still okay here\n");

	(void) fprintf(file_handle, "%f,%f,%s,%s,%zu,%zu,%zu,%zu,%s,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%s\n",
			timespec_to_microseconds(time),
			elapsed_time,
			"N/A",
			"true",
			dim, mode, n, block_n, algo_name,
			standard_deviation,
			mworst,mbest,perf,mbuf,
			0, 0, operation);
			// tensor_size_in_bytes,vector_size_in_bytes,result_size_in_bytes,block_size_in_bytes,compare_block_size);

	// printf("we are stilll okay after the print statement\n");

	if (ferror(file_handle)) {
		(void) printf("Error writing <header>.\n");
	}
}

void
write_multicore_result(FILE * const file_handle, 
		const size_t * const layout, const int parameter_p,
		const double elapsed_time, const double standard_deviation,
		const size_t dim, const size_t mode, const size_t n, const size_t block_n, const size_t block_size, const char * const algo_name, const char * const operation, const size_t * const p_array, const size_t * const n_array, const size_t * const b_array, const int p_size, const size_t size_ten, const size_t size_vec, const size_t size_res)  {
	
	(void) block_size;

	// printf("Printing the tensor dimensions as considered in calculation of performance:\n");
	// print_to_console_sizet(layout, dim);

	// REDO ALL THIS(!!!)
	size_t tensor_size = 1;
	size_t vector_size = 1;
	size_t result_size = 1;
	for (size_t d=0; d<=dim-1; ++d) {
		tensor_size *= layout[d];
		if (d != mode) {
			result_size *=layout[d];
		} else {
			vector_size = layout[d];
		}
	}
	assert(tensor_size == size_ten);
	assert(vector_size == size_vec);
	assert(result_size == size_res);

	// printf("ten=%zu, vec=%zu, res=%zu\n", tensor_size, vector_size, result_size);
	// printf("COMPARE: tensor_size=%zu, pow(n,dim)=%zu\n", tensor_size, (size_t) pow(n,dim));

	int mkl_threads = mkl_get_max_threads();
	int omp_threads = p_size;

	// Idiotyczna wstawka aby obliczyc ile jak tam miałem mkl_threads (eh)
	if (strcmp(algo_name, "tvBaseline3") == 0) {

		int nthreads = omp_threads;

		size_t right_size = 1;
		for (size_t d=dim-1; d<dim; --d) {
			if (d > mode) {
				right_size *= layout[d];
			}
		}
		size_t left_size = tensor_size / (vector_size * right_size);
		if ((left_size / nthreads) == 0) {
	    	// printf("we are in case 0, because %zu div by %d is %d\n", left_size, nthreads, left_size / nthreads);
			mkl_threads = nthreads / left_size;
			omp_threads = left_size;
		} else {
	    	// printf("we are in case 1, because %zu div by %d is %d\n", left_size, nthreads, left_size / nthreads);
			mkl_threads = 1;
			omp_threads = nthreads;
		}
		if (mode == dim-1) {
			mkl_threads = nthreads;
			omp_threads = 1;
		}
	    printf("INFO: BASELINE3: omp threads = %d, mkl threads = %d which is in total no more than what we have initially (%d)\n",
	    	omp_threads, mkl_threads, nthreads);
		
	}
	
	char new_algo_name[20];
	strcpy(new_algo_name, algo_name);
	// if (strcmp(algo_name, "tvPsync") == 0 || strcmp(algo_name, "tvPsyncMKL") == 0) {
	// 	// printf("Writing a result of P-paramterized algorithm.\n");
	// 	size_t string_len = strlen(algo_name);
	// 	new_algo_name[string_len] = 'P';
	// 	new_algo_name[string_len+1] = ((char) parameter_p) + '0';
	// 	new_algo_name[string_len+2] = '\0';
	// 	// printf("====\n %s,\n ", new_algo_name);
	// }

	if ((strcmp(algo_name, "tvLIBX_") == 0) || (strcmp(algo_name, "tvMKL_") == 0)) {
		// We have a special case, let's rename this "algorithm"
		size_t string_len = strlen(algo_name);
		if (get_size(p_array, dim) == (size_t) omp_threads) {
			new_algo_name[string_len] = '0';
		} else {
			new_algo_name[string_len] = 'p';
		}
		new_algo_name[string_len+1] = 's';
		new_algo_name[string_len+2] = 'y';
		new_algo_name[string_len+3] = 'n';
		new_algo_name[string_len+4] = 'c';
		#ifdef TESTP
		new_algo_name[string_len+5] = '_';
		new_algo_name[string_len+6] = ((char) parameter_p) + '0';
		new_algo_name[string_len+7] = '\0';
		#else
		new_algo_name[string_len+5] = '\0';
		#endif
	}
	
	// calculate mbest,mworst,perf
	double tensor_size_in_bytes = tensor_size * sizeof(DTYPE);
	double vector_size_in_bytes = vector_size * sizeof(DTYPE);
	double result_size_in_bytes = result_size * sizeof(DTYPE);
	double time_in_nanoseconds = elapsed_time;
	// Important: performance is not referring to bytes(!)
	double perf = ((tensor_size * 2) * 1000000000.0) / (time_in_nanoseconds * 1073741824.0);
	// double block_size_in_bytes = block_size * sizeof(DTYPE);
	double mworst = ((tensor_size_in_bytes * 3) * 1000000000.0) / (time_in_nanoseconds * 1073741824.0);
	double mbest = ((tensor_size_in_bytes + vector_size_in_bytes + result_size_in_bytes) * 1000000000.0) / (time_in_nanoseconds * 1073741824.0);
	double mbuf = ((tensor_size_in_bytes * 2) * 1000000000.0) / (time_in_nanoseconds * 1073741824.0);
	struct timespec time;
	int rc = clock_gettime(CLOCK_MONOTONIC, &time);
	if( rc != 0 ) {
		fprintf( stderr, "Error while getting current time. Attempting to continue anyway...\n" );
	}

	printf("	INFO (DEBUG): P_ARRAY_IN:\n");
	print_to_console_sizet(p_array, dim);

	#if (TEST_ENV == 0)
	(void) fprintf(file_handle, "%f,%f,%s,%s,%zu,%zu,%zu,%zu,%s,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%s,",
		timespec_to_microseconds(time),
		elapsed_time,
		"N/A",
		"true",
		dim, mode, n, block_n, new_algo_name,
		standard_deviation,
		mworst,mbest,perf,mbuf,
		mkl_threads, omp_threads, GIT_HASH);
	fprint_ptr_sizet(file_handle, b_array, dim);
	fprintf(file_handle, ",");
	fprint_ptr_sizet(file_handle, p_array, dim);
	fprintf(file_handle, ",");
	fprint_ptr_sizet(file_handle, n_array, dim);
	(void) fprintf(file_handle, ",%s\n", operation);

	#else
	printf("%f,%f,%s,%s,%zu,%zu,%zu,%zu,%s,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%s,",
		timespec_to_microseconds(time),
		elapsed_time,
		"N/A",
		"true",
		dim, mode, n, block_n, new_algo_name,
		standard_deviation,
		mworst,mbest,perf,mbuf,
		mkl_threads, omp_threads, GIT_HASH);
	print_to_console_sizet(b_array, dim);
	printf(",");
	print_to_console_sizet(p_array, dim);
	printf(",");
	print_to_console_sizet(n_array, dim);
	printf(",%s\n",operation);
	#endif

	if (ferror(file_handle)) {
		(void) printf("Error writing <header>.\n");
	}
}

void
write_pm_perf_result(FILE * const file_handle, 
		const size_t * const layout,
		const double elapsed_time, const double standard_deviation,
		const size_t dim, const size_t n, const size_t block_n, const size_t block_size_unused, char * algo_name, char * operation)  {
	
	(void) block_size_unused;

	printf("Printing the tensor dimensions as considered in calculation of performance:\n");
	print_to_console_sizet(layout, dim);

	// REDO ALL THIS(!!!)
	size_t tensor_size = 1;
	size_t block_size = 1;
	for (size_t d=0; d<=dim-1; ++d) {
		tensor_size *= layout[d];
		block_size *= block_n;
	}

	// calculate mbest,mworst,perf
	size_t vector_size_in_bytes = layout[0] * sizeof(DTYPE);
	size_t tensor_size_in_bytes = tensor_size * sizeof(DTYPE);
	size_t result_size_in_bytes = (tensor_size / layout[0]) * sizeof(DTYPE);

	// Input step
	// Step for remaining dimensions... (dim-1)!
	// Last step: add vectors(!)
	size_t memory_touched = tensor_size_in_bytes;
	for(size_t i=0; i<dim-2; ++i) 
	{
		memory_touched += 2*result_size_in_bytes;
		result_size_in_bytes /= vector_size_in_bytes; // This is not in bytes ANYMORE(!!!)
		result_size_in_bytes *= sizeof(DTYPE); // that's why you should never ever do weird maths...
	}
	// dn is by the vectors touched in the comutation, 2n by normalization (!)
	memory_touched += (dim+2) * vector_size_in_bytes;
	// printf("Memory touched by the algorithm is %zu\n", memory_touched);

	double time_in_nanoseconds = elapsed_time;

	// multiply by dimensionality (because the number of steps = number of dimensions)
	double mbuf = (memory_touched*dim) / time_in_nanoseconds;
	
	struct timespec time;
	int rc = clock_gettime(CLOCK_MONOTONIC, &time);
	if( rc != 0 ) {
		fprintf( stderr, "Error while getting current time. Attempting to continue anyway...\n" );
	}

	(void) fprintf(file_handle, "%f,%f,%s,%s,%zu,%zu,%zu,%zu,%s,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%s\n",
			timespec_to_microseconds(time),
			elapsed_time,
			"N/A",
			"true",
			dim, (size_t) 0, n, block_n, algo_name,
			standard_deviation,
			0.0,0.0,0.0,mbuf,
			0, 0, operation);

	if (ferror(file_handle)) {
		(void) printf("Error writing <header>.\n");
	}
}

void
write_tmm_perf_result(FILE * const file_handle, 
		const size_t * const layout,
		const double elapsed_time, const double standard_deviation,
		const size_t dim, const size_t mode, const size_t n, const size_t block_n, const size_t block_size, char * algo_name, char * operation,
		const int l_dim, const int l_block_dim)  {
	
	(void) block_size;

	// printf("Printing the tensor dimensions as considered in calculation of performance:\n");
	// print_to_console_sizet(layout, dim);

	// REDO ALL THIS(!!!)
	size_t tensor_size = 1;
	size_t matrix_size = 1;
	size_t result_size = 1;
	for (size_t d=0; d<=dim-1; ++d) {
		tensor_size *= layout[d];
		if (d != mode) {
			result_size *=layout[d];
		} else {
			matrix_size = layout[d];
		}
	}
	matrix_size *= l_dim;
	printf("ten=%zu, vec=%zu, res=%zu (in doubles)\n", tensor_size, matrix_size, result_size);
	// printf("COMPARE: tensor_size=%zu, pow(n,dim)=%zu\n", tensor_size, (size_t) pow(n,dim));

	// calculate mbest,mworst,perf
	double tensor_size_in_bytes = tensor_size * sizeof(DTYPE);
	double matrix_size_in_bytes = matrix_size * sizeof(DTYPE);
	double result_size_in_bytes = result_size * sizeof(DTYPE);
	double time_in_nanoseconds = elapsed_time;

	// Important: performance is not referring to bytes(!)
	double perf = (tensor_size * 2 * l_dim) / time_in_nanoseconds;
	// double block_size_in_bytes = block_size * sizeof(DTYPE);
	double mworst = (tensor_size_in_bytes * 3) / time_in_nanoseconds;
	double mbest = (tensor_size_in_bytes + matrix_size_in_bytes + result_size_in_bytes) / time_in_nanoseconds;
	double mbuf = (tensor_size_in_bytes * 2) / time_in_nanoseconds;

	struct timespec time;
	int rc = clock_gettime(CLOCK_MONOTONIC, &time);
	if( rc != 0 ) {
		fprintf( stderr, "Error while getting current time. Attempting to continue anyway...\n" );
	}

	(void) fprintf(file_handle, "%f,%f,%s,%s,%zu,%zu,%zu,%zu,%s,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%s,%d,%d\n",
			timespec_to_microseconds(time),
			elapsed_time,
			"N/A",
			"true",
			dim, mode, n, block_n, algo_name,
			standard_deviation,
			mworst,mbest,perf,mbuf,
			0, 0, operation, l_dim, l_block_dim);
			// tensor_size_in_bytes,vector_size_in_bytes,result_size_in_bytes,block_size_in_bytes,compare_block_size);

	if (ferror(file_handle)) {
		(void) printf("Error writing <header>.\n");
	}
}

/** TODO make comments better. For instance, the function is not actually returning anything
 * @param[in] f The tensor multiply vector kernel to be used.
 * Measures each function according to the benchmark strategy and writes the result using write_perf_result
 */
void
measure_unfold(
	void (*f)(),
	const struct tensor_storage * restrict const a,
	const struct lin_storage * restrict const b,
	struct lin_storage * restrict const c,
	const size_t d, FILE * const file,
	const size_t n, const size_t block_n,
	DTYPE * restrict const e,
	const size_t block_size
) {
	struct timespec start;
	struct timespec stop;
	double average_time = 0;
	double standard_deviation = 0;
	double results[TIMES];
	DTYPE checksum = 0;

	// Make a single "cold" call
	reset_array(c->data, c->size, 0);
	(void) memset(e, 0, block_size*sizeof(DTYPE));
	(*f)(a,b,c,d,e);

	printf("Begining the measurement for these params.\n");

	// Measure a sample runtime
	int rc1 = clock_gettime(CLOCK_MONOTONIC, &start);
	(*f)(a,b,c,d,e);
	int rc2 = clock_gettime(CLOCK_MONOTONIC, &stop);
	if (rc1 != 0 || rc2 != 0) {
		fprintf(stderr, "Clock_gettime: Non-zero return code %d/%d from clock_gettime\n", rc1, rc2);
	}

	struct timespec a_diff = timespec_diff(start,stop);
	const double sample_time = timespec_to_ns(&a_diff);
	size_t iterations = (BILLION / sample_time) + 1;
	if (iterations == 0) {
		iterations = 1;
	}
	// printf("Sample time = %.17g, iterations = %zu\n",  timespec_to_microseconds(timespec_diff(start, stop)), iterations);

	for (int i=0; i<TIMES; ++i) {
		// This "trashes" the cache 
		reset_array(c->data, c->size, 0);
		(void) memset(e, 0, block_size*sizeof(DTYPE));
		// Sleep before having a single cold run
		sleep(1);
		(*f)(a,b,c,d,e);

		rc1 = clock_gettime(CLOCK_MONOTONIC, &start);
		for ( size_t count = 0; count < iterations; ++count ) {
			(*f)(a,b,c,d,e);
		}
		rc2 = clock_gettime(CLOCK_MONOTONIC, &stop);
		//total_time contains the mini_result for a single run
		a_diff = timespec_diff(start,stop);
		const double total_time = (timespec_to_ns(&a_diff)) / ((double)iterations);
		//printf("Current meas: %f microsec\n", total_time);
		for( size_t k = 0; k < c->size; ++k ) {
			checksum += c->data[k];
		}
		printf( "Checksum at iteration %d: %f.\n", i, checksum );
		//checksum also the unfold (For algorithms which ONLY? compute the unfold)
		checksum = 0;
		for( size_t k = 0; k < block_size; ++k ) {
			checksum += e[k];
		}
		printf( "Checksum at iteration %d: %f.\n", i, checksum );
		average_time += total_time;
		results[i] = total_time;
		checksum = 0;
	}

	average_time /= (double)TIMES;
	for (int i=0; i<TIMES; ++i) {
		//printf("ave(%.2f) - result(%.2f): %.2f microsec\n", average_time, results[i], results[i] - average_time);
		standard_deviation += ((results[i]-average_time) * (results[i]-average_time));
	}

	// assert( TIMES > 1 );
	standard_deviation = standard_deviation / ((double)(TIMES-1));
	standard_deviation = sqrt(standard_deviation);
	write_perf_result(file, a->layout,
			average_time, standard_deviation, 
			a->dim, d, n, block_n, block_size, toString(f), "tvm", a->lin.size, b->size, c->size);
}


void
measure_unfold_ht(
	void (*f)(), 
	const struct tensor_storage * restrict const a, 
	const struct lin_storage * restrict const b, 
	struct lin_storage * restrict const c, 
	const size_t d, FILE * const file, 
	const size_t n, const size_t block_n, 
	DTYPE * restrict const e, 
	buffer_t * restrict const buf, 
	const size_t block_size
) {
	(void) e;
	// ???????

	struct timespec start;
	struct timespec stop;
	double average_time = 0;
	double standard_deviation = 0;
	double results[TIMES];
	DTYPE checksum = 0;

	// Make a single "cold" call
	reset_array(c->data, c->size, 0);
	(void) memset(buf->unfold_1, 0, block_size*sizeof(DTYPE));
	(void) memset(buf->unfold_2, 0, block_size*sizeof(DTYPE));
	(*f)(a,b,c,d,NULL,buf);

	// Measure a sample runtime
	int rc1 = clock_gettime(CLOCK_MONOTONIC, &start);
	(*f)(a,b,c,d,NULL,buf);
	int rc2 = clock_gettime(CLOCK_MONOTONIC, &stop);
	if (rc1 != 0 || rc2 != 0) {
		fprintf(stderr, "Clock_gettime: Non-zero return code %d/%d from clock_gettime\n", rc1, rc2);
	}

	struct timespec a_diff = timespec_diff(start,stop);
	const double sample_time = timespec_to_ns(&a_diff);
	size_t iterations = (BILLION / sample_time) + 1;
	if (iterations == 0) {
		iterations = 1;
	}
	// printf("Sample time = %.17g, iterations = %zu\n",  timespec_to_microseconds(timespec_diff(start, stop)), iterations);

	for (int i=0; i<TIMES; ++i) {
		// This "trashes" the cache 
		reset_array(c->data, c->size, 0);
		(void) memset(buf->unfold_1, 0, block_size*sizeof(DTYPE));
		(void) memset(buf->unfold_2, 0, block_size*sizeof(DTYPE));
		// Sleep before having a single cold run
		// sleep(1);
		(*f)(a,b,c,d,NULL,buf);

		rc1 = clock_gettime(CLOCK_MONOTONIC, &start);
		for ( size_t count = 0; count < iterations; ++count ) {
			(*f)(a,b,c,d,NULL,buf);
		}
		rc2 = clock_gettime(CLOCK_MONOTONIC, &stop);
		//total_time contains the mini_result for a single run
		a_diff = timespec_diff(start,stop);
		const double total_time = (timespec_to_ns(&a_diff)) / ((double)iterations);
		//printf("Current meas: %f microsec\n", total_time);
		for( size_t k = 0; k < c->size; ++k ) {
			checksum += c->data[k];
		}
		printf( "Checksum at iteration %d: %f.\n", i, checksum );
		checksum = 0;
		//checksum also the unfold (For algorithms which ONLY? compute the unfold)
		for( size_t k = 0; k < block_size; ++k ) {
			checksum += buf->unfold_1[k];
			checksum += buf->unfold_2[k];
		}
		printf( "Checksum at iteration %d: %f.\n", i, checksum );
		average_time += total_time;
		results[i] = total_time;
		checksum = 0;
	}

	average_time /= (double)TIMES;
	for (int i=0; i<TIMES; ++i) {
		//printf("ave(%.2f) - result(%.2f): %.2f microsec\n", average_time, results[i], results[i] - average_time);
		standard_deviation += ((results[i]-average_time) * (results[i]-average_time));
	}

	// assert( TIMES > 1 );
	standard_deviation = standard_deviation / ((double)(TIMES-1));
	standard_deviation = sqrt(standard_deviation);
	write_perf_result(file, a->layout,
			average_time, standard_deviation, 
			a->dim, d, n, block_n, block_size, toString(f), "tvm", a->lin.size, b->size, c->size);
}


// returns the measured time (at least 1 second)
void
measure_tmm(
	void (*f)(), 
	const struct tensor_storage * restrict const a, 
	const struct lin_storage * restrict const b, 
	struct lin_storage * restrict const c, 
	const size_t d, FILE * const file, 
	const size_t n, const size_t block_n, 
	const size_t block_size,
	const int l_dimension,
	const int l_block_dimension
) {

	struct timespec start;
	struct timespec stop;
	double average_time = 0;
	double standard_deviation = 0;
	double results[TIMES];
	DTYPE checksum = 0;

	printf("cold run first\n");

	// Make a single "cold" call
	reset_array(c->data, c->size, 0);
	(*f)(a,b,c,d,l_dimension,l_block_dimension);

	// Measure a sample runtime
	int rc1 = clock_gettime(CLOCK_MONOTONIC, &start);
	(*f)(a,b,c,d,l_dimension,l_block_dimension);
	int rc2 = clock_gettime(CLOCK_MONOTONIC, &stop);
	if (rc1 != 0 || rc2 != 0) {
		fprintf(stderr, "Clock_gettime: Non-zero return code %d/%d from clock_gettime\n", rc1, rc2);
	}

	struct timespec a_diff = timespec_diff(start,stop);
	const double sample_time = timespec_to_ns(&a_diff);
	size_t iterations = (BILLION / sample_time) + 1;
	if (iterations == 0) {
		iterations = 1;
	}
	// printf("Sample time = %.17g, iterations = %zu\n",  timespec_to_microseconds(timespec_diff(start, stop)), iterations);

	for (int i=0; i<TIMES; ++i) {
		// This "trashes" the cache 
		reset_array(c->data, c->size, 0);
		// Sleep before having a single cold run
		sleep(1);
		// printf("so these are the params apssed: %d, %d\n", l_dimension, l_block_dimension);
		(*f)(a,b,c,d,l_dimension,l_block_dimension);

		rc1 = clock_gettime(CLOCK_MONOTONIC, &start);
		for ( size_t count = 0; count < iterations; ++count ) {
			(*f)(a,b,c,d,l_dimension,l_block_dimension);
		}
		rc2 = clock_gettime(CLOCK_MONOTONIC, &stop);
		//total_time contains the mini_result for a single run
		a_diff = timespec_diff(start,stop);
		const double total_time = (timespec_to_ns(&a_diff)) / ((double)iterations);
		//printf("Current meas: %f microsec\n", total_time);
		for( size_t k = 0; k < c->size; ++k ) {
			checksum += c->data[k];
		}
		// printf("adding %zu elements of c (result); for example c[10] = %f\n", c->size, c->data[10]);
		printf( "Checksum at iteration %d: %f.\n", i, checksum );
		average_time += total_time;
		results[i] = total_time;
		checksum = 0;
	}

	average_time /= (double)TIMES;
	for (int i=0; i<TIMES; ++i) {
		//printf("ave(%.2f) - result(%.2f): %.2f microsec\n", average_time, results[i], results[i] - average_time);
		standard_deviation += ((results[i]-average_time) * (results[i]-average_time));
	}

	// assert( TIMES > 1 );
	standard_deviation = standard_deviation / ((double)(TIMES-1));
	standard_deviation = sqrt(standard_deviation);
	write_tmm_perf_result(file, a->layout,
			average_time, standard_deviation, 
			a->dim, d, n, block_n, block_size, toString(f), "tmm", l_dimension, l_block_dimension);
}

// returns the measured time (at least 1 second)
void
measure_multicore(
	void (*f)(), 
	const struct tensor_storage * restrict const a, 
	const struct lin_storage * restrict const b, 
	struct lin_storage * restrict const c, 
	const size_t d, FILE * const file, 
	const size_t n, const size_t block_n, 
	const size_t block_size
) {
	struct timespec start;
	struct timespec stop;
	double average_time = 0;
	double standard_deviation = 0;
	double results[TIMES];
	DTYPE checksum = 0;
		
	if (c->data[0] != 0.0) { // If the global result array is non-empty, reset it
		reset_array(c->data, c->size, 0.0);
	}
	if (c->local_data[0][0] != 0.0) {
		reset_array_double_locally(c); // Result is local only for algorithm psync (when dim != 0)
	}

	// COLD CALL
	(*f)(a,b,c,d);

	// Measure a sample runtime
	int rc1 = clock_gettime(CLOCK_MONOTONIC, &start);
	(*f)(a,b,c,d);
	int rc2 = clock_gettime(CLOCK_MONOTONIC, &stop);
	if (rc1 != 0 || rc2 != 0) {
		fprintf(stderr, "Clock_gettime: Non-zero return code %d/%d from clock_gettime\n", rc1, rc2);
		assert(rc1 == 0);
		assert(rc2 == 0);
	}
	struct timespec a_diff = timespec_diff(start,stop);
	const double sample_time = timespec_to_ns(&a_diff);

	size_t iterations = (BILLION / sample_time) + 1;
	if (iterations == 0) {
		iterations = 1;
	}
	printf("INFO: Sample time = %f, yields %zu iterations in a second\n",  sample_time, iterations);

	for (int i=0; i<TIMES; ++i) {
		// This "trashes" the cache 
		if (c->data[0] != 0.0) { // If the global result array is non-empty, reset it
			reset_array(c->data, c->size, 0.0);
		}
		if (c->local_data[0][0] != 0.0) {
			reset_array_double_locally(c);
		}

		// Sleep before having a single cold run
		sleep(1);
		rc1 = clock_gettime(CLOCK_MONOTONIC, &start);
		(*f)(a,b,c,d);
		rc2 = clock_gettime(CLOCK_MONOTONIC, &stop);
		a_diff = timespec_diff(start,stop);
		printf( "INFO: Experiment warm-up time: %f.\n", timespec_to_ns(&a_diff));
		assert(rc1 == 0);
		assert(rc2 == 0);

		rc1 = clock_gettime(CLOCK_MONOTONIC, &start);
		for ( size_t count = 0; count < iterations; ++count ) {
			(*f)(a,b,c,d);
		}
		rc2 = clock_gettime(CLOCK_MONOTONIC, &stop);
		assert(rc1 == 0);
		assert(rc2 == 0);

		a_diff = timespec_diff(start,stop);
		const double total_time = (timespec_to_ns(&a_diff)) / ((double)iterations);
		//printf("Current meas: %f microsec\n", total_time);

		if (c->data[0] != 0.0) { // If the global result array is non-empty, reset it
			for( size_t k = 0; k < c->size; ++k ) {
				checksum += c->data[k];
			}
			printf( "Checksum at iteration %d: %f (iterations performed: %zu).\n", i, checksum, iterations);
		}
		if (c->local_data[0][0] != 0.0) {
			printf("Cleaning the local output!\n");
			for (int t=0; t<(a->lin.p_size)/20; ++t) {
				for (size_t k = 0; k < (c->size/a->lin.p_size)/10; ++k) {
					checksum += c->local_data[t][k];
				}
				printf("(At local result %d) Checksum at iteration %d: %f (iterations performed %zu).\n",
					t, i, checksum, iterations);
			}
		}
		printf("INFO: Total time taken by the experiment of %zu iterations (in ns): %f; Single runtime: %f.\n", iterations, timespec_to_ns(&a_diff), total_time);
		average_time += total_time;
		results[i] = total_time;
		checksum = 0;
	}

	average_time /= (double)TIMES;
	for (int i=0; i<TIMES; ++i) {
		//printf("ave(%.2f) - result(%.2f): %.2f microsec\n", average_time, results[i], results[i] - average_time);
		standard_deviation += ((results[i]-average_time) * (results[i]-average_time));
	}

	standard_deviation = standard_deviation / ((double)(TIMES-1));
	standard_deviation = sqrt(standard_deviation);

	// Here: compute p_array to be an actual array....
	size_t runtime_p_array[a->dim];
	for (int i=0; i<a->dim; ++i) {
		runtime_p_array[i] = a->layout[i] / a->layout2[i];
	}

	write_multicore_result(file, a->layout, a->p,
			average_time, standard_deviation, 
			a->dim, d, n, block_n, block_size, toString(f), "tvm_multicore", runtime_p_array, a->layout, a->block_layout, a->lin.p_size, a->lin.size, b->size, c->size);

}

// returns the measured time (at least 1 second)
void
measure(
	void (*f)(), 
	const struct tensor_storage * restrict const a, 
	const struct lin_storage * restrict const b, 
	struct lin_storage * restrict const c, 
	const size_t d, FILE * const file, 
	const size_t n, const size_t block_n, 
	const size_t block_size
) {
	struct timespec start;
	struct timespec stop;
	double average_time = 0;
	double standard_deviation = 0;
	double results[TIMES];
	DTYPE checksum = 0;

	// Make a single "cold" call
	reset_array(c->data, c->size, 0);
	(*f)(a,b,c,d);

	// Measure a sample runtime
	int rc1 = clock_gettime(CLOCK_MONOTONIC, &start);
	(*f)(a,b,c,d);
	int rc2 = clock_gettime(CLOCK_MONOTONIC, &stop);
	if (rc1 != 0 || rc2 != 0) {
		fprintf(stderr, "Clock_gettime: Non-zero return code %d/%d from clock_gettime\n", rc1, rc2);
	}

	struct timespec a_diff = timespec_diff(start,stop);
	const double sample_time = timespec_to_ns(&a_diff);
	size_t iterations = (BILLION / sample_time) + 1;
	if (iterations == 0) {
		iterations = 1;
	}
	// printf("Sample time = %.17g, iterations = %zu\n",  timespec_to_microseconds(timespec_diff(start, stop)), iterations);

	for (int i=0; i<TIMES; ++i) {
		// This "trashes" the cache 
		reset_array(c->data, c->size, 0);
		// Sleep before having a single cold run
		sleep(1);
		(*f)(a,b,c,d);

		rc1 = clock_gettime(CLOCK_MONOTONIC, &start);
		for ( size_t count = 0; count < iterations; ++count ) {
			(*f)(a,b,c,d);
		}
		rc2 = clock_gettime(CLOCK_MONOTONIC, &stop);
		struct timespec warmup_diff = timespec_diff(start,stop);
		printf( "Warm up runtime: %f.\n", timespec_to_ns(&warmup_diff));
		
		//total_time contains the mini_result for a single run
		a_diff = timespec_diff(start,stop);
		const double total_time = (timespec_to_ns(&a_diff)) / ((double)iterations);
		//printf("Current meas: %f microsec\n", total_time);
		for( size_t k = 0; k < c->size; ++k ) {
			checksum += c->data[k];
		}
		printf( "Checksum at iteration %d: %f (iterations performed: %zu).\n", i, checksum, iterations);

		printf( "Total time taken by the experiment (in ns): %f; Single runtime: %f.\n", timespec_to_ns(&a_diff), total_time);
		average_time += total_time;
		results[i] = total_time;
		checksum = 0;
	}

	average_time /= (double)TIMES;
	for (int i=0; i<TIMES; ++i) {
		//printf("ave(%.2f) - result(%.2f): %.2f microsec\n", average_time, results[i], results[i] - average_time);
		standard_deviation += ((results[i]-average_time) * (results[i]-average_time));
	}

	// assert( TIMES > 1 );
	standard_deviation = standard_deviation / ((double)(TIMES-1));
	standard_deviation = sqrt(standard_deviation);
	write_perf_result(file, a->layout,
			average_time, standard_deviation, 
			a->dim, d, n, block_n, block_size, toString(f), "tvm", a->lin.size, b->size, c->size);
}


void
measure_unfold_mem(
	void (*f)(), 
	const struct lin_storage * restrict const a, 
	const struct tensor_storage * restrict const b,
	const int unblock,
	const size_t d, FILE * const file, 
	const size_t n, const size_t block_n, 
	const size_t block_size
) {

	(void) unblock; 
	// ???????

	struct timespec start;
	struct timespec stop;
	double average_time = 0;
	double standard_deviation = 0;
	double results[TIMES];
	DTYPE checksum = 0;

	// Make a single "cold" call
	reset_array(a->data, a->size, 0);
	(*f)(a,b,0);

	// Measure a sample runtime
	int rc1 = clock_gettime(CLOCK_MONOTONIC, &start);
	(*f)(a,b,0);
	int rc2 = clock_gettime(CLOCK_MONOTONIC, &stop);
	if (rc1 != 0 || rc2 != 0) {
		fprintf(stderr, "Clock_gettime: Non-zero return code %d/%d from clock_gettime\n", rc1, rc2);
	}

	struct timespec a_diff = timespec_diff(start,stop);
	const double sample_time = timespec_to_ns(&a_diff);
	size_t iterations = (BILLION / sample_time) + 1;
	if (iterations == 0) {
		iterations = 1;
	}
	// printf("Sample time = %.17g, iterations = %zu\n",  timespec_to_microseconds(timespec_diff(start, stop)), iterations);

	for (int i=0; i<TIMES; ++i) {
		// This "trashes" the cache 
		reset_array(a->data, a->size, 0);
		// Sleep before having a single cold run
		sleep(1);
		(*f)(a,b,0);

		rc1 = clock_gettime(CLOCK_MONOTONIC, &start);
		for ( size_t count = 0; count < iterations; ++count ) {
			(*f)(a,b,0);
		}
		rc2 = clock_gettime(CLOCK_MONOTONIC, &stop);
		//total_time contains the mini_result for a single run
		a_diff = timespec_diff(start,stop);
		const double total_time = (timespec_to_ns(&a_diff)) / ((double)iterations);
		//printf("Current meas: %f microsec\n", total_time);
		for( size_t k = 0; k < a->size; ++k ) {
			checksum += a->data[k];
		}
		printf( "Checksum at iteration %d: %f.\n", i, checksum );
		average_time += total_time;
		results[i] = total_time;
		checksum = 0;
	}

	average_time /= (double)TIMES;
	for (int i=0; i<TIMES; ++i) {
		//printf("ave(%.2f) - result(%.2f): %.2f microsec\n", average_time, results[i], results[i] - average_time);
		standard_deviation += ((results[i]-average_time) * (results[i]-average_time));
	}

	// assert( TIMES > 1 );
	standard_deviation = standard_deviation / ((double)(TIMES-1));
	standard_deviation = sqrt(standard_deviation);
	write_perf_result(file, b->layout,
			average_time, standard_deviation, 
			b->dim, d, n, block_n, block_size, toString(f), "tvm", a->size, b->lin.size, b->lin.size);
}


// returns the measured time (at least 1 second)
void
measure_libx(
	void (*f)(), 
	const struct tensor_storage * restrict const a, 
	const struct lin_storage * restrict const b, 
	struct lin_storage * restrict const c, 
	const size_t d, FILE * const file, 
	const size_t n, const size_t block_n, 
	const size_t block_size,
	libxsmm_dmmfunction * const kernel
) {
	struct timespec start;
	struct timespec stop;
	double average_time = 0;
	double standard_deviation = 0;
	double results[TIMES];
	DTYPE checksum = 0;

	// Make a single "cold" call
	reset_array(c->data, c->size, 0);
	(*f)(a,b,c,d, kernel);

	// Measure a sample runtime
	int rc1 = clock_gettime(CLOCK_MONOTONIC, &start);
	(*f)(a,b,c,d, kernel);
	int rc2 = clock_gettime(CLOCK_MONOTONIC, &stop);
	if (rc1 != 0 || rc2 != 0) {
		fprintf(stderr, "Clock_gettime: Non-zero return code %d/%d from clock_gettime\n", rc1, rc2);
	}

	struct timespec a_diff = timespec_diff(start,stop);
	const double sample_time = timespec_to_ns(&a_diff);
	size_t iterations = (BILLION / sample_time) + 1;
	if (iterations == 0) {
		iterations = 1;
	}
	// printf("Sample time = %.17g, iterations = %zu\n",  timespec_to_microseconds(timespec_diff(start, stop)), iterations);

	for (int i=0; i<TIMES; ++i) {
		// This "trashes" the cache 
		reset_array(c->data, c->size, 0);
		// Sleep before having a single cold run
		sleep(1);
		(*f)(a,b,c,d, kernel);

		rc1 = clock_gettime(CLOCK_MONOTONIC, &start);
		for ( size_t count = 0; count < iterations; ++count ) {
			(*f)(a,b,c,d, kernel);
		}
		rc2 = clock_gettime(CLOCK_MONOTONIC, &stop);
		//total_time contains the mini_result for a single run
		a_diff = timespec_diff(start,stop);
		const double total_time = (timespec_to_ns(&a_diff)) / ((double)iterations);
		//printf("Current meas: %f microsec\n", total_time);
		for( size_t k = 0; k < c->size; ++k ) {
			checksum += c->data[k];
		}
		printf( "Checksum at iteration %d: %f.\n", i, checksum );
		average_time += total_time;
		results[i] = total_time;
		checksum = 0;
	}

	average_time /= (double)TIMES;
	for (int i=0; i<TIMES; ++i) {
		//printf("ave(%.2f) - result(%.2f): %.2f microsec\n", average_time, results[i], results[i] - average_time);
		standard_deviation += ((results[i]-average_time) * (results[i]-average_time));
	}

	// assert( TIMES > 1 );
	standard_deviation = standard_deviation / ((double)(TIMES-1));
	standard_deviation = sqrt(standard_deviation);
	write_perf_result(file, a->layout,
			average_time, standard_deviation, 
			a->dim, d, n, block_n, block_size, toString(f), "tvm", a->lin.size, b->size, c->size);
}



void
write_header(FILE * file_handle) {
	// Write CPU info in the file
	// FILE *cpuinfo = fopen("/proc/cpuinfo", "rb");
 //   	size_t size = 0;
 //   	char *arg = 0;
 //   	while (getdelim(&arg, &size, 0, cpuinfo) != -1) {
	// 	fprintf(file_handle, "%s\n", arg);
 //   		free(arg);
 //   		fclose(cpuinfo);
	// // }
	// char hostname[1024];
	// gethostname(hostname, 1024);
	// fprintf(file_handle, "hostname=%s\n", hostname);
	fprintf(file_handle, HEADER);
	if (ferror(file_handle)) {
		printf("Error writing <header>.\n");
	}	
}

void
write_tmm_header(FILE * file_handle) {
	fprintf(file_handle, HEADERTMM);
	if (ferror(file_handle)) {
		printf("Error writing <header>.\n");
	}	
}

void
write_powermethod_header(FILE * file_handle) {
	fprintf(file_handle, HEADERPM);
	if (ferror(file_handle)) {
		printf("Error writing <header>.\n");
	}	
}

// returns the measured time (at least 1 second)
void
measure_powermethod(
	void (*f)(), 
	const struct tensor_storage * restrict const tensor, 
	struct lin_storage * restrict * restrict const vectors, 
	struct lin_storage * restrict * restrict const model_vectors, 
	struct lin_storage * restrict const result_1,
	struct lin_storage * restrict const result_2, 
	const int iters,
	FILE * const file,
	const size_t n, const size_t block_n,
	const size_t block_size
) {

	(void) block_size;
	
	struct timespec start;
	struct timespec stop;
	double average_time = 0;
	double results[TIMES] = {0}; // Initialize in case we cut through TIMES 
	DTYPE checksum = 0;

	double standard_deviation = 0.0;

	// printf("Cold run first\n");
	// Make a single "cold" call
	reset_array(result_1->data, result_1->size, 0.0);
	reset_array(result_2->data, result_2->size, 0.0);
	for (size_t i=0; i<tensor->dim; ++i) {
			// printf("Resetting the vectors of size %zu!\n", tensor->layout[i]);
		for (size_t j=0; j<tensor->layout[i]; ++j) {
			vectors[i]->data[j] = model_vectors[i]->data[j];
		}
	}
	// Cold run -> 1 iter is enough
	(*f)(tensor, vectors, result_1, result_2, 1);

	// Measure a sample runtime
	int rc1 = clock_gettime(CLOCK_MONOTONIC, &start);
	(*f)(tensor, vectors, result_1, result_2, iters);
	int rc2 = clock_gettime(CLOCK_MONOTONIC, &stop);
	if (rc1 != 0 || rc2 != 0) {
		fprintf(stderr, "Clock_gettime: Non-zero return code %d/%d from clock_gettime\n", rc1, rc2);
	}
	
	struct timespec a_diff = timespec_diff(start,stop);
	const double sample_time = timespec_to_ns(&a_diff);
	size_t iterations = (BILLION / sample_time) + 1;
	if (iterations == 0) {
		iterations = 1;
	}
	printf("Sample time = %.17g, iterations = %zu\n",  timespec_to_microseconds(timespec_diff(start, stop)), iterations);
	
	// CALLGRIND_START_INSTRUMENTATION;
	// CALLGRIND_TOGGLE_COLLECT;
	
	for (int i=0; i<TIMES; ++i) {
		printf("Iteration %d\n", i);
		// This "trashes" the cache 
		reset_array(result_1->data, result_1->size, 0.0);
		reset_array(result_2->data, result_2->size, 0.0);
		for (size_t j=0; j<tensor->dim; ++j) {
			for (size_t k=0; k<tensor->layout[j]; ++k) {
				vectors[j]->data[k] = model_vectors[j]->data[k];
			}
		}

		// Sleep before having a single cold run
		sleep(1);

		// Warm-up run(!) Check that timings are different?
		rc1 = clock_gettime(CLOCK_MONOTONIC, &start);
		(*f)(tensor, vectors, result_1, result_2, 1);
		rc2 = clock_gettime(CLOCK_MONOTONIC, &stop);
		struct timespec warmup_diff = timespec_diff(start,stop);
		printf( "Warm up runtime: %f.\n", timespec_to_ns(&warmup_diff));

		rc1 = clock_gettime(CLOCK_MONOTONIC, &start);
		for ( size_t count = 0; count < iterations; ++count ) {
			(*f)(tensor, vectors, result_1, result_2, iters);
		}
		rc2 = clock_gettime(CLOCK_MONOTONIC, &stop);

		//total_time contains the mini_result for a single run
		a_diff = timespec_diff(start,stop);
		const double total_time = (timespec_to_ns(&a_diff)) / ((double)iterations);
		//printf("Current meas: %f microsec\n", total_time);

		for (size_t j=0; j<tensor->dim; ++j) {
			for (size_t k=0; k<tensor->layout[j]; ++k) {
				checksum += vectors[j]->data[k];
			}
			printf( "Checksum of vector %zu: %f.\n", j, checksum);
			checksum = 0;
		}

		printf( "Total time taken by the experiment (in ns): %f; Single runtime: %f.\n", timespec_to_ns(&a_diff), total_time);
		average_time += total_time;
		results[i] = total_time;
	}

	// CALLGRIND_TOGGLE_COLLECT;
	// CALLGRIND_STOP_INSTRUMENTATION;

	average_time /= (double)TIMES;
	for (int i=0; i<TIMES; ++i) {
		//printf("ave(%.2f) - result(%.2f): %.2f microsec\n", average_time, results[i], results[i] - average_time);
		standard_deviation += ((results[i]-average_time) * (results[i]-average_time));
	}
	// assert(TIMES > 1);
	
	standard_deviation = standard_deviation / ((double)(TIMES-1));
	standard_deviation = sqrt(standard_deviation);
	write_pm_perf_result(file, tensor->layout,
		average_time, standard_deviation, 
		tensor->dim, n, block_n, 0, toString(f), "pm");
	
}
