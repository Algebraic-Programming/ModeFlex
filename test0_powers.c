#define _GNU_SOURCE
#include <algorithms.h>
#include <gen_utils.h> // for randomize array int 
#include <gen_data.h> // get_vector, gen_vector, gen_block_tensor
#include <file_utils.h> // for save_to_file
#include <test.h> // for inline functions
#include <rand_utils.h>
#include <stdlib.h> // for free
#include <string.h>
#include <sched.h>
#include <unistd.h>
#include <pthread.h>
#include <stdint.h>
#include <sys/types.h>
#include <signal.h>
#include <errors.h>

#define TOTAL_RUNS 30

int test0_powers(int argc, char ** argv) {

	// MPI_Init(NULL, NULL);
 //    int world_size;
 //    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
 //    int my_rank;
 //    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
 //    printf("Hello from processor of rank %d!\n", my_rank);

	int dim_min, dim_max, n_min, n_max;
	int mode_min, mode_max;
	int block_n_min, block_n_max;

	// we must provide default arguments
	dim_min = 5;
	dim_max = 5;
	mode_min = 0;
	mode_max = dim_max-1;
	n_min = 1;
	n_max = 64;
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

	mode_max = dim_max - 1;

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

	TVM model_algorithm = tvm_tensor_major;

	// improvement: could include the numbered versions for completeness
	TVM unfold_unfold_algorithms[] = {

		// 5+1 = 6
		tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold,
		tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_v3,
		tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_v3_mine,
		tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_v4,
		// tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_mine,
		tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_v3_libx,
		tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_intorow,
		tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_intorow2,
		// tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_intocol_linin,
		// tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_intocol_linin2,
		// tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_intocol_linin3,
		// tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_intocol_linout,

		// 2 -> they just copy so they only work for mode=0
		// tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_bench,

		// 1 -> works only for leftmost mode (copy; then do col-major)
		// tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_mode1,

		// 1
		// tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_v2_unfold,

		///////////////////// MORTON
		tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_libx_result,
		tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_libx_tensor,
		tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_libx_vector,
		tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_libx_result2,
		tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_libx_tensor2,
		tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_libx_vector2,
		// tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_libx_tensor3,
		// tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_libx_vector3,
		tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_libx,
		tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_mklgemm,
		tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_mklgemminlined,
		
		// tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_libx_tensor,
		tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_libx_result,
		tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3,

		// 4
		tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_2_unfold,
		tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3,
		tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_mine,
		tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_4,
		//tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_old,
		
		// 1
		tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_bench,
		tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal,

		// 1
		tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold_mine_nontemporal_mode1,

		// 1
		tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_unfold

	};

	int count[] = {
		7,0,0,0,
		9,0,0,0
		// 5,2,1,1
	};

	for (int runs=1; runs<TOTAL_RUNS; ++runs) {

		printf("runs=%d\n", runs);

	for (size_t dim=(size_t) dim_min; dim<=(size_t) dim_max; ++dim) {
		printf("dim=%zu:\n", dim);
		size_t block_layout[dim];
		size_t tensor_layout[dim];

		//for (int mode=mode_min; mode<=temp_mode_max; ++mode) {
		for (size_t mode=mode_min; mode <= (size_t) mode_max; ++mode) {

			printf("    mode=%zu:\n", mode);

			randomize_array_int(block_layout, dim, 1);
			// randomize_array_int(block_layout, dim, rand_int(1,rand_int(runs, runs)));
			// reset_array_sizet(block_layout, dim, 8);
			printf("            block_layout = ");
			print_to_console_sizet(block_layout, dim);

			
			// make tensor layout a multiple of that
			size_t block_size = 1;
			size_t mat_size = 1;
			for (size_t d=dim-1; d<dim; --d) {
				if (d > mode) {
					mat_size *= block_layout[d];
				}
			}
			mat_size *= block_layout[mode];

			// int tensor_ex[][3] = {{5,5,22}};
			for (size_t d=0; d<dim; ++d) {
				// if (d==1) {
				// 	tensor_layout[d] = block_layout[d] * 8;		
				// } else {
				// 	tensor_layout[d] = block_layout[d] * 2;		
				// }
				tensor_layout[d] = block_layout[d] * 10;
				// tensor_layout[d] = block_layout[d] * tensor_ex[0][d];
				// tensor_layout[d] = block_layout[d] * 5;
				block_size *= block_layout[d];
			}

			DTYPE * unfold_small = get_aligned_memory(sizeof(DTYPE) * mat_size, ALIGNMENT_BLOCK);
			memset(unfold_small, 0, mat_size);
			DTYPE * unfold = get_aligned_memory(sizeof(DTYPE) * block_size, ALIGNMENT_BLOCK);
			memset(unfold, 0, block_size);
			
			// printf("BLOCK SIZE=%d\n", block_size);
			// printf("right size=%d\n", mat_size);

			printf("            tensor_layout = ");
			print_to_console_sizet(tensor_layout, dim);

			//////////////////////////////////////////////

			struct tensor_storage *tensor = gen_block_tensor(dim, tensor_layout, block_layout);
			struct lin_storage *vector = gen_vector(tensor->layout[mode]);

			struct tensor_storage  *result = get_block_result_tensor(tensor, mode);
			struct tensor_storage *model_result = get_block_result_tensor(tensor, mode);		
			// Perform the model algorithm TMV
			model_algorithm(tensor, vector, &model_result->lin, mode);

			DTYPE * const sorted_result = malloc(model_result->lin.size*sizeof(DTYPE));// * sizeof(DTYPE));
			memcpy(sorted_result, model_result->lin.data, model_result->lin.size*sizeof(DTYPE));
			qsort(sorted_result, model_result->lin.size, sizeof(DTYPE), compare);

			////////////////////////////////////////////// ALL OF THEM ARE POWERS versions
			int out_algo = -1;
			int algo_counter = 0;

			size_t n = 0;
			size_t block_n = 0;
			
			//////////////////////////////////////////////////////////////////// BLOCK
					
			struct tensor_storage *blocked_tensor = get_block_tensor(tensor, 0, 0);
			struct tensor_storage *unblocked_result = get_block_tensor(model_result, 0, 0);

			printf("First block algorithms (%d)\n", count[0]);
			out_algo = test_algorithms(unfold_unfold_algorithms, algo_counter, count[0], &result->lin, unblocked_result->lin.data, blocked_tensor, vector, mode,
				filename, filename2, dim, n, block_n, out_algo, unfold, NULL, NULL);
			algo_counter += count[0];

			printf("mode==dim-1 block algorithms (%d)\n", count[1]);
			if (mode == dim-1) {
				out_algo = test_algorithms(unfold_unfold_algorithms, algo_counter, count[1], &result->lin, unblocked_result->lin.data, blocked_tensor, vector, mode,
					filename, filename2, dim, n, block_n, out_algo, unfold, NULL, NULL);
			}
			algo_counter += count[1];

			// if (vector->size % 4 == 0) {
			// 	out_algo = test_algorithms(unfold_unfold_algorithms, algo_counter, count[1]+1, &result->lin, unblocked_result->lin.data, blocked_tensor, vector, mode,
			// 		filename, filename2, dim, n, block_n, out_algo, unfold, NULL, NULL);
			// }
			// algo_counter += 1;

			printf("Mode==0 block algorithms (%d)\n", count[2]);
			if (mode == 0) {
				out_algo = test_algorithms(unfold_unfold_algorithms, algo_counter, count[2], &result->lin, unblocked_result->lin.data, blocked_tensor, vector, mode,
					filename, filename2, dim, n, block_n, out_algo, unfold_small, NULL, NULL);
			}
			algo_counter += count[2];

			// Use sorted_result for comparison here
			printf("Sorted block algorithms (%d)\n", count[3]);
			out_algo = test_algorithms(unfold_unfold_algorithms, algo_counter, count[3], &result->lin, sorted_result, blocked_tensor, vector, mode,
				filename, filename2, dim, n, block_n, out_algo, unfold, NULL, NULL);
			algo_counter += count[3];

			free_tensor_storage(unblocked_result);
			free_tensor_storage(blocked_tensor);

			//////////////////////////////////////////////////////////////////// MORTON

			struct tensor_storage *morton_blocked_tensor = get_block_tensor(tensor, 0, 1);
			struct tensor_storage *morton_unblocked_result = get_block_tensor(model_result, 0, 1);

			printf("Morton block algorithms (%d)\n", count[4]);
			out_algo = test_algorithms(unfold_unfold_algorithms, algo_counter, count[4], &result->lin, morton_unblocked_result->lin.data, morton_blocked_tensor, vector, mode,
					filename, filename2, dim, n, block_n, out_algo, unfold, NULL, NULL);
			algo_counter += count[4];

			printf("mode==dim-1 morton block algorithms (%d)\n", count[5]);
			if (mode == dim-1) {
				out_algo = test_algorithms(unfold_unfold_algorithms, algo_counter, count[5], &result->lin, morton_unblocked_result->lin.data, morton_blocked_tensor, vector, mode,
					filename, filename2, dim, n, block_n, out_algo, unfold, NULL, NULL);
			}
			algo_counter += count[5];

			printf("mode==0 morton block algorithms (%d)\n", count[5]);
			if (mode == 0) {
				out_algo = test_algorithms(unfold_unfold_algorithms, algo_counter, count[6], &result->lin, morton_unblocked_result->lin.data, morton_blocked_tensor, vector, mode,
					filename, filename2, dim, n, block_n, out_algo, unfold_small, NULL, NULL);
			}
			algo_counter += count[6];

			out_algo = test_algorithms(unfold_unfold_algorithms, algo_counter, count[7], &result->lin, sorted_result, morton_blocked_tensor, vector, mode,
				filename, filename2, dim, n, block_n, out_algo, unfold, NULL, NULL);
			algo_counter += count[7];

			free_tensor_storage(morton_unblocked_result);
			free_tensor_storage(morton_blocked_tensor);

			////////////////////////////////////////////// 

			if (out_algo != -1 && DUMP) {
				snprintf(filename, BUFSIZE, "%zu %zu %d", dim, mode, -1);
				SAVE(model_result->lin);
				exit(-1);
			}

			free(unfold);
			free(unfold_small);
			free_tensor_storage(tensor);
			free_tensor_storage(result);
			free_tensor_storage(model_result);
			free_lin_storage(vector);

		}
	}

	}

	return 0;
}

