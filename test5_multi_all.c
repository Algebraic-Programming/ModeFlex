#include<algorithms.h>
#include<gen_utils.h> // for reset_array_sizet
#include<gen_data.h> // get_vector, gen_vector, gen_block_tensor
#include<file_utils.h> // for save_to_file
#include<test.h>
#include<stdlib.h> // for free

int test5_multi_all(int argc, char ** argv) {

	int dim_min, dim_max, n_min, n_max;
	int mode_min, mode_max;
	int block_n_min, block_n_max;

	// we must provide default arguments
	dim_min = 3;
	dim_max = 3;
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

	if (dim_max != 3) {
		mode_max = dim_max - 1;
	}

	printf("int dim_min=%d\n", dim_min);
	printf("int dim_max=%d\n", dim_max);
	printf("int mode_min=%d\n", mode_min);
	printf("int mode_max=%d\n", mode_max);
	printf("N RANDOMIZED!\n");
	printf("BLOCK_N RANDOMIZED!\n");

	char filename[BUFSIZE];
	char filename2[BUFSIZE];

	typedef void (*TVM)();

	// Set bounds for each parameter for testing (expressed as a loop)
	// Params: dim, mode, n, block_n

	// for now: do not test blockmode

	// model algorithm
	TVM model_algorithm = tvm_tensor_major;

	TVM unfold_unfold_algorithms[] = {
		tvm_vector_major,
		tvm_output_major,
		tvm_block_major,
		tvm_vector_major_BLAS_col, // this computes an unfold
		tvm_output_major_BLAS_row, // this computes an unfold
		tvm_vector_major_BLAS_col_mode,
		tvm_vector_major_BLAS_col_mode,

		// require k_leftmost
		tvm_vector_major_input_aligned,
		tvm_vector_major_BLAS_col_BLAS,
		tvm_BLIS_col,

		// require k_rightmost
		tvm_output_major_input_aligned,
		tvm_output_major_BLAS_row_BLAS, // this computes an unfold
		tvm_BLIS_row,

		// block
		tvm_block_major_input_aligned,
		tvm_block_major_input_aligned_output_aligned,
		
		// morton block
		tvm_morton_block_major_input_aligned,
		tvm_morton_block_major_input_aligned_output_aligned,

		// blockmode
		tvm_blockmode_major_input_aligned,
		tvm_blockmode_major_input_aligned_output_aligned,
		tvm_blockmode_major_BLAS_input_aligned_output_aligned
	};

	// parameters' loops ordered according to their dependency
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

			// keep this loop: these set a max for the actual array of n
			for (size_t n=(size_t) n_min; n<=(size_t) n_max; ++n) {			

				// HERE: we randomize the array of tensor_layout rather than go through it
				randomize_array_int(tensor_layout, dim, n);
				printf("        n = ");
				print_to_console_sizet(tensor_layout, dim);

				size_t min_n = tensor_layout[0];
				// find minimum over the array
				for (size_t i=1; i<dim; ++i) {
					if (tensor_layout[i] < min_n) {
						min_n = tensor_layout[i];
					}
				}

				// put block_n for a dim as random(1,tensor_layout[dim])
				randomize_array_int_from_array(block_layout, dim, tensor_layout);
				printf("            block_n = ");
				print_to_console_sizet(block_layout, dim);

				size_t block_size = 1;
				for (size_t d=0; d<dim; ++d) {
					block_size *= block_layout[d];
				}
				DTYPE * const restrict unfold = calloc(block_size, sizeof(DTYPE));

				// allocate tensor,vector,result on the stack
				struct tensor_storage *tensor = gen_block_tensor(dim, tensor_layout, block_layout);
				struct lin_storage *vector = gen_vector(tensor->layout[mode]);
				struct tensor_storage  *result = get_block_result_tensor(tensor, mode);
				struct tensor_storage *model_result = get_block_result_tensor(tensor, mode);		

				model_algorithm(tensor, vector, &model_result->lin, mode);

				int out_algo = -1;
				int algo_counter = 0;

				size_t n = 0;
				size_t block_n = 0;

				//////////////////////////////////////////////////////////////////// UNFOLD

				out_algo = test_algorithms(unfold_unfold_algorithms, algo_counter, 7, &result->lin, model_result->lin.data, tensor, vector, mode,
						filename, filename2, dim, n, block_n, out_algo, unfold, NULL, NULL);
				algo_counter += 7;


				struct tensor_storage *tensor_k_leftmost = get_in_out_unfold(tensor, 0, mode);
				out_algo = test_algorithms(unfold_unfold_algorithms, algo_counter, 3, &result->lin, model_result->lin.data, tensor_k_leftmost, vector, mode,
						filename, filename2, dim, n, block_n, out_algo, unfold, NULL, NULL);
				free_tensor_storage(tensor_k_leftmost);
				algo_counter += 3;

				struct tensor_storage *tensor_k_rightmost = get_in_out_unfold(tensor, 1, mode);
				out_algo = test_algorithms(unfold_unfold_algorithms, algo_counter, 3, &result->lin, model_result->lin.data, tensor_k_rightmost, vector, mode,
						filename, filename2, dim, n, block_n, out_algo, unfold, NULL, NULL);
				free_tensor_storage(tensor_k_rightmost);
				algo_counter += 3;

				//////////////////////////////////////////////////////////////////// BLOCK

				struct tensor_storage *blocked_tensor = get_block_tensor(tensor, 0, 0);
				out_algo = test_algorithms(unfold_unfold_algorithms, algo_counter, 1, &result->lin, model_result->lin.data, blocked_tensor, vector, mode,
						filename, filename2, dim, n, block_n, out_algo, unfold, NULL, NULL);
				algo_counter += 1;
				struct tensor_storage *unblocked_result = get_block_tensor(model_result, 0, 0);
				out_algo = test_algorithms(unfold_unfold_algorithms, algo_counter, 1, &result->lin, unblocked_result->lin.data, blocked_tensor, vector, mode,
						filename, filename2, dim, n, block_n, out_algo, unfold, NULL, NULL);
				algo_counter += 1;
				free_tensor_storage(unblocked_result);
				free_tensor_storage(blocked_tensor);

				//////////////////////////////////////////////////////////////////// MORTON

				struct tensor_storage *morton_blocked_tensor = get_block_tensor(tensor, 0, 1);
				out_algo = test_algorithms(unfold_unfold_algorithms, algo_counter, 1, &result->lin, model_result->lin.data, morton_blocked_tensor, vector, mode,
						filename, filename2, dim, n, block_n, out_algo, unfold, NULL, NULL);
				algo_counter += 1;
				struct tensor_storage *morton_unblocked_result = get_block_tensor(model_result, 0, 1);
				out_algo = test_algorithms(unfold_unfold_algorithms, algo_counter, 1, &result->lin, morton_unblocked_result->lin.data, morton_blocked_tensor, vector, mode,
						filename, filename2, dim, n, block_n, out_algo, unfold, NULL, NULL);
				algo_counter += 1;
				free_tensor_storage(morton_unblocked_result);
				free_tensor_storage(morton_blocked_tensor);

				//////////////////////////////////////////////////////////////////// BLOCKMODE
				// BLOCKMODE -> destructive for the model_result, hence commented out

#if 0
				struct tensor_storage *blockmode_tensor = get_blockmode_tensor(tensor, mode, 0); // block 
				out_algo = test_algorithms(unfold_unfold_algorithms, algo_counter, 1, &result->lin, model_result->lin.data, blockmode_tensor, vector, mode,
						filename, filename2, dim, n, block_n, out_algo, unfold, NULL, NULL);
				algo_counter += 1;
				qsort(model_result->lin.data, model_result->lin.size, sizeof(DTYPE), compare);
				out_algo = test_algorithms(unfold_unfold_algorithms, algo_counter, 2, &result->lin, model_result->lin.data, blockmode_tensor, vector, mode,
						filename, filename2, dim, n, block_n, out_algo, unfold, NULL, NULL);
				algo_counter += 2;
				free_tensor_storage(blockmode_tensor);
#endif

				//////////////

				if (out_algo != -1 && DUMP) {
					snprintf(filename, BUFSIZE, "%zu %zu %zu %d", dim, mode, n, -1);
					SAVE(model_result->lin);
				}

				free(unfold);
				free_tensor_storage(model_result);
				free_tensor_storage(result);
				free_lin_storage(vector);
				free_tensor_storage(tensor);
			}
		}
	}

	return 0;
}








