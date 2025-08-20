// HT requirements
#define _GNU_SOURCE

#include <string.h> // for memcmp
#include <assert.h>
#include <stdlib.h> // for free for _SC_NPROCESSORS_ONLN for gethostname
#include <math.h> // for pow, round

#include <algorithms.h>
#include <gen_utils.h> // for reset_array_sizet
#include <gen_data.h> // get_vector, gen_vector, gen_block_tensor
#include <file_utils.h> // for save_to_file
#include <unistd.h> // 
// #include <test.h>
#include <bench_utils.h>

#define TEST_ENV 1

// test case: add some code which appears only in tests
// test case: make numbers round (or integers!)

// These are all numbers of doubles
#define L1 4096.0 // 32 KB
#define L1_measured 2048.0
#define L3 3276800.0 
#define L3_measured 2097152.0 // 16 MB of L3 taken for the measurement (not half...)

#define TEST_SIZE 4096.0

#define FILENAME(x); snprintf(filename, BUFSIZE, "%s/%s_test_file.csv", RESULTS_FOLDER, hostname);

// also defined in test header
#define epsilon 1e-6
inline int fequal(double a, double b) {
	return fabs(a-b) < epsilon;
}
extern int compare( const void* _a, const void* _b);

#ifndef bitmask_t_
#define bitmask_t_
#define bitmask_t 	unsigned long long
#endif

int test_tmm(int argc, char ** argv) {

	int dim_min = 2;
	int dim_max = 2;
	int l_min = 2;
	int l_max = 2;

	switch (argc) {
		case 4:
			// l
			sscanf (*(argv+argc--), "%d", &l_max);
			sscanf (*(argv+argc--), "%d", &l_min);
			// dim
			// read from end to begin?
			sscanf (*(argv+argc--), "%d", &dim_max);
			sscanf (*(argv+argc--), "%d", &dim_min);

	}

	// Implement basic tmm test case

	// Get a matrix instead of a vector, but for now treat matrix as a simply "long" vector?

	// Make it flexible, we want to move through all possible sizes of k,l,etc

	// Get the file anyway
	// char hostname[1024];
	// gethostname(hostname, 1024);
	// char filename[BUFSIZE];
	// struct timespec time;
	// clock_gettime(CLOCK_MONOTONIC, &time);
	// FILENAME("results");
	// printf("filename=%s\n", filename);
	// FILE * file = fopen(filename, "w"); // r+ if you want to update without deleteing it
	// if (file == NULL) {
	// 	perror("Error opening file.\n");
	// }
	// write_header(file);

	size_t max_length = TEST_SIZE;
	// Assume 1D(!)
	// gen_block_tensor_clean - copies the content instead of using the pointer(!) of max_length
	struct tensor_storage *tensor = gen_block_tensor_clean(1, &max_length, &max_length);
	struct tensor_storage  *result = gen_block_tensor_clean(1, &max_length, &max_length);

	// Get 2D storage for matrices
	struct tensor_storage *secondary_matrix = gen_block_tensor_clean(1, &max_length, &max_length);
	size_t matrix_layout[2];
	size_t matrix_block_layout[2];
	size_t matrix_layout_perm[2] = {0,1};
	secondary_matrix->dim = 2;
	secondary_matrix->layout = copy_array_int(matrix_layout,2); // Important: we cannot free() this memory otherwise
	secondary_matrix->block_layout = copy_array_int(matrix_block_layout,2); // Important: we cannot free() this memory otherwise
	secondary_matrix->layout_perm = copy_array_int(matrix_layout_perm,2); // Important: we cannot free() this memory otherwise

	// still left: layout, block_layout, size(!)
	// POTENTIAL ERROR: same seed for tensor and the vector(!)
	struct lin_storage *matrix = gen_vector(max_length);
	struct lin_storage *temp_matrix = gen_vector(max_length);
	for (int i=0; i<matrix->size; ++i) {
		secondary_matrix->lin.data[i] = (int) matrix->data[i];
		temp_matrix->data[i] = (int) matrix->data[i];
	}
	struct lin_storage *unblocked_result = gen_vector(max_length);

	if (TEST_ENV == 1) {
		round_numbers(tensor->lin.data, tensor->lin.size);
		round_numbers(matrix->data, matrix->size);
	}

	// print_to_console(result->lin.data, result->lin.size);
	// printf("FINISHED PRINTING OUT THE RESULT\n");

	// // Assume 2D
	// size_t square_mat_dim_length = ceil(pow(L3, 1/(double)2));

	// size_t block_dimensions[2] = 	{square_mat_dim_length, square_mat_dim_length};
	// size_t tensor_dimensions[2] = 	{square_mat_dim_length, square_mat_dim_length};
	// size_t l_dim_length = square_mat_dim_length;

	// // Get one tensor object for the whole scenario
	// struct tensor_storage *tensor = gen_block_tensor_clean(2, tensor_dimensions, block_dimensions);
	// struct lin_storage *matrix = gen_vector(square_mat_dim_length * l_dim_length);

	// // To get the result right, 9D is just a little smaller than 10D, we must allocate a lot for the result
	// // But as this is TMM case, we can simply allocate the same for the result as for the tensor?
	// // Result is just another tensor, we do not care (...?)
	// struct tensor_storage  *result = gen_block_tensor_clean(2, tensor_dimensions, block_dimensions);

	// Move through different block sizes, n sizes, dims, modes
	// At each step we have to adjust the tensor (anyway!)

	typedef void (*TMM)();
	TMM model_algorithm = tmm_looped_mkl;

	const int tmm_algorithms_count = 1;
	TVM tmm_algorithms[1] = {
		tmm_looped_mkl
	};

	// parameters' loops ordered according to their dependency
	for (size_t dim=(size_t) dim_min; dim<=(size_t) dim_max; ++dim) {

		printf("dim=%zu:\n", dim);

		// Pick the block size (!) first
		size_t block_array[3];
		if (!TEST_ENV) {
			if (dim == 2) {
				block_array[0] =    44;
				block_array[1] =   124;
				block_array[2] =   572;//1276;
			} else if (dim == 3) {
				block_array[0] =   12;
				block_array[1] =   24;
				block_array[2] =   68;//116;
			} else if (dim == 4) {
				block_array[0] =   6; 
				block_array[1] =  10;
				block_array[2] =  34;//24;
			} else if (dim == 5) {
				block_array[0] =  4;
				block_array[1] =  6;
				block_array[2] = 16;//12;
			} else if (dim == 6) {
				block_array[0] =   3;
				block_array[1] =   4;
				block_array[2] =   10;//8;
			} else if (dim == 7) {
				block_array[0] =   2;
				block_array[1] =   4;
				block_array[2] =   7;//6;
			} else if (dim == 8) {
				block_array[0] =   2;
				block_array[1] =   3;
				block_array[2]  =  5;
			} else if (dim == 9) {
				block_array[0] =   2;
				block_array[1] =   3;
				block_array[2]  =  4;
			} else if (dim == 10) {
				block_array[0] =  2;
				block_array[1]  = 4;
				block_array[2]  = 0; // will be ignored(!)
			}
		} else {
			block_array[0] = floor(pow(L1, 1/(double)dim));
			block_array[1] = 0;
			block_array[2] = 0;
		}

		// Loop over possible block_sizes 
		for (size_t block_id=0; block_id<1; block_id++) {
			size_t block_dim_length = block_array[block_id];
			if (block_dim_length == 0) {
				break;
			}

			size_t block_layout[dim];
			size_t tensor_layout[dim];
			// Set the dimensionality of tensor / result 
			result->dim = dim;
			tensor->dim = dim;

			// Get the max size per dimension(!) given this dim
			size_t tensor_dim_length = floor(pow(TEST_SIZE, 1/(double)dim));
			// Check that this is valid(!)
			assert(pow(tensor_dim_length, dim) <= (double) max_length);

			// Get a random distribution of dimension sizes (with a threshold)
			// randomize_array_int(tensor_layout, dim, tensor_dim_length/8);
			reset_array_sizet(tensor_layout, dim, 6);
			// tensor_layout[0] = 6;

			printf("        n = ");
			print_to_console_sizet(tensor_layout, dim);
			// Update layout, layout_perm and size
			free(tensor->layout);
			free(tensor->layout_perm);
			set_tensor_layout(tensor, tensor_layout);

			// Get block sizes which actually "perfectly" divide our sizes...
			// for (int mode=0; mode<=dim-1; ++mode) {
			// 	int divisor = 2;
			// 	for (; divisor<=tensor->layout[mode]; ++divisor) {
			// 		if (tensor->layout[mode] % divisor == 0) {
			// 			break;
			// 		}
			// 	}
			// 	if (tensor->layout[mode] == 1) {
			// 		block_layout[mode] = 1;
			// 	} else {
			// 		block_layout[mode] = divisor;
			// 	}
			// }
			// randomize_array_int_from_array(block_layout, dim, tensor_layout);
			reset_array_sizet(block_layout, dim, 3);
			// block_layout[0] = 3;

			printf("            block_n = ");
			print_to_console_sizet(block_layout, dim);
			// Update block_layout + result block _ layout reset back to normal(!!)
			free(tensor->block_layout);
			tensor->block_layout = copy_array_int(block_layout, tensor->dim);
			free(result->block_layout);
			result->block_layout = copy_array_int(block_layout, tensor->dim);

			// MAYBE AT THSI POINT THE PROBLEM IS THE BLOCK LAYOUT?????

			// Create storage for testing (validating)
			struct tensor_storage *blocked_tensor = get_block_tensor(tensor, 0, 0);
			struct tensor_storage *morton_blocked_tensor = get_block_tensor(tensor, 0, 1);
			// struct tensor_storage *unblocked_result = get_block_tensor(model_result, 0, 0);

			// Move through all possible modes
			// Q: Do we need to update vector length?
			for (size_t mode=0; mode<=dim-1; ++mode) {
				printf("\nmode %zu\n", mode);

				for (size_t l_dim_length=l_min; l_dim_length<=l_max; l_dim_length++) {
					// if (l_dim_length != tensor->layout[mode]) {
					// 	printf("	l_dim (does not match n_k; break) %zu\n", l_dim_length);
					// 	continue;
					// } else {
					// 	printf("	l_dim %zu\n", l_dim_length);
					// }
					size_t l_block_length;
					// Find optimal size of block for l_dim
					for (l_block_length=(l_dim_length)/2; l_block_length>0; l_block_length++) {
						if (l_dim_length % l_block_length == 0) {
							break;
						}
					}
					if (l_block_length == 1) {
						l_block_length = l_dim_length;
					}
					printf("\nL = %zu\n", l_dim_length);
					printf("=========================== Optimal block chosen for this l_dim_length is %zu\n", l_block_length);

					secondary_matrix->lin.size = tensor->layout[mode] * l_dim_length;
					printf("We set secondary_matrix size to be %zu (%zu times %zu)\n", secondary_matrix->lin.size, tensor_layout[mode], l_dim_length);
					// This laoyut information is not used *currently

					if (mode != dim-1) {
						secondary_matrix->layout[0] = l_dim_length;
						secondary_matrix->layout[1] = tensor->layout[mode];
						secondary_matrix->block_layout[0] = l_dim_length;
						secondary_matrix->block_layout[1] = tensor->block_layout[mode];
						secondary_matrix->layout_perm[0] = 0;
						secondary_matrix->layout_perm[1] = 1;
					} else {
						// Alternatively we could do Trans computation (with the first matrix!); Probably worth benchmarking both options
						secondary_matrix->layout[0] = tensor->layout[mode];
						secondary_matrix->layout[1] = l_dim_length;
						secondary_matrix->block_layout[0] = tensor->block_layout[mode];
						secondary_matrix->block_layout[1] = l_dim_length;
						secondary_matrix->layout_perm[0] = 0;
						secondary_matrix->layout_perm[1] = 1;
					}

					// Update the mode dimension in the result
					free(result->layout);
					// Update layout, layout_perm and size (MANUALLY!)
					// Ignore the layout_perm
					result->layout = copy_array_int_replace_mode(tensor->layout, result->dim, mode, l_dim_length);
					result->lin.size = get_size(result->layout, result->dim);

					printf("Result_layout (size=%zu) = ", result->lin.size);
					print_to_console_sizet(result->layout, dim);

					printf("\nThis is tmm looped\n(Original) Vector (%zu):\n", secondary_matrix->lin.size);
					if (TEST_ENV == 1) print_to_console(secondary_matrix->lin.data, secondary_matrix->lin.size);
					reset_array(result->lin.data, result->lin.size, 0); 
					tmm_looped_mkl(tensor, &secondary_matrix->lin, &result->lin, mode, l_dim_length, l_block_length);
					printf("Result of tmm_looped_mkl (size %zu):\n", result->lin.size);
					if (TEST_ENV == 1) print_to_console(result->lin.data, result->lin.size);
					// Let's assume this si ground truth -> perform this just once and dont use result later on
					qsort(result->lin.data, result->lin.size, sizeof(DTYPE), compare);

					// for blocked variant, we use two new objects
					matrix->size = secondary_matrix->lin.size;
					unblocked_result->size = result->lin.size;

					block_array_int(matrix, secondary_matrix, 0, 0);
					printf("\nThis is tmm blocked\n(Blocked) Vector (%zu):\n", matrix->size);
					if (TEST_ENV == 1) print_to_console(matrix->data, matrix->size);
					reset_array(unblocked_result->data, unblocked_result->size, 0); 
					tmm_blocked_mkl(blocked_tensor, matrix, unblocked_result, mode, l_dim_length, l_block_length);
					printf("Result of tmm_blocked_mkl (size %zu):\n", unblocked_result->size);
					if (TEST_ENV == 1) print_to_console(unblocked_result->data, unblocked_result->size);

					// compare that the results are identical
					// unblocked should get the right size(!)
					// UPDATE BLOCKING INFORMATION(!!!) destructive change, but this is updated when we change mode
					// result->block_layout[mode] = 1;
					// unfortunately this function is broken when block size equals layout size (methinks)! - block_array_int(unblocked_result, result, 1);
					// printf("Result of tmm_blocked_mkl (size %zu):\n", result->lin.size);
					// if (TEST_ENV == 1) print_to_console(result->lin.data, result->lin.size);

					// sort both results to see if they are equal...
					
					qsort(unblocked_result->data, unblocked_result->size, sizeof(DTYPE), compare);
					size_t pos = 0;
					while ( fequal(unblocked_result->data[pos], result->lin.data[pos]) ) {
						if (++pos == result->lin.size) {
							break;
						}
					}
					if (pos != unblocked_result->size) {
						printf("Algorithm incorrect.\n");
						exit(-1);
					} else {
						printf("Algorithm correct.\n");
					}

					// Finally, morton algorithm
					// Careful: matrix actually contains the blocked one
					printf("\nThis is morton blocked\n(Original) Vector (%zu):\n", matrix->size);
					if (TEST_ENV == 1) print_to_console(matrix->data, matrix->size);
					reset_array(unblocked_result->data, unblocked_result->size, 0); 
					tmm_mortonblocked_mkl(morton_blocked_tensor, matrix, unblocked_result, mode, l_dim_length, l_block_length);
					printf("Result of morton blocked (size %zu):\n", unblocked_result->size);
					if (TEST_ENV == 1) print_to_console(unblocked_result->data, unblocked_result->size);

					qsort(unblocked_result->data, unblocked_result->size, sizeof(DTYPE), compare);
					pos = 0;
					while ( fequal(unblocked_result->data[pos], result->lin.data[pos]) ) {
						if (++pos == result->lin.size) {
							break;
						}
					}
					if (pos != unblocked_result->size) {
						printf("Algorithm incorrect.\n");
						// exit(-1);
					} else {
						printf("Algorithm correct.\n");
					}
					
					printf("\nThis is tmm blocked (which blocks on L as well)\n");
					// Create a blocked "matrix"
					if (mode != dim-1) {
						secondary_matrix->layout[0] = l_dim_length;
						secondary_matrix->layout[1] = tensor->layout[mode];
						secondary_matrix->block_layout[0] = l_block_length;
						secondary_matrix->block_layout[1] = tensor->block_layout[mode];
						secondary_matrix->layout_perm[0] = 0;
						secondary_matrix->layout_perm[1] = 1;
					} else {
						// Alternatively we could do Trans computation (with the first matrix!); Probably worth benchmarking both options
						secondary_matrix->layout[0] = tensor->layout[mode];
						secondary_matrix->layout[1] = l_dim_length;
						secondary_matrix->block_layout[0] = tensor->block_layout[mode];
						secondary_matrix->block_layout[1] = l_block_length;
						secondary_matrix->layout_perm[0] = 0;
						secondary_matrix->layout_perm[1] = 1;
					}
					block_array_int(matrix, secondary_matrix, 0, 0);
					printf("(DoubleBlocked) Vector -- broken for dim-1 but still -- (%zu):\n", matrix->size);
					if (TEST_ENV == 1) print_to_console(matrix->data, matrix->size);
					// // Seriously just transpose...
					// temp_matrix->size = matrix->size;
					// size_t next = 0;
					// for (size_t rows=0; rows<l_dim_length; ++rows) {
					// 	for (size_t cols=0; cols<tensor->layout[mode]; ++cols) {
					// 		temp_matrix->data[next++] = secondary_matrix->lin.data[cols*l_dim_length + rows];
					// 	}
					// }
					// out_in_array_int(matrix, secondary_matrix, 0);
					// printf("(Blocked) Vector (%zu):\n", temp_matrix->size);
					// if (TEST_ENV == 1) print_to_console(temp_matrix->data, temp_matrix->size);
					reset_array(unblocked_result->data, unblocked_result->size, 0); 
					tmm_blocked_libx(blocked_tensor, matrix, unblocked_result, mode, l_dim_length, l_block_length);
					printf("Result of tmm_blocked_mkl (size %zu):\n", unblocked_result->size);
					if (TEST_ENV == 1) print_to_console(unblocked_result->data, unblocked_result->size);

					qsort(unblocked_result->data, unblocked_result->size, sizeof(DTYPE), compare);
					pos = 0;
					while ( fequal(unblocked_result->data[pos], result->lin.data[pos]) ) {
						if (++pos == result->lin.size) {
							break;
						}
					}
					if (pos != unblocked_result->size) {
						printf("Algorithm incorrect.\n");
						exit(-1);
					} else {
						printf("Algorithm correct.\n");
					}

					// For now, just use the same block_layout of the result

					// struct tensor_storage *blocked_tensor = get_block_tensor(tensor, 0, 0);
					// struct tensor_storage *unblocked_result = get_block_tensor(model_result, 0, 0);

					// printf("First block algorithms (%d)\n", count[0]);
					// out_algo = test_algorithms(unfold_unfold_algorithms, algo_counter, count[0], &result->lin, unblocked_result->lin.data, blocked_tensor, matrix, mode,
					// 	filename, filename2, dim, n, block_n, out_algo, unfold, NULL, NULL);
					// algo_counter += count[0];

					// // run all algorithms in a loop
					// for (int algo=0; algo<algos_block_unfold; ++algo) {
					// 	// if(mode==0 || mode==dim-1) {
					// 	// 	continue;
					// 	// }
					// 	reset_array(result->lin.data, result->lin.size, 0);
					// 	measure_unfold(
					// 		algorithms_block_unfold[algo], tensor, matrix, &result->lin, mode,
					// 		file, tensor_layout[0], block_n, unfold, block_size);
					// 	fflush(file);
					// }

				}
			}

			free_tensor_storage(blocked_tensor);
			free_tensor_storage(morton_blocked_tensor);

		}
	}

	free_tensor_storage(tensor);
	free_tensor_storage(result);
	free_tensor_storage(secondary_matrix);
	free_lin_storage(matrix);
	free_lin_storage(temp_matrix);


  	const int maxDim = 3;
	int coord[maxDim], coord1[maxDim], nDims, nBits, nBytes, i;
	bitmask_t r, r1;

  for (;;)
    {
      printf( "Enter nDims, nBits: " );
      scanf( "%d", &nDims );
      if ( nDims == 0 )
	break;
      scanf( "%d", &nBits );
      while ( (i = getchar()) != '\n' && i != EOF )
	;
      if ( i == EOF )
	break;
      nBytes = (nBits+31)/32*4;
      
      for ( r = 0; r < 1 << (nDims*nBits); r++ )
        {
	  hilbert_i2c( nDims, nBits, r, coord );
	  printf("%d: ", (unsigned)r);
	  for ( i = 0; i < nDims; i++ )
	    printf(" %d", coord[i]);
	  printf("\n");
	  r1 = hilbert_c2i( nDims, nBits, coord );
	  if ( r != r1 )
	    printf( "r = 0x%x; r1 = 0x%x\n", (unsigned)r, (unsigned)r1);

	  for (r1 = 0; r1 < r; ++r1 )
	    {
	      int j;
	      int inf_dist = 0;
	      int ans;
	      hilbert_i2c( nDims, nBits, r1, coord1 );
	      ans = hilbert_cmp( nDims, nBytes, coord, coord1);
	      if (ans != 1)
		printf( "cmp r = 0x%0*x; r1 = 0x%0*x, ans = %2d\n", (nDims*nBits+3)/4, (unsigned)r,
                        (nDims*nBits+3)/4, (unsigned)r1, ans );
	    }
	  hilbert_i2c( nDims, nBits, r1, coord1 );
	  if (hilbert_cmp( nDims, nBytes, coord, coord1) != 0)
	    printf( "cmp r = 0x%0*x; r1 = 0x%0*x\n", (nDims*nBits+3)/4, (unsigned)r,
		    (nDims*nBits+3)/4, (unsigned)r1 );

        }
    }


    
	// if (file != NULL) {
	// 	fclose(file);
	// }

	return 0;
}
