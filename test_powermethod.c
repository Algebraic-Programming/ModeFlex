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
#include <bench_utils.h>
#include <time_meas.h>

// test case: add some code which appears only in tests
// test case: make numbers round (or integers!)
// These are all numbers of doubles
#define L1 4096.0 // 32 KB
#define L1_measured 2048.0
#define L3 3276800.0 
#define L3_measured 2097152.0 // 16 MB of L3 taken for the measurement (not half...)

#define RAM_new 5368709120.0

#define RAM 4294967296.0 // not the actual max of RAM, just 32 GB (double RAM_measured)
#define RAM_measured 2684354560.0 // 20 GB
		             
#define BENCH_SIZE_RESULT 		9 // size_10d = 9 is 3GB
#define MINI_BENCH_SIZE 		10077696.0 // RAM_measured/4
#define MINI_BENCH_SIZE_RESULT 	5 // size_10d = 8 is 1GB

#define TENGIGABYTES 1342177280
#define TENGIGABYTES_RESULT 8
#define FIXFORSEVEND 1801088541

// #define SINGLEBLOCK
#define ONEBLOCKSIZE
#define BLOCK_SIZES 3

// For now, normalize input tensor only for the test case (stochastic tensor)
#if (TEST_ENV == 1) || defined(SINGLEBLOCK)
	#define SIZE 10077696.0
	// #define NORMALIZE
#else 
	#define SIZE TENGIGABYTES // MINI_BENCH_SIZE
#endif

#define FILENAME(x); snprintf(filename, BUFSIZE, "%s/%s_%.0f_dimmin_%d_dimmax_%d_nmin_%d_nmax_%d_modemin_%d_modemax_%d_blockn_%d_%s.csv", FOLDER, hostname, timespec_to_microseconds(time), dim_min, dim_max, 0, 0, 0, 0, 0, x)

#define epsilon 1e-3 // adjusted from 1e-6 to 1e-5
inline int fequal(double a, double b) {
	return fabs(a-b) < epsilon;
}
extern int compare( const void* _a, const void* _b);
#ifndef bitmask_t_
#define bitmask_t_
#define bitmask_t 	unsigned long long
#endif

int
lin_compare(const struct lin_storage * const model, const struct lin_storage * const trial) {
	size_t pos = 0;
	while ( fequal(model->data[pos], trial->data[pos]) ) {
		if (++pos == model->size) {
			break;
		}
	}
	if (pos != model->size) {
		printf("error: compare %lf vs %lf (eror is %lf)\n", model->data[pos-1], trial->data[pos-1], fabs(model->data[pos-1]-trial->data[pos-1]));
		printf("from where its broken till end:\n");
		size_t print_size = model->size - pos;
		if (print_size > 20) {
			print_size = 20;
		}
		print_to_console(model->data + pos, print_size);
		print_to_console(trial->data + pos, print_size);
		exit(-1);
		// printf("this particular vector is not equal\n");
		return 0;
	} else {
		// printf("this particular vector is equal\n");
		return 1;
	}
}

int
lin_compare_series(struct lin_storage ** const model_array, struct lin_storage ** const compare_array, int array_size) {
	int equal = 1;
	size_t edge_size = compare_array[0]->size;

	for (int i=0; i<array_size; ++i) {
		// Compare each lin_storage in the array
		int subequal = lin_compare(model_array[i], compare_array[i]);
		// printf("comparing: with:\n");
		// print_to_console(model_array[i]->data, array_size);
		// print_to_console(compare_array[i]->data, array_size);
		equal &= subequal;
		if (equal != 1) {
			printf("error found in vector %d\n", i);
		}
	}
	if (equal == 1) {
		printf("lin_compare_series: arrays are equal\n");
	} else {
		size_t print_size = edge_size;
		if (print_size > 10) {
			print_size = 10;
		}
		printf("lin_compare_series: error found in one of the vectors\n");
		for (int d=0; d<array_size; ++d) {
			printf("%d: ", d);
			print_to_console(compare_array[d]->data, print_size);
		}
		exit(-1);
	}
	return equal;
}

int test_powermethod(int argc, char ** argv) {

	int algo_to_test;
	if (TEST_ENV == 1) {
		algo_to_test = 4; // count pmModel in
	} else {
		algo_to_test = 4; // count pmModel in
	}
	
	TVM algorithms_powermethod[] = {

		pmModel,
		pmMortonLibx,
		pmBlockLibx,
		pmLooped,

		pmMorton,


		pmBlock,



		// pmTaco10, // works only for 10D(!)
		// pmTaco5,

		pmMortonLibxVms,
		pmMortonLibxSingle,
		pmMortonLibxSingleMvs,

		pmMortonSingleMvs, 
		pmLoopedSingle,
		pmBlockSingleMvs,
		pmLoopedSingleMvs,
		pmMortonSingle,

		pmMortonMyself,
		
	};
	
	int dim_min = 2;
	int dim_max = 2;
	switch (argc) {
		case 2:
			sscanf (*(argv+argc--), "%d", &dim_max);
			sscanf (*(argv+argc--), "%d", &dim_min);
	}

	char hostname[1024];
	gethostname(hostname, 1024);
	struct timespec time;
	clock_gettime(CLOCK_MONOTONIC, &time);

	char filename[BUFSIZE];
	if (TEST_ENV == 1) {
		printf("test_powermethod.c: TEST ENV\n");
		FILENAME("test");
	} else {
		printf("test_powermethod.c: REAL ENV\n");
		FILENAME("bench");
	}
	printf("filename=%s\n", filename);
	// printf("BUFSIZE is %d, FOLDER is %s, TIME is %.0f\n", BUFSIZE, FOLDER, timespec_to_microseconds(time));

	FILE * file;
	if (TEST_ENV == 0) {
		file = fopen(filename, "w"); // r+ if you want to update without deleteing it
		if (file == NULL) {
			perror("Error opening file.\n");
		}
		write_powermethod_header(file);
	}
	
	size_t size_1d, size_10d;
	size_1d = SIZE;
	if (TEST_ENV == 1) {
		printf("We are in test environment.");
		size_10d = ceil(pow(size_1d, 1/(double)10));
	} else {
		size_10d = TENGIGABYTES_RESULT; // BENCH_SIZE_RESULT;
	}

	#ifdef SINGLEBLOCK
		size_10d = ceil(pow(size_1d, 1/(double)10));
	#endif
		
	printf("size_1d = %zu, size_10d = %zu\n", size_1d, size_10d);

	struct tensor_storage * const tensor = gen_block_tensor_clean(1, &size_1d, &size_1d);
	printf("10 el of tensor memory: ");
	print_to_console(tensor->lin.data, 10);
	// block_layout (init stage, same as layout)
	free(tensor->block_layout);
	tensor->block_layout = copy_array_int(tensor->layout, tensor->dim);
	print_to_console_sizet(tensor->block_layout, tensor->dim);

	// Vector size is at most half of size_1d
	size_t size_max_vector = ceil(pow(size_1d, 1.0/(double)2));
	printf("Size of each vector is half of the total linear size (worst case): %zu\n", size_max_vector);

	struct lin_storage * vector_array[dim_max];
	struct lin_storage * number_vectors[dim_max];
	struct lin_storage * model_vector[dim_max];
	for (int i=0; i<dim_max; ++i) {
		vector_array[i] = gen_vector_seeded(size_max_vector, i);
		number_vectors[i] = gen_vector_seeded(size_max_vector, i);
	}
	if (TEST_ENV == 1) {
		for (int i=0; i<dim_max; ++i) {
			model_vector[i] = gen_vector_seeded(size_max_vector, i);
		}
	}

	// Precalculate how much memory is needed for the result -> for now assume the worst case, which is 10D 
	// size per dimension for dim_max case
	size_t size_maxd_result = ceil(pow(size_1d, 1/(double)dim_max));
	size_t size_max_result = pow(size_maxd_result, dim_max-1);
	if (size_max_result > size_1d) {
		printf("Should never happen?, size_max_result points to %zu", size_max_result);
		size_max_result = size_1d;
	}
	printf("Size of temp result is in total (worst case): %zu\n", size_max_result);

	printf("Allocating result tensors:\n");
	struct tensor_storage * const result_1 = gen_block_tensor_clean(1, &size_max_result, &size_max_result);
	struct tensor_storage * const result_2 = gen_block_tensor_clean(1, &size_max_result, &size_max_result);

	for (size_t dim=(size_t) dim_min; dim<=(size_t) dim_max; ++dim) {

		printf("\ndim = %zu ==================================================================\n", dim);
		size_t block_array[BLOCK_SIZES];
		size_t size_1d_modified = floor(pow(size_1d, 1/(double)dim));
		printf("(dim %zu) n (calculated): %zu\n", dim, size_1d_modified);

		// if (TEST_ENV == 1) {

		// 	assert(pow(size_1d_modified, dim) <= (double) size_1d);
		// 	size_t block_dim_length = floor(pow(size_1d, 1/(double)dim));
		// 	if (block_dim_length > size_1d_modified) {
		// 		printf("reduce block size; ");
		// 		block_dim_length = size_1d_modified;
		// 	}
		// 	if (size_1d_modified % block_dim_length != 0) {
		// 		printf("make it divisible; ");
		// 		block_dim_length = size_1d_modified;
		// 	}
		// 	printf("(dim %zu) block_n (calculated): %zu\n", dim, block_dim_length);

		// 	// Currently in test case we assume block size (2)
		// 	// Let's change it to have more than 1 block size ?
		// 	block_array[0] = 2;
		// 	block_array[1] = 3;
		// 	block_array[2] = 0;

		// 	block_array[0] = 2;
		// 	block_array[1] = 2;

		if (dim == 2) {
			block_array[0] = 44;
			block_array[1] = 124;
			block_array[2] = 1276; // 10% (2.6MB): 572; 13MB: 1276;
		} else if (dim == 3) {
			block_array[0] = 12;
			block_array[1] = 24;
			block_array[2] = 116; // 10% (2.5MB): 68; 12.5MB: 116;
		} else if (dim == 4) {
			block_array[0] = 6; 
			block_array[1] = 10;
			block_array[2] = 34; // 10% (2.7MB): 24; 10MB: 34;
		} else if (dim == 5) {
			block_array[0] = 4;
			block_array[1] = 6;
			block_array[2] = 16; // 10% (2MB): 12; 8.5MB: 16;
		} else if (dim == 6) {
			block_array[0] = 3;
			block_array[1] = 4;
			block_array[2] = 10; // 10% (2MB): 8; 8MB: 10;
		} else if (dim == 7) {
			block_array[0] = 2;
			block_array[1] = 4;
			block_array[2] = 9;//8;//7; // 10% (2.2MB): 6; 6.6MB: 7;
		} else if (dim == 8) {
			block_array[0] = 2;
			block_array[1] = 3;
			block_array[2] = 6;//5; // 10% (3MB): 5; 
		} else if (dim == 9) {
			block_array[0] = 2;
			block_array[1] = 3;
			block_array[2] = 5;//4; // 10% (2MB): 4; 15MB: 5;
		} else if (dim == 10) {
			block_array[0] = 2;
			block_array[1] = 4; // ?% (8MB): 4;
			block_array[2] = 0;
		}

		printf("(dim %zu) block_n (final): ", dim);
		print_to_console_sizet(block_array, BLOCK_SIZES);

		// CRUCIAL (!!!) OTHERWISE BUGS
		tensor->dim = dim;
		result_1->dim = dim - 1;
		result_2->dim = dim - 1;

		// reset_array_sizet(tensor_layout, dim, size_1d_modified);
		// printf("       n = ");
		// print_to_console_sizet(tensor_layout, dim);
		// free(tensor->layout);
		// free(tensor->layout_perm);
		// set_tensor_layout(tensor, tensor_layout);

		// // technically it's enoguh to just do this
		// free(result_1->layout);
		// free(result_1->layout_perm);
		// set_tensor_layout(result_1, tensor_layout);
		// free(result_2->layout);
		// free(result_2->layout_perm);
		// set_tensor_layout(result_2, tensor_layout);

		// for (int i=0; i<dim; ++i) {
		// 	vector_array[i]->size = size_1d_modified;
		// 	model_vector[i]->size = size_1d_modified;
		// }
		// // if (TEST_ENV == 1) {
		// 	// round_numbers(tensor->lin.data, tensor->lin.size);
		// for (int i=0; i<dim; ++i) {
		// 	double size1 = normalize(vector_array[i], size_1d_modified);
		// 	double size2 = normalize(model_vector[i], size_1d_modified);
		// }
		// // }
		// lin_compare_series(model_vector, vector_array, dim);

		// Here we already know the dimensionality, etc
		// Normalize per row

		#ifdef NORMALIZE
			normalize_rows(tensor, 0);
		#endif

		// printf("Initial vectors (normalized as vectors of %zu elements i.e. mode size):\n", size_1d_modified);
		// for (int d=0; d<dim; ++d) {
		// 	printf("%d: ", d);
		// 	print_to_console(vector_array[d]->data, 4);
		// }
		
		/////////////////////////////////////////////////////// SETUP
		// ALl those algorithms rely on the following:
		// Size of: tensor, vector, result
		// Additionally: tensor layout and dim

		///////////////////////////////////////////////////////

		#define ITERATIONS 1

		///////////////////////////////////////////////////////

		// printf("\nMODEL ALGORITHM\n");
		// struct tensor_storage * input;
		// struct lin_storage * output;
		// for (int iter=0; iter<ITERATIONS; ++iter) {
		// 	for (int vector_up=0; vector_up<dim; ++vector_up) {
		// 	#ifdef DEBUG_ENV
		// 		printf("vector_up=%d, \n", vector_up);
		// 	#endif
		// 		// in/out set for iterations == 0(!)
		// 		int iterations = 0;
		// 		input = tensor;
		// 		output = &result_1->lin;
		// 		for (int mode=dim-1; mode>=0; --mode) {
		// 			if (mode == vector_up) continue;
		// 			#ifdef DEBUG_ENV
		// 			printf("compute with mode %d\n", mode);
		// 			#endif
		// 			if (iterations == dim-2) {
		// 				// printf("Output is a vector!\n");
		// 				output = model_vector[vector_up];
		// 			}
		// 			input->dim = dim-iterations;
		// 			output->size = input->lin.size / model_vector[0]->size;
		// 			// printf("input: %zu (dim %zu), output: %zu\n",
		// 				// input->lin.size, input->dim,
		// 				// output->size);
		// 			tvm_vector_major_BLAS_col_mode(input, model_vector[mode], output, mode);
		// 			// printf("Current state\n");
		// 			// print_to_console(output->data, output->size);
		// 			++iterations;
		// 			if (iterations % 2 == 0) {
		// 				output = &result_1->lin;
		// 				reset_array(result_1->lin.data, result_1->lin.size, 0);
		// 				input = result_2;
		// 			} else {
		// 				output = &result_2->lin;
		// 				reset_array(result_2->lin.data, result_2->lin.size, 0);
		// 				input = result_1;
		// 			}
		// 		}

		// 	#ifdef NORMALIZE
		// 	// printf("norm, ");
		// 	(void) normalize(model_vector[vector_up], size_1d_modified);
		// 	#endif 

		// 	}
		// 	// printf("\n");
		// }
		// result_1->dim = dim;
		// result_2->dim = dim;

		// if (TEST_ENV == 1) {
		// 	// Printing one of the model vectors
		// 	printf("Model vectors (normalized):\n");
		// 	for (int d=0; d<dim; ++d) {
		// 		printf("%d: ", d);
		// 		print_to_console(model_vector[d]->data, size_1d_modified);
		// 	}
		// 	printf("\n");
		// }

		/////////////////////////////////////////////////////// 
		/////////////////////////////////////////////////////// 

		for (size_t block_id=0; block_id<BLOCK_SIZES; block_id++) {

			size_t block_dim_length = block_array[2];
			if (block_dim_length == 0) {
				block_dim_length = block_array[1];
				// break;
			}
			printf("\nblock_n = %zu ======================================\n", block_dim_length);

			// Technically this code for the benchmarking case can be simplified (!) we do not need an actual blocked tensor...
			size_t block_layout[dim];
			reset_array_sizet(block_layout, dim, block_dim_length);
			printf("block_n = ");
			print_to_console_sizet(block_layout, dim);
			free(tensor->block_layout);
			tensor->block_layout = copy_array_int(block_layout, dim);

			size_t result_block_layout[dim];
			reset_array_sizet(result_block_layout, dim-1, block_dim_length);
			free(result_1->block_layout);
			free(result_2->block_layout);
			result_1->block_layout = copy_array_int(result_block_layout, dim-1);
			result_2->block_layout = copy_array_int(result_block_layout, dim-1);

			// Make the tensor divisible
			#ifdef SINGLEBLOCK
				size_t temp = 1;
			#else 
				size_t temp = size_1d_modified/block_dim_length;
				if (temp == 0) {
					temp = 1;
				}
				while (temp>1 && pow(temp*block_dim_length,dim)>size_1d) {
					temp--;
					printf("decreasing the temp size to match available tensor memory; ");
				}
			#endif

			printf("Number of block in each dimension is %zu. Taking in total %zu elements in each dimension. That raised to the dimensionality of the tensor is pow(%zu, %zu) = %d. Total available size is %zu.\n",
				temp,
				temp*block_dim_length,
				temp*block_dim_length,
				dim,
				(int) pow(temp*block_dim_length,dim),
				size_1d);

			assert(pow(temp*block_dim_length,dim)<=size_1d);
			size_t size_1d_local = temp*block_dim_length;
			printf("n: %zu (ensures divisibility)\n", size_1d_local);

			size_t tensor_layout[dim];
			reset_array_sizet(tensor_layout, dim, size_1d_local);
			free(tensor->layout);
			free(tensor->layout_perm);
			set_tensor_layout(tensor, tensor_layout);

			size_t result_tensor_layout[dim];
			reset_array_sizet(result_tensor_layout, dim-1, size_1d_local);
			free(result_1->layout);
			free(result_1->layout_perm);
			free(result_2->layout);
			free(result_2->layout_perm);
			set_tensor_layout(result_1, result_tensor_layout);
			set_tensor_layout(result_2, result_tensor_layout);

			// CAREFUL (!!!!) MODIFY THIS
			struct tensor_storage *blocked_tensor;
			struct tensor_storage *morton_blocked_tensor;
			if (TEST_ENV == 1) {
				printf("Allocating blocked (and morton blocked) tensors for correctness (REMOVE FROM BENCHMARKING)\n");
				blocked_tensor = get_block_tensor(tensor, 0, 0);
				morton_blocked_tensor = get_block_tensor(tensor, 0, 1);	
			}

			for (int algo_id=0; algo_id<algo_to_test; ++algo_id) {
				printf("\n");

				// Skip the model algorithm if we are not testing
				if (TEST_ENV == 0 && algo_id == 0) {
					printf("Do not benchmark the model algorithm\n");
					continue;
				}

				// Execute pmLooped only once
				// if (block_id !=0 && (
				// 	algorithms_powermethod[algo_id] == pmLooped ||
				// 	algorithms_powermethod[algo_id] == pmLoopedSingle ||
				// 	algorithms_powermethod[algo_id] == pmLoopedSingleMvs)) {
				// 	continue; 
				// }

				// If benchmarking, do not need to reset the vectors? -> I think we do(?)
				// double check if this is not done in bench utils
				// If testing, reset like the following but if benchmarking we never reset (or need) the model vectors
				if (TEST_ENV == 1) {
					for (size_t j=0; j<dim; ++j) {
						vector_array[j]->size = size_1d_local;
						model_vector[j]->size = size_1d_local;
						if (algo_id == 0) {
							for (size_t el=0; el<size_1d_local; ++el) {
								model_vector[j]->data[el] = number_vectors[j]->data[el];
							}			
						} else {
							for (size_t el=0; el<size_1d_local; ++el) {
								vector_array[j]->data[el] = number_vectors[j]->data[el];
							}	
						}
					}
					size_t print_size = size_1d_local;
					if (print_size > 10) {
						print_size = 10;
					}
					// printf("Vectors (at most 10 el):\n");
					// for (size_t i=0; i<dim; ++i) {
					// 	printf("%zu: ", i);
					// 	print_to_console(vector_array[i]->data, print_size);
					// }
				}

				// Set which tensor will be used as input
				struct tensor_storage * input_tensor = tensor;
				if (TEST_ENV == 1) {
					if (algorithms_powermethod[algo_id] == pmMorton || 
						algorithms_powermethod[algo_id] == pmMortonMyself ||
						algorithms_powermethod[algo_id] == pmMortonLibx ||
						algorithms_powermethod[algo_id] == pmMortonLibxVms ||
						algorithms_powermethod[algo_id] == pmMortonSingle ||
						algorithms_powermethod[algo_id] == pmMortonLibxSingle ||
						algorithms_powermethod[algo_id] == pmMortonSingleMvs || 
						algorithms_powermethod[algo_id] == pmMortonLibxSingleMvs) {
						printf("Selecting morton blocked tensor as input.\n");
						input_tensor = morton_blocked_tensor;
					} else if (algorithms_powermethod[algo_id] == pmLooped ||
						algorithms_powermethod[algo_id] == pmTaco10 ||
						algorithms_powermethod[algo_id] == pmTaco5 ||
						algorithms_powermethod[algo_id] == pmLoopedSingle ||
						algorithms_powermethod[algo_id] == pmLoopedSingleMvs ||
						algorithms_powermethod[algo_id] == pmModel) {
						printf("Selecting regular tensor as input.\n");
						input_tensor = tensor;
					} else {
						printf("Selecting blocked tensor as input.\n");
						input_tensor = blocked_tensor;
					}
				}
				
				if (TEST_ENV == 1) {
					printf("Algorithm %d; Resetting the result_1 and result_2\n", algo_id);
					reset_array(result_1->lin.data, result_1->lin.size, 0.0);
					reset_array(result_2->lin.data, result_2->lin.size, 0.0);
				} else {
					printf("Algorithm %d; Benchmark will reset the results\n", algo_id);
				}

				size_t print_size = size_1d_local;
				if (print_size > 10) {
					print_size = 10;
				}

				if (TEST_ENV == 1) {
					if (algo_id == 0) {
						algorithms_powermethod[algo_id](input_tensor, model_vector, result_1, result_2, ITERATIONS);
						// Reset back those (avoids side effects)
						tensor->dim = dim;
						result_1->dim = dim - 1;
						result_2->dim = dim - 1;
					} else {
						algorithms_powermethod[algo_id](input_tensor, vector_array, &result_1->lin, &result_2->lin, ITERATIONS);
						for (size_t i=0; i<dim; ++i) {
							printf("%zu: ", i);
							print_to_console(vector_array[i]->data, print_size);
						}
					}
				} else {
					measure_powermethod(
						algorithms_powermethod[algo_id],
						input_tensor,
						vector_array,
						number_vectors, // The original vectors should be number vectors
						&result_1->lin, &result_2->lin,
						ITERATIONS,
						file, tensor->layout[0], block_dim_length, 0);
					fflush(file);
				}

				if (TEST_ENV == 1) {
					if (algo_id == 0) {
						// printf("(normalized) model_vector vectors (size_1d_local = %zu, %d iterations):\n", ITERATIONS);
						for (size_t i=0; i<dim; ++i) {
							// double size = normalize(model_vector[i], size_1d_local);
							printf("%zu: ", i);
							print_to_console(model_vector[i]->data, print_size);
						}
					} else {
						// printf("(normalized) vector_array vectors (size_1d_local = %zu, %d iterations):\n", ITERATIONS);
						for (size_t i=0; i<dim; ++i) {
							// double size = normalize(vector_array[i], size_1d_local);
							printf("%zu: ", i);
							print_to_console(vector_array[i]->data, print_size);
						}
						printf("Verification result: ");
						lin_compare_series(model_vector, vector_array, dim);
					}
				} else {
					printf("Vectors (at most 10 el):\n");
					for (size_t i=0; i<dim; ++i) {
						printf("%zu: ", i);
						print_to_console(vector_array[i]->data, print_size);
					}
				}

			}
			
			if (TEST_ENV == 1) {
				free_tensor_storage(blocked_tensor);
				free_tensor_storage(morton_blocked_tensor);
			}

			// In case we only have one block_n to measure (!)
			#ifdef ONEBLOCKSIZE
				break;
			#endif
		}

	}

	free_tensor_storage(tensor);
	free_tensor_storage(result_1);
	free_tensor_storage(result_2);

	for (int i=0; i<dim_max; ++i) {
		if (TEST_ENV == 1) {
			free_lin_storage(model_vector[i]);
		}
		free_lin_storage(vector_array[i]);
		free_lin_storage(number_vectors[i]);
	}

	if (TEST_ENV == 0) {
		if (file != NULL) {
			fclose(file);
		}	
	}

	return 0;

}
