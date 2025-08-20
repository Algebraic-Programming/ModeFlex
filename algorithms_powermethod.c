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
#include <gen_data.h>

// #define SWITCHMODE

/////////////////////////////////////////
// Loops:
// 1) go over all powers in the method
// 2) go over all blocks
// 3) go over all dimensions (!) essentially tvLooped
// PROBLEM:
// Idea of the algorithm(!)
// Result are just temporary storages, the result is actually the vector(!!!)

// #define SINGLEBLOCK
// #define PRINT_AUX
// #define DEBUG_ENV
// #define NORMALIZE
// #define SINGLEBLOCK
// #define SINGLEVECTORUP
// #define PRINT_AUX

#define VECTOR_START 0

#if (TEST_ENV==1)				// VERIFICATION MODE

	#define NORMALIZE 			// MUST INCLUDE if want no errors of type 0.00000 different from 0.00000
	// #define NORM_TRICK
	// #define RESETS_ENABLE 	// Old requirement, now we revised our methods by using BETA parameter of BLAS
	// #define SINGLEVECTORUP
	// #define SINGLEBLOCK

#else	 						// BENCHMARKING MODE	

	#define NORMALIZE 			// MUST INCLUDE 
	// #define NORM_TRICK
	// #define SINGLEBLOCK
	// #define SINGLEVECTORUP

#endif

inline size_t
block_inc(size_t * const counters, const size_t * const thresholds, const size_t init_offset) {
	size_t offset = init_offset;
	while ( offset<=init_offset && (++counters[offset] == thresholds[offset])) {
		counters[offset--] = 0;
		// ALT IMPLEMENTATION
		//if (offset == -1) {
			//break;
		//}
	}
	return offset;
}

inline void
block_inc_fast(short * const counters, const int mode_size, const size_t dim_minus_one) {
	size_t offset = dim_minus_one;
	while ( offset<=dim_minus_one && (++counters[offset] == mode_size)) {
		counters[offset--] = 0;
	}
}

inline int ipow(int base, uint8_t exp) {
    static const uint8_t highest_bit_set[] = {
        0, 1, 2, 2, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4, 4, 4,
        5, 5, 5, 5, 5, 5, 5, 5,
        5, 5, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 255, // anything past 63 is a guaranteed overflow with base > 1
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
    };

    uint64_t result = 1;

    switch (highest_bit_set[exp]) {
    case 255: // we use 255 as an overflow marker and return 0 on overflow/underflow
        if (base == 1) {
            return 1;
        }
        
        if (base == -1) {
            return 1 - 2 * (exp & 1);
        }
        
        return 0;
    case 6:
        if (exp & 1) result *= base;
        exp >>= 1;
        base *= base;
    case 5:
        if (exp & 1) result *= base;
        exp >>= 1;
        base *= base;
    case 4:
        if (exp & 1) result *= base;
        exp >>= 1;
        base *= base;
    case 3:
        if (exp & 1) result *= base;
        exp >>= 1;
        base *= base;
    case 2:
        if (exp & 1) result *= base;
        exp >>= 1;
        base *= base;
    case 1:
        if (exp & 1) result *= base;
    default:
        return result;
    }
}


// Potential optimizations:
// reset result should be of smaller size? what size? sometimes its okay NOT to reset the result, LOL

// okay anyway changing this wont make sense (anymore); Correct (same reuslt) in both cases
void
pmModel(struct tensor_storage * restrict tensor, struct lin_storage * restrict * restrict vector_array,
	struct tensor_storage * result_1, struct tensor_storage * result_2, const int iters) {

	size_t resets = 0;
	size_t total_memory = 0;

	// printf("vectors zies are:\n");
	// for(int d=0;d<tensor->dim;++d) {
	// 	printf("sizeis%zu\n", vector_array[d]->size);
	// }
	#ifdef DEBUG_ENV
		printf("tvModel\n");
	#endif

	int save_dim = tensor->dim;
	int dim = tensor->dim;
	struct tensor_storage * restrict input;
	struct lin_storage * restrict output;
	size_t mode_size = tensor->layout[0];

	for (int iter=0; iter<iters; ++iter) {

		for (int vector_up=VECTOR_START; vector_up<dim; ++vector_up) {

			#ifdef DEBUG_ENV
				printf("vector_up=%d, \n", vector_up);
			#endif

			// in/out set for iterations == 0(!)
			int iterations = 0;
			input = tensor;
			output = &result_1->lin;

			// printf("YES ITS ME\n");
			// print_to_console(result_1->lin.data, result_1->lin.size);

			// quick check with reverse mode 

			// it's very interesitng but if we move forward along modes its' always the same mode

			size_t run_mode = 0;
			// #ifdef SWITCHMODE
			// for (int mode=0; mode<=dim-1; ++mode) { 
			// #else
			for (int mode=dim-1; mode>=0; --mode) {
			// #endif
			// for (int mode=dim-1; mode>=0; --mode) {

				if (mode == vector_up) continue;

				// WOW AMAZING ONE LINER (!) to solve my problems

				// #ifdef SWITCHMODE
				// run_mode = mode-iterations;
				// #else
				run_mode = mode;
				// #endif

				#ifdef DEBUG_ENV
				printf("compute with mode %d\n", mode);
				#endif
				if (iterations == dim-2) {
					// printf("Output is a vector!\n");
					output = vector_array[vector_up];
					reset_array(vector_array[vector_up]->data, vector_array[vector_up]->size, 0.0);
				}

				input->dim = dim-iterations;
				// printf("input %zu / vector %zu\n", input->lin.size, vector_array[0]->size);
				output->size = input->lin.size / vector_array[0]->size;

				#ifdef DEBUG_ENV
					if (tensor->lin.size > 100) {
						printf("tensor_ptr (limited to 100 out of %zu):", tensor->lin.size);
						print_to_console(input->lin.data, 100);
					} else {
						printf("tensor_ptr (of size %zu):", tensor->lin.size);
						print_to_console(input->lin.data, tensor->lin.size);
					}

					if (mode_size > 100) {
						printf("vector (limited to 100 out of %zu):", mode_size);
						print_to_console(vector_array[mode]->data, 100);
					} else {
						printf("vector (of size %zu):", mode_size);
						print_to_console(vector_array[mode]->data, mode_size);
					}

					if (output->size > 100) {
						printf("output(before) (limited to 100 out of %zu):", output->size);
						print_to_console(output->data, 100);
					} else {
						printf("output(before) (of size %zu):", output->size);
						print_to_console(output->data, output->size);
					}
				#endif

		        // size_t el=0;
		        // for (size_t k=0; k<mode_size; ++k) {
		        //   for (size_t n22=0; n22<(tensor->lin.size/mode_size);++n22){
		        //     printf("mulnumber %zu\n",el);
		        //     output->data[k] += input->lin.data[el++]*vector_array[mode]->data[n22];
		        //   }
		        // }

				tvm_vector_major_BLAS_col_mode(input, vector_array[mode], output, run_mode);
				if (vector_up == 0) total_memory += input->lin.size + input->layout[mode] + output->size;

				#ifdef DEBUG_ENV
					if (output->size > 100) {
						printf("output(after) (limited to 100 out of %zu):", output->size);
						print_to_console(output->data, 100);
					} else {
						printf("output(after) (of size %zu):", output->size);
						print_to_console(output->data, output->size);
					}
				#endif

				++iterations;
				if (iterations % 2 == 0) {
					output = &result_1->lin;
					reset_array(result_1->lin.data, result_1->lin.size, 0);
					#ifdef PRINT_AUX
						resets += result_1->lin.size;
					#endif
					input = result_2;
				} else {
					output = &result_2->lin;
					#ifdef PRINT_AUX
						resets += result_2->lin.size;
					#endif
					reset_array(result_2->lin.data, result_2->lin.size, 0);
					input = result_1;
				}
			}

			#ifdef NORMALIZE
				// printf("norm, ");
				(void) normalize(vector_array[vector_up], mode_size);
			#endif 

			#ifdef PRINT_AUX
				printf("Resets (model): %zu\n", resets);
			#endif

			#ifdef SINGLEVECTORUP
				break;
			#endif
		}

	}

	printf("total_memory touched is %zu\n", total_memory);

	tensor->dim = save_dim;
	result_2->dim = save_dim - 1;
	result_1->dim = save_dim - 1;
}


// taco code to generate this:

void
pmTaco5(const struct tensor_storage * restrict tensor, struct lin_storage * restrict * restrict vector_array,
	struct lin_storage * restrict result_1, struct lin_storage * restrict result_2, const int iters) {
	
	const int mode_size = tensor->layout[0];

	for (int32_t iA = 0; iA < mode_size; iA++) {
	  double tj = 0.0;
	  for (int32_t jA = 0; jA < mode_size; jA++) {
	    int32_t pA2 = iA * mode_size + jA;
	    double tk = 0.0;
	    for (int32_t kA = 0; kA < mode_size; kA++) {
	      int32_t pA3 = pA2 * mode_size + kA;
	      double tl = 0.0;
	      for (int32_t lA = 0; lA < mode_size; lA++) {
	        int32_t pA4 = pA3 * mode_size + lA;
	        double tm = 0.0;
	        for (int32_t mA = 0; mA < mode_size; mA++) {
	          int32_t pA5 = pA4 * mode_size + mA;
	          tm += tensor->lin.data[pA5] * vector_array[1]->data[jA] * vector_array[2]->data[kA] * vector_array[3]->data[lA] * vector_array[4]->data[mA];
	        }
	        tl += tm;
	      }
	      tk += tl;
	    }
	    tj += tk;
	  }
	  vector_array[0]->data[iA] = tj;
	}
	(void) normalize(vector_array[0], mode_size);

	for (int32_t pb = 0; pb < mode_size; pb++) {
	  vector_array[1]->data[pb] = 0.0;
	}
	for (int32_t iA = 0; iA < mode_size; iA++) {
	  double ti = vector_array[0]->data[iA];
	  for (int32_t jA = 0; jA < mode_size; jA++) {
	    int32_t pA2 = iA * mode_size + jA;
	    double tk = 0.0;
	    for (int32_t kA = 0; kA < mode_size; kA++) {
	      int32_t pA3 = pA2 * mode_size + kA;
	      double tl = 0.0;
	      for (int32_t lA = 0; lA < mode_size; lA++) {
	        int32_t pA4 = pA3 * mode_size + lA;
	        double tm = 0.0;
	        for (int32_t mA = 0; mA < mode_size; mA++) {
	          int32_t pA5 = pA4 * mode_size + mA;
	          tm += tensor->lin.data[pA5] * ti * vector_array[2]->data[kA] * vector_array[3]->data[lA] * vector_array[4]->data[mA];
	        }
	        tl += tm;
	      }
	      tk += tl;
	    }
	    vector_array[1]->data[jA] = vector_array[1]->data[jA] + tk;
	  }
	}
	(void) normalize(vector_array[1], mode_size);

	// Generated by the Tensor Algebra Compiler (tensor-compiler.org)
	for (int32_t pc = 0; pc < mode_size; pc++) {
	  vector_array[2]->data[pc] = 0.0;
	}
	for (int32_t iA = 0; iA < mode_size; iA++) {
	  double ti = vector_array[0]->data[iA];
	  for (int32_t jA = 0; jA < mode_size; jA++) {
	    int32_t pA2 = iA * mode_size + jA;
	    double tj = ti;
	    double tj0 = vector_array[1]->data[jA];
	    for (int32_t kA = 0; kA < mode_size; kA++) {
	      int32_t pA3 = pA2 * mode_size + kA;
	      double tl = 0.0;
	      for (int32_t lA = 0; lA < mode_size; lA++) {
	        int32_t pA4 = pA3 * mode_size + lA;
	        double tm = 0.0;
	        for (int32_t mA = 0; mA < mode_size; mA++) {
	          int32_t pA5 = pA4 * mode_size + mA;
	          tm += tensor->lin.data[pA5] * tj * tj0 * vector_array[3]->data[lA] * vector_array[4]->data[mA];
	        }
	        tl += tm;
	      }
	      vector_array[2]->data[kA] = vector_array[2]->data[kA] + tl;
	    }
	  }
	}
	(void) normalize(vector_array[2], mode_size);

	// Generated by the Tensor Algebra Compiler (tensor-compiler.org)
	for (int32_t pd = 0; pd < mode_size; pd++) {
	  vector_array[3]->data[pd] = 0.0;
	}
	for (int32_t iA = 0; iA < mode_size; iA++) {
	  double ti = vector_array[0]->data[iA];
	  for (int32_t jA = 0; jA < mode_size; jA++) {
	    int32_t pA2 = iA * mode_size + jA;
	    double tj = ti;
	    double tj0 = vector_array[1]->data[jA];
	    for (int32_t kA = 0; kA < mode_size; kA++) {
	      int32_t pA3 = pA2 * mode_size + kA;
	      double tk = tj;
	      double tk0 = tj0;
	      double tk1 = vector_array[2]->data[kA];
	      for (int32_t lA = 0; lA < mode_size; lA++) {
	        int32_t pA4 = pA3 * mode_size + lA;
	        double tm = 0.0;
	        for (int32_t mA = 0; mA < mode_size; mA++) {
	          int32_t pA5 = pA4 * mode_size + mA;
	          tm += tensor->lin.data[pA5] * tk * tk0 * tk1 * vector_array[4]->data[mA];
	        }
	        vector_array[3]->data[lA] = vector_array[3]->data[lA] + tm;
	      }
	    }
	  }
	}
	(void) normalize(vector_array[3], mode_size);

	// Generated by the Tensor Algebra Compiler (tensor-compiler.org)
	for (int32_t pe = 0; pe < mode_size; pe++) {
	  vector_array[4]->data[pe] = 0.0;
	}
	for (int32_t iA = 0; iA < mode_size; iA++) {
	  double ti = vector_array[0]->data[iA];
	  for (int32_t jA = 0; jA < mode_size; jA++) {
	    int32_t pA2 = iA * mode_size + jA;
	    double tj = ti;
	    double tj0 = vector_array[1]->data[jA];
	    for (int32_t kA = 0; kA < mode_size; kA++) {
	      int32_t pA3 = pA2 * mode_size + kA;
	      double tk = tj;
	      double tk0 = tj0;
	      double tk1 = vector_array[2]->data[kA];
	      for (int32_t lA = 0; lA < mode_size; lA++) {
	        int32_t pA4 = pA3 * mode_size + lA;
	        double tl = tk;
	        double tl0 = tk0;
	        double tl1 = tk1;
	        double tl2 = vector_array[3]->data[lA];
	        for (int32_t mA = 0; mA < mode_size; mA++) {
	          int32_t pA5 = pA4 * mode_size + mA;
	          vector_array[4]->data[mA] = vector_array[4]->data[mA] + tensor->lin.data[pA5] * tl * tl0 * tl1 * tl2;
	        }
	      }
	    }
	  }
	}
	(void) normalize(vector_array[4], mode_size);

}

void
pmTaco10(const struct tensor_storage * restrict tensor, struct lin_storage * restrict * restrict vector_array,
	struct lin_storage * restrict result_1, struct lin_storage * restrict result_2, const int iters) {
	
	const int mode_size = tensor->layout[0]; // FIXED (WRONG)

	size_t vector_ids [10] = {};
	int counter = 0;
	for (int i=0; i<10; ++i) {
		if (counter == VECTOR_START) {
			vector_ids[0] = counter;
		} else {
			vector_ids[i] = counter;
		}
		++counter;
	}
	print_to_console_sizet(vector_ids, 10);

	// Generated by the Tensor Algebra Compiler (tensor-compiler.org)
	for (int iA = 0; iA < mode_size; iA++) {
	  double tj = 0.0;
	  for (int jA = 0; jA < mode_size; jA++) {
	    int pA2 = iA * mode_size + jA;
	    double tk = 0.0;
	    for (int kA = 0; kA < mode_size; kA++) {
	      int pA3 = pA2 * mode_size + kA;
	      double tl = 0.0;
	      for (int lA = 0; lA < mode_size; lA++) {
	        int pA4 = pA3 * mode_size + lA;
	        double tm = 0.0;
	        for (int mA = 0; mA < mode_size; mA++) {
	          int pA5 = pA4 * mode_size + mA;
	          double tn = 0.0;
	          for (int nA = 0; nA < mode_size; nA++) {
	            int pA6 = pA5 * mode_size + nA;
	            double to = 0.0;
	            for (int oA = 0; oA < mode_size; oA++) {
	              int pA7 = pA6 * mode_size + oA;
	              double tp = 0.0;
	              for (int pA = 0; pA < mode_size; pA++) {
	                int pA8 = pA7 * mode_size + pA;
	                double tr = 0.0;
	                for (int rA = 0; rA < mode_size; rA++) {
	                  int pA9 = pA8 * mode_size + rA;
	                  double ts = 0.0;
	                  for (int sA = 0; sA < mode_size; sA++) {
	                    int pA10 = pA9 * mode_size + sA;
	                    ts += tensor->lin.data[pA10] * vector_array[vector_ids[1]]->data[jA] * vector_array[vector_ids[2]]->data[kA] * vector_array[vector_ids[3]]->data[lA] * vector_array[vector_ids[4]]->data[mA] * vector_array[vector_ids[5]]->data[nA] * 
	                    vector_array[vector_ids[6]]->data[oA] * vector_array[vector_ids[7]]->data[pA] * vector_array[vector_ids[8]]->data[rA] * vector_array[vector_ids[9]]->data[sA];
	                  }
	                  tr += ts;
	                }
	                tp += tr;
	              }
	              to += tp;
	            }
	            tn += to;
	          }
	          tm += tn;
	        }
	        tl += tm;
	      }
	      tk += tl;
	    }
	    tj += tk;
	  }
	  vector_array[0]->data[iA] = tj;
	}
	(void) normalize(vector_array[0], mode_size);

}

void
pmLooped(const struct tensor_storage * restrict tensor, struct lin_storage * restrict * restrict vector_array,
	struct lin_storage * restrict result_1, struct lin_storage * restrict result_2, const int iters) {
	
	// size_t total_memory = 0;
	#ifdef DEBUG_ENV
		printf("tvLooped - result sizes are %zu and %zu\n", result_1->size, result_2->size);
	#endif

	size_t resets = 0;

	int dim = tensor->dim;

	size_t right_size[dim];
	size_t left_size[dim];
	left_size[0] = 1;
	right_size[dim-1] = 1;
	for (int d=1; d<dim; ++d) {
		left_size[d] = left_size[d-1] * tensor->layout[d];
		// printf("we have incremented left size d(%d) to %zu be from elft size d-1(%d)= %zu\n", d,left_size[d],  d-1, left_size[d-1]);
	}	
	for (int d=dim-2; d>=0; --d) {
		right_size[d] = right_size[d+1] * tensor->layout[d];
		// printf("we have incremented right size d(%d) to %zu be from right size d+1(%d)= %zu\n", d,right_size[d],  d+1, right_size[d+1]);
	}	

	const MKL_INT mode_size = tensor->layout[0]; // FIXED (WRONG)
	const MKL_INT incx = 1; 
	const MKL_INT incy = 1; 
	const MKL_INT n2 = tensor->lin.size / mode_size;
	double alpha = 1;
	
	// double sum = 0;
	double beta = 0; // we always write from scratch (!)

	for (int j=0; j<iters; ++j) {

		#ifdef DEBUG_ENV
			printf("\niteration %d\n", j);
		#endif

		// vector up is the one being produced
		for (int vector_up=VECTOR_START; vector_up<dim; ++vector_up) {

			#ifdef DEBUG_ENV
				printf("vector_up=%d, ", vector_up);
			#endif
			size_t divisor = 1;
			// in/out set for iterations == 0(!)
			int iterations = 0;
			const double * restrict input = tensor->lin.data;
			double * restrict output = result_1->data;
			const double * restrict vector;

			#ifdef RESETS_ENABLE
				reset_array(output, n2, 0.0);
			#endif

			#ifdef PRINT_AUX
				resets += n2;
			#endif
			// reset_array(result_2->data, n2, 0.0);
			// printf("at this point result2 is cleared;");
			size_t n3 = n2;

			// double beta = 1;

			int temp_dim = dim;
			for (int mode=dim-1; mode>=0; --mode) {

				if (mode == vector_up) continue;
				#ifdef DEBUG_ENV
					printf("compute with mode %d\n", mode);
				#endif
				vector = vector_array[mode]->data;
				if (iterations == dim-2) {
					output = vector_array[vector_up]->data;
					// beta = 0;
					// reset_array(vector_array[vector_up]->data, vector_array[vector_up]->size, 0.0);
					// printf("Output is now a vector pointer vector up = %d\n", vector_up);
				}
				// printf("iteration ==================== %zu\n", iterations);

				#ifdef DEBUG_ENV
					if (tensor->lin.size > 100) {
						printf("tensor_ptr (limited to 100 out of %zu):", tensor->lin.size);
						print_to_console(input, 100);
					} else {
						printf("tensor_ptr (of size %zu):", tensor->lin.size);
						print_to_console(input, tensor->lin.size);
					}

					if (mode_size > 100) {
						printf("vector (limited to 100 out of %zu):", mode_size);
						print_to_console(vector, 100);
					} else {
						printf("vector (of size %zu):", mode_size);
						print_to_console(vector, mode_size);
					}

					if (n3 > 100) {
						printf("output (limited to 100 out of %zu):", n3);
						print_to_console(output, 100);
					} else {
						printf("output (of size %zu):", n3);
						print_to_console(output, n3);
					}
				#endif

				size_t fixed_right_size = right_size[mode] / divisor;
				if (mode != --temp_dim) {
					#ifdef DEBUG_ENV
					printf("left hand side: %zu, right_size: %zu (times mode size %zu)\n", left_size[mode], fixed_right_size, mode_size);
					#endif

					for (size_t i=0; i<left_size[mode]; ++i) {
						const double * restrict next = input + i*mode_size*fixed_right_size;
						double * restrict next_result = output + i*fixed_right_size;
						cblas_dgemv(
							CblasColMajor, 
							CblasNoTrans, 
							fixed_right_size, mode_size,
							alpha, 
							next, fixed_right_size,
							vector, incx,
							beta, 
							next_result, incy);
					}
					// if (vector_up == 0) total_memory += ((left_size[mode]*mode_size*fixed_right_size) + mode_size + (fixed_right_size*left_size[mode]));
				} else {
					// HERE? SHOULD WE UPDATE n2????
					#ifdef DEBUG_ENV
					printf("right hand side: n3=%d, mode_size=%d\n", n3, mode_size);
					#endif

					cblas_dgemv(
						CblasRowMajor, 
						CblasNoTrans, 
						n3, mode_size, 
						alpha, 
						input, mode_size,
						vector, incx,
						beta, 
						output, incy);
					// if (vector_up == 0) total_memory += ((mode_size*n3) + mode_size + (n3));

				}

				#ifdef DEBUG_ENV
					if (n3 > 100) {
						printf("output resultant (limited to 100 out of %zu):", n3);
						print_to_console(output, 100);
					} else {
						printf("output resultant (of size %zu):", n3);
						print_to_console(output, n3);
					}
				#endif

				// Super important finding: we have to clear all of it (its its the last step(!)
				// In all other steps: just clear n3 or less
				if (++iterations == dim-1) {
					break;
				}

				// We have one more iteration to go - clear the UPCOMING buffer (!)
				if (iterations % 2 == 0) {
					output = result_1->data;
					// printf("at this point result1 is cleared\n");
					// reset_array(result_1->data, n3, 0.0);
					input = result_2->data;
				} else {
					output = result_2->data;
					// reset_array(result_2->data, n3, 0.0);
					// printf("at this point result2 is cleared\n");
					input = result_1->data;
				}

				divisor *= mode_size;
				n3 = n2 / divisor;
				#ifdef RESETS_ENABLE
					reset_array(output, n3, 0.0);
				#endif
				#ifdef PRINT_AUX
					resets += n3;
				#endif
			}
			// double temp_sum = 0;
			#ifdef NORMALIZE
			(void) normalize(vector_array[vector_up], mode_size);
			#endif
			// printf("temp_sum =%f\n", temp_sum);

			// printf("temp_sum is %f, sum is %f\n", temp_sum, sum);
			// sum += temp_sum;
			// printf("after addition, sum is %f\n", sum);

			#ifdef SINGLEVECTORUP
				break;
			#endif
		}
		// printf("sum is %f\n", sum);
	}

	#ifdef PRINT_AUX
		printf("Resets: %zu\n", resets);
	#endif
	// printf("total_memory touched is %zu\n", total_memory);

}

void
pmLoopedSingle(const struct tensor_storage * restrict tensor, struct lin_storage * restrict * restrict vector_array,
	struct lin_storage * restrict result_1, struct lin_storage * restrict result_2, const int iters) {

	#ifdef DEBUG_ENV
		printf("tvLooped - result sizes are %zu and %zu\n", result_1->size, result_2->size);
	#endif
	size_t resets = 0;
	int dim = tensor->dim;
	size_t right_size[dim];
	size_t left_size[dim];
	left_size[0] = 1;
	right_size[dim-1] = 1;
	for (int d=1; d<dim; ++d) {
		left_size[d] = left_size[d-1] * tensor->layout[d];
		// printf("we have incremented left size d(%d) to %zu be from elft size d-1(%d)= %zu\n", d,left_size[d],  d-1, left_size[d-1]);
	}	
	for (int d=dim-2; d>=0; --d) {
		right_size[d] = right_size[d+1] * tensor->layout[d];
		// printf("we have incremented right size d(%d) to %zu be from right size d+1(%d)= %zu\n", d,right_size[d],  d+1, right_size[d+1]);
	}	
	const MKL_INT mode_size = tensor->layout[0]; // FIXED (WRONG)
	const MKL_INT incx = 1; 
	const MKL_INT incy = 1; 
	const MKL_INT n2 = tensor->lin.size / mode_size;
	double alpha = 1;
	double beta = 0; // we always write from scratch (!)
	for (int j=0; j<iters; ++j) {
		#ifdef DEBUG_ENV
			printf("\niteration %d\n", j);
		#endif
		for (int vector_up=VECTOR_START; vector_up<dim; ++vector_up) {
			#ifdef DEBUG_ENV
				printf("vector_up=%d, ", vector_up);
			#endif
			size_t divisor = 1;
			int iterations = 0;
			const double * restrict input = tensor->lin.data;
			double * restrict output = result_1->data;
			const double * restrict vector;
			#ifdef RESETS_ENABLE
				reset_array(output, n2, 0.0);
			#endif
			#ifdef PRINT_AUX
				resets += n2;
			#endif			
			int right_mode = dim-1;
			for (int iterations = 0; iterations < dim-1; ++iterations) {

				if (iterations == dim-2) {
					output = vector_array[vector_up]->data;
				}

				if (iterations < vector_up) {
					vector = vector_array[iterations]->data;
					#ifdef DEBUG_ENV
					printf("mode multiplied with is %d, ", iterations);
					printf("left hand side multiplication:, right size is the follwoing %zu\n", right_size[iterations]);
							printf("tensor_ptr: ");
							print_to_console(input, mode_size);
							// printf("Mul this block with vector[%d]\n", mode);
							printf("vector: ");
							print_to_console(vector, mode_size);
							printf("output (JUST BEFORE): ");
							print_to_console(output, mode_size);
					#endif
					// left_kernel[iterations](input, vector, output, NULL, NULL, NULL);
					cblas_dgemv(
						CblasColMajor,
						CblasNoTrans, 
						right_size[iterations], mode_size,
						alpha, 
						input, right_size[iterations],
						vector, incx,
						beta, 
						output, incy);

				} else {
					vector = vector_array[right_mode--]->data;
					#ifdef DEBUG_ENV
					printf("mode multiplied with is %d, ", right_mode+1);
					printf("right hand side multiplication:, right size is the follwoing %zu\n", right_size[iterations]);
							printf("tensor_ptr: ");
							print_to_console(input, mode_size);
							// printf("Mul this block with vector[%d]\n", mode);
							printf("vector: ");
							print_to_console(vector, mode_size);
							printf("output (JUST BEFORE): ");
							print_to_console(output, mode_size);
					#endif
					// right_kernel[iterations](vector, input, output, NULL, NULL, NULL);
					cblas_dgemv(
						CblasRowMajor, 
						CblasNoTrans, 
						right_size[iterations], mode_size, 
						alpha, 
						input, mode_size,
						vector, incx,
						beta, 
						output, incy);
				}

				#ifdef DEBUG_ENV
					printf("output (4 el): ");
					print_to_console(output, 4);
				#endif

				if (iterations % 2 == 0) {
					output = result_2->data;
					input = result_1->data;
				} else {
					output = result_1->data;
					input = result_2->data;
				}

				#ifdef PRINT_AUX
					executions += (right_size[iterations]*mode_size);
					resets += right_size[iterations];
				#endif
				// sum += output[el];
				#ifdef RESETS_ENABLE
					reset_array(output, right_size[iterations], 0.0);
				#endif

			}

			#ifdef NORMALIZE
			(void) normalize(vector_array[vector_up], mode_size);
			#endif
			#ifdef SINGLEVECTORUP
				break;
			#endif
		}
	}
	#ifdef PRINT_AUX
		printf("Resets: %zu\n", resets);
	#endif
}

void
pmLoopedSingleMvs(const struct tensor_storage * restrict tensor, struct lin_storage * restrict * restrict vector_array,
	struct lin_storage * restrict result_1, struct lin_storage * restrict result_2, const int iters) {

	#ifdef DEBUG_ENV
		printf("tvLooped - result sizes are %zu and %zu\n", result_1->size, result_2->size);
	#endif
	size_t resets = 0;
	int dim = tensor->dim;
	size_t right_size[dim];
	size_t left_size[dim];
	left_size[0] = 1;
	right_size[dim-1] = 1;
	for (int d=1; d<dim; ++d) {
		left_size[d] = left_size[d-1] * tensor->layout[d];
		// printf("we have incremented left size d(%d) to %zu be from elft size d-1(%d)= %zu\n", d,left_size[d],  d-1, left_size[d-1]);
	}	
	for (int d=dim-2; d>=0; --d) {
		right_size[d] = right_size[d+1] * tensor->layout[d];
		// printf("we have incremented right size d(%d) to %zu be from right size d+1(%d)= %zu\n", d,right_size[d],  d+1, right_size[d+1]);
	}	
	const MKL_INT mode_size = tensor->layout[0]; // FIXED (WRONG)
	const MKL_INT incx = 1; 
	const MKL_INT incy = 1; 
	const MKL_INT n2 = tensor->lin.size / mode_size;
	double alpha = 1;
	double beta = 0; // we always write from scratch (!)
	for (int j=0; j<iters; ++j) {
		#ifdef DEBUG_ENV
			printf("\niteration %d\n", j);
		#endif
		for (int vector_up=VECTOR_START; vector_up<dim; ++vector_up) {
			#ifdef DEBUG_ENV
				printf("vector_up=%d, ", vector_up);
			#endif
			size_t divisor = 1;
			int iterations = 0;
			const double * restrict input = tensor->lin.data;
			double * restrict output = result_1->data;
			const double * restrict vector;
			#ifdef RESETS_ENABLE
				reset_array(output, n2, 0.0);
			#endif
			#ifdef PRINT_AUX
				resets += n2;
			#endif			

			int right_mode = dim-1; // literally a mode to countdown from
			int left_mode = 0;
			for (int iterations = 0; iterations < dim-1; ++iterations) {

				if (iterations == dim-2) {
					output = vector_array[vector_up]->data;
				}

				if (right_mode > vector_up) {
					vector = vector_array[right_mode--]->data;
					#ifdef DEBUG_ENV
					printf("mode multiplied with is %d, ", right_mode+1);
					printf("right hand side multiplication:, right size is the follwoing %zu\n", right_size[iterations]);
							printf("tensor_ptr: ");
							print_to_console(input, mode_size);
							// printf("Mul this block with vector[%d]\n", mode);
							printf("vector: ");
							print_to_console(vector, mode_size);
							printf("output (JUST BEFORE): ");
							print_to_console(output, mode_size);
					#endif
					// right_kernel[iterations](vector, input, output, NULL, NULL, NULL);
					cblas_dgemv(
						CblasRowMajor, 
						CblasNoTrans, 
						right_size[iterations], mode_size, 
						alpha, 
						input, mode_size,
						vector, incx,
						beta, 
						output, incy);

				} else {
					vector = vector_array[left_mode++]->data;
					#ifdef DEBUG_ENV
					printf("mode multiplied with is %d, ", iterations);
					printf("left hand side multiplication:, right size is the follwoing %zu\n", right_size[iterations]);
							printf("tensor_ptr: ");
							print_to_console(input, mode_size);
							// printf("Mul this block with vector[%d]\n", mode);
							printf("vector: ");
							print_to_console(vector, mode_size);
							printf("output (JUST BEFORE): ");
							print_to_console(output, mode_size);
					#endif
					// left_kernel[iterations](input, vector, output, NULL, NULL, NULL);
					cblas_dgemv(
						CblasColMajor,
						CblasNoTrans, 
						right_size[iterations], mode_size,
						alpha, 
						input, right_size[iterations],
						vector, incx,
						beta, 
						output, incy);
				}

				#ifdef DEBUG_ENV
					printf("output (4 el): ");
					print_to_console(output, 4);
				#endif

				if (iterations % 2 == 0) {
					output = result_2->data;
					input = result_1->data;
				} else {
					output = result_1->data;
					input = result_2->data;
				}

				#ifdef PRINT_AUX
					executions += (right_size[iterations]*mode_size);
					resets += right_size[iterations];
				#endif
				// sum += output[el];
				#ifdef RESETS_ENABLE
					reset_array(output, right_size[iterations], 0.0);
				#endif

			}

			#ifdef NORMALIZE
			(void) normalize(vector_array[vector_up], mode_size);
			#endif
			#ifdef SINGLEVECTORUP
				break;
			#endif
		}
	}
	#ifdef PRINT_AUX
		printf("Resets: %zu\n", resets);
	#endif
}

// CURRENT PROBLEM
// vector_poiunters vs vector_Parray actually there is no differnece

// void
// pmBlock(const struct tensor_storage * restrict tensor, struct lin_storage * restrict * restrict vector_array,
// 	struct lin_storage * restrict result_1, struct lin_storage * restrict result_2, const int iters) {

// 	#ifdef DEBUG_ENV
// 		printf("tvBlock\n");
// 	#endif

// 	const int dim = tensor->dim;

// 	size_t global_vector_size = tensor->layout[0];
// 	size_t blocks = 1;
// 	size_t mul_left[dim];
// 	size_t mul_mode[dim];
// 	mul_mode[dim-1] = 1;
// 	mul_left[dim-1] = 1;
// 	for (int d=dim-1; d>=0; --d) {
// 		size_t temp = (tensor->layout[d] + tensor->block_layout[d] - 1) / tensor->block_layout[d];
// 		if (d == dim-1) {
// 			mul_mode[d] = mul_left[d];
// 			mul_left[d] = temp;
// 		} else {
// 			mul_mode[d] = mul_left[d+1];
// 			mul_left[d] = mul_left[d+1] * temp;
// 		}
// 		blocks *= temp;
// 	}

// 	#ifdef DEBUG_ENV
// 		printf("mul left: ");
// 		print_to_console_sizet(mul_left, dim);
// 		printf("mul right: ");
// 		print_to_console_sizet(mul_mode, dim);
// 	#endif

// 	size_t block_size = 1;
// 	for (int d=0; d<dim; ++d) {
// 		block_size *= tensor->block_layout[d];
// 	}

// 	////////////// THIS CODE IS FROM THE TvSingle function
// 	size_t right_size[dim];
// 	size_t left_size[dim];
// 	left_size[0] = 1;
// 	right_size[dim-1] = 1;
// 	for (int d=1; d<dim; ++d) {
// 		left_size[d] = left_size[d-1] * tensor->block_layout[d];
// 	}	
// 	for (int d=dim-2; d>=0; --d) {
// 		right_size[d] = right_size[d+1] * tensor->block_layout[d];
// 	}	

// 	#ifdef DEBUG_ENV
// 		printf("left sizes vector: ");
// 		print_to_console_sizet(left_size, dim);
// 		printf("right sizes vector: ");
// 		print_to_console_sizet(right_size, dim);
// 	#endif

// 	// PROBLEM NON SQARE (!!!!)
// 	const MKL_INT mode_size = tensor->block_layout[0]; // vector_up is the "mode"?
// 	const MKL_INT incx = 1; 
// 	const MKL_INT incy = 1; 

// 	// const MKL_INT n2 = tensor->lin.size / mode_size;
// 	const MKL_INT result_size = block_size / mode_size;

// 	double alpha = 1;
// 	double beta = 1;
// 	////////////// THIS CODE IS FROM THE TvSingle function

// 	for (int j=0; j<iters; ++j) {

// 		#ifdef DEBUG_ENV
// 			printf("\niteration %d\n", j);
// 		#endif

// 		// vector up is the one being produced
// 		for (int vector_up=VECTOR_START; vector_up<dim; ++vector_up) {

// 			#ifdef DEBUG_ENV
// 			printf("vector_up=%d\n", vector_up);
// 			#endif

// 			size_t el = 0;

// 			size_t vector_size = tensor->block_layout[vector_up]; // vector_size is the size of the block in that dim
// 			// to simply - improve this (size of each vector depends on the mode)

// 			double * restrict vector_pointers[dim]; // we have a set of pointers pointing to those vector memoriess
// 			const double * restrict tensor_ptr = tensor->lin.data; // reset the pointer to beginning
// 			for (int i=0; i<dim; ++i) {
// 				vector_pointers[i] = vector_array[i]->data;
// 			}

// 			while (1) {

// 				#ifdef DEBUG_ENV
// 							printf("block %zu, ", el);
// 				#endif
// 				// // process that block (with each vector except vector_up) producing vector_up
// 				// tvLoopedSingle(
// 				// 	tensor,
// 				// 	tensor_ptr, // current block pointer
// 				// 	vector_pointers, // it needs all "sub" vectors for this block (TODO)
// 				// 	result_1, // result vectors should be always the same (just temp)
// 				// 	result_2,
// 				// 	vector_up, // for now 0 (but should be the vector up!)
// 				// 	block_size
// 				// );

// 				size_t divisor = 1;

// 				int iterations = 0;
// 				const double * restrict input = tensor_ptr;
// 					  double * restrict output = result_1->data;
// 				const double * restrict vector;

// 				reset_array(output, result_size, 0.0);
// 				size_t n3 = result_size;

// 				int temp_dim = dim;
// 				for (int mode=dim-1; mode>=0; --mode) {
// 					if (mode == vector_up) continue;

// 					#ifdef DEBUG_ENV
// 						printf("compute with mode %d\n", mode);
// 					#endif

// 					vector = vector_pointers[mode];
// 					if (iterations == dim-2) {
// 						output = vector_pointers[vector_up];
// 						// printf("Output is now a vector pointer vector up = %d\n", vector_up);
// 					}

// 					#ifdef DEBUG_ENV
// 							printf("tensor_ptr: ");
// 							print_to_console(input, block_size);
// 							// printf("Mul this block with vector[%d]\n", mode);
// 							printf("vector: ");
// 							print_to_console(vector, mode_size);
// 							printf("output (JUST BEFORE): ");
// 							print_to_console(output, mode_size);
// 					#endif
// 					// #ifdef DEBUG_ENV
// 					// 				if (iterations == dim-2) {
// 					// 					printf("Result before its produced:\n");
// 					// 					print_to_console(output, tensor->block_layout[vector_up]);
// 					// 				} else {
// 					// 					printf("Temporary result_1 storage (intermediate storage):");
// 					// 					print_to_console(output, tensor->block_layout[vector_up]);
// 					// 				}
// 					// 				printf("Mul this block with vector[%d]\n", mode);
// 					// 				printf("Vector used to multiply this block is: \n");
// 					// 				print_to_console(vector, mode_size);
// 					// #endif

// 					size_t fixed_right_size = right_size[mode] / divisor;
// 					if (mode != --temp_dim) {
// 						#ifdef DEBUG_ENV
// 						printf("left_hand side: %zu, right_size: %zu, (mode=%zu)\n", left_size[mode], fixed_right_size, mode_size);
// 						#endif

// 						for (size_t i=0; i<left_size[mode]; ++i) {
// 							const double * restrict next = input + i*mode_size*fixed_right_size;
// 							double * restrict next_result = output + i*fixed_right_size;
// 							cblas_dgemv(
// 								CblasColMajor, 
// 								CblasNoTrans, 
// 								fixed_right_size, mode_size,
// 								alpha, 
// 								next, fixed_right_size,
// 								vector, incx,
// 								beta, 
// 								next_result, incy);
// 						}
// 					} else {
// 						#ifdef DEBUG_ENV
// 							printf("right hand side: n3=%d, mode_Size=%d\n", n3, mode_size);
// 							#endif

// 						cblas_dgemv(
// 							CblasRowMajor, 
// 							CblasNoTrans, 
// 							n3, mode_size, 
// 							alpha, 
// 							input, mode_size,
// 							vector, incx,
// 							beta, 
// 							output, incy);
// 					}

// 					#ifdef DEBUG_ENV
// 						printf("output (4 el): ");
// 						print_to_console(output, 4);
// 						if (mode != dim-1) {
// 							// print_to_console(output, fixed_right_size*left_size[mode]);
// 						} else {
// 							// print_to_console(output, n2);
// 						}
// 					#endif

// 					if (++iterations == dim-1) {
// 						break;
// 					}

// 					if (iterations % 2 == 0) {
// 						output = result_1->data;
// 						// printf("Output is now result_1\n");
// 						input = result_2->data;
// 					} else {
// 						output = result_2->data;
// 						// printf("Output is now result_2\n");
// 						// reset_array(result_2->data, result_size, 0);
// 						input = result_1->data;
// 					}

// 					divisor *= mode_size;
// 					n3 = result_size / divisor;
// 					reset_array(output, result_size, 0.0);

// 				}

// 				#ifdef SINGLEBLOCK
// 					break;
// 				#endif

// 				// OKay perhaps the sizeof reset can be limited???
// 				#ifdef DEBUG_ENV
// 					printf("we will reset the result memories to %zu (%zu / %zu)!\n", tensor->lin.size / mode_size,
// 						tensor->lin.size, mode_size);
// 				#endif
// 				// reset_array(result_1->data, result_1->size, 0);
// 				// reset_array(result_2->data, result_2->size, 0);

// 				if (++el == blocks) {
// 					break;
// 				}

// 				#ifndef SINGLEBLOCK
// 					tensor_ptr += block_size;
// 				#endif
// 				for (int d=0; d<dim; ++d) {
// 					if (el % mul_left[d] == 0) {
// 						#ifdef DEBUG_ENV
// 							printf("Resetting mode %d vector (to beginning of its memory)\n", d);
// 							#endif
// 							vector_pointers[d] = vector_array[d]->data;
// 					} else if (el % mul_mode[d] == 0) {
// 							vector_pointers[d] += vector_size;		
// 							#ifdef DEBUG_ENV
// 							printf("Move along mode %d vector by %zu elements (inc its memory)\n", d, vector_size);
// 							#endif	
// 					}	
// 				}

// 			}

// 			#ifdef NORMALIZE
// 				(void) normalize(vector_array[vector_up], global_vector_size);
// 			#endif

// 			#ifdef SINGLEVECTORUP
// 				break;
// 			#endif

// 		}
// 		// printf("\n");

// 	// #ifdef DEBUG_ENV
// 	// 	printf("normalizing all vectors after a single iteration\n");
// 	// 	printf("(normalized) vectors (4 el, %d iterations):\n", iters);
// 	// 	for (int i=0; i<dim; ++i) {
// 	// 		(void) normalize(vector_array[i], global_vector_size); // tensor->layout?
// 	// 		printf("%d: ", i);
// 	// 		print_to_console(vector_array[i]->data, 4);
// 	// 	}
// 	// #endif

// 	}

// }

void
pmMorton(const struct tensor_storage * restrict tensor, struct lin_storage * restrict * restrict vector_array,
	struct lin_storage * restrict result_1, struct lin_storage * restrict result_2, const int iters) {
	
	// size_t total_memory = 0;

	#ifdef DEBUG_ENV
		printf("tvMorton\n");
	#endif

	const int dim = tensor->dim;

	size_t global_vector_size = tensor->layout[0];
	// Morton stuff (1)
	size_t * const block_counter = calloc(dim, sizeof(size_t));
	size_t * const block_counter_threshold = calloc(dim, sizeof(size_t));
	size_t blocks = 1;

	size_t * const mul = malloc(dim * sizeof(size_t));
	mul[dim-1] = 1;
	size_t max_block = 0;

	// size_t mul_left[dim];
	// size_t mul_mode[dim];
	// mul_mode[dim-1] = 1;
	// mul_left[dim-1] = 1;

	for (int d=dim-1; d>=0; --d) {
		size_t temp = (tensor->layout[d] + tensor->block_layout[d] - 1) / tensor->block_layout[d];

		// Morton only
		block_counter_threshold[d] = temp;
		if (block_counter_threshold[d] > max_block) {
			max_block = block_counter_threshold[d];
		}

		// if (d == dim-1) {
			// mul_mode[d] = mul_left[d];
			// mul_left[d] = temp;
		// } else {
			// mul_mode[d] = mul_left[d+1];
			// mul_left[d] = mul_left[d+1] * temp;
		// }

		// Morton only
		// if (d!=0) {
		// 	mul[d-1] = mul[d] * block_counter_threshold[d];
		// }
		blocks *= temp;
	}

	// #ifdef DEBUG_ENV
	// 	printf("mul left: ");
		// print_to_console_sizet(mul_left, dim);
		// printf("mul right: ");
		// print_to_console_sizet(mul_mode, dim);
	// #endif

	size_t block_size = 1;
	for (int d=0; d<dim; ++d) {
		block_size *= tensor->block_layout[d];
	}
	// Morton stuff (2)
	// const size_t morton_block_levels = log2(max_block)+1; // round-up (take front size_teger and add 1)
	// size_t * const morton_block_indices = calloc(morton_block_levels, sizeof(size_t));
	size_t mask;
	size_t level;
	size_t inc_game;
	size_t offset;
	// int block_diff;
	// double block_diff_log;

	////////////// THIS CODE IS FROM THE TvSingle function
	size_t right_size[dim];
	size_t left_size[dim];
	left_size[0] = 1;
	right_size[dim-1] = 1;
	for (int d=1; d<dim; ++d) {
		left_size[d] = left_size[d-1] * tensor->block_layout[d];
	}	
	for (int d=dim-2; d>=0; --d) {
		right_size[d] = right_size[d+1] * tensor->block_layout[d];
	}	

	#ifdef DEBUG_ENV
		printf("left sizes vector: ");
		print_to_console_sizet(left_size, dim);
		printf("right sizes vector: ");
		print_to_console_sizet(right_size, dim);
	#endif

	// PROBLEM NON SQARE (!!!!)
	const MKL_INT mode_size = tensor->block_layout[0]; // vector_up is the "mode"?
	const MKL_INT incx = 1; 
	const MKL_INT incy = 1; 

	// const MKL_INT n2 = tensor->lin.size / mode_size;
	const MKL_INT result_size = block_size / mode_size;

	double alpha = 1;
	// double beta = 1;
	////////////// THIS CODE IS FROM THE TvSingle function

	for (int j=0; j<iters; ++j) {

		#ifdef DEBUG_ENV
			printf("\niteration %d\n", j);
		#endif

		// vector up is the one being produced
		for (int vector_up=VECTOR_START; vector_up<dim; ++vector_up) {
			reset_array(vector_array[vector_up]->data, vector_array[vector_up]->size, 0.0);

			#ifdef DEBUG_ENV
			printf("vector_up=%d\n", vector_up);
			#endif

			size_t el = 0;

			// size_t vector_size = tensor->block_layout[vector_up]; // vector_size is the size of the block in that dim
			// to simply - improve this (size of each vector depends on the mode)

			double * restrict vector_pointers[dim]; // we have a set of pointers pointing to those vector memoriess
			const double * restrict tensor_ptr = tensor->lin.data; // reset the pointer to beginning
			for (int i=0; i<dim; ++i) {
				vector_pointers[i] = vector_array[i]->data;
			}

			// printf("BEFRpore\n");
			// print_to_console_sizet(block_counter, dim);
			memset(block_counter, 0, dim*sizeof(size_t));
			// print_to_console_sizet(block_counter, dim);

			while (1) {

				#ifdef DEBUG_ENV
					printf("block %zu, ", el);
				#endif
				// // process that block (with each vector except vector_up) producing vector_up
				// tvLoopedSingle(
				// 	tensor,
				// 	tensor_ptr, // current block pointer
				// 	vector_pointers, // it needs all "sub" vectors for this block (TODO)
				// 	result_1, // result vectors should be always the same (just temp)
				// 	result_2,
				// 	vector_up, // for now 0 (but should be the vector up!)
				// 	block_size
				// );

				size_t divisor = 1;

				#ifdef DEBUG_ENV
					printf("Current tensor is:\n");
					print_to_console(tensor_ptr, 4); //block_size);
					printf("For example first element is: %f\n", tensor_ptr[0]);
					// int dim = tensor->dim;
				#endif

				int iterations = 0;

				const double * restrict input = tensor_ptr;
					  double * restrict output = result_1->data;
				const double * restrict vector;

				// reset_array(output, result_size, 0.0);
				size_t n3 = result_size;

				int temp_dim = dim;

				double beta = 0;

				for (int mode=dim-1; mode>=0; --mode) {
					if (mode == vector_up) continue;

					vector = vector_pointers[mode];
					if (iterations == dim-2) {
						beta = 1;
						output = vector_pointers[vector_up];
					}

					// #ifdef DEBUG_ENV
					// 	if (iterations == dim-2) {
					// 		// printf("Result before its produced:\n");
					// 		// print_to_console(output, tensor->block_layout[vector_up]);
					// 	} else {
					// 		// printf("Temporary result_1 storage (intermediate storage):");
					// 		// print_to_console(output, tensor->block_layout[vector_up]);
					// 	}
					// 	// printf("Mul this block with vector[%d]\n", mode);
					// 	// printf("Vector used to multiply this block is: \n");
					// 	// print_to_console(vector, mode_size);
					// #endif

					size_t fixed_right_size = right_size[mode] / divisor;
					if (mode != --temp_dim) {
						#ifdef DEBUG_ENV
							printf("left_size: %zu, right_size: %zu\n", left_size[mode], fixed_right_size);
							printf("mode multiplied with is %d, ", mode);
							printf("tensor_ptr: ");
							print_to_console(input, block_size);
							// printf("Mul this block with vector[%d]\n", mode);
							printf("vector: ");
							print_to_console(vector, mode_size);
							printf("output (JUST BEFORE): ");
							print_to_console(output, mode_size);
						#endif
						for (size_t i=0; i<left_size[mode]; ++i) {
							const double * restrict next = input + i*mode_size*fixed_right_size;
							double * restrict next_result = output + i*fixed_right_size;
							cblas_dgemv(
								CblasColMajor, 
								CblasNoTrans, 
								fixed_right_size, mode_size,
								alpha, 
								next, fixed_right_size,
								vector, incx,
								beta, 
								next_result, incy);
						}

						// if (vector_up == 0) {
							// total_memory += ((left_size[mode]*mode_size*fixed_right_size) + (fixed_right_size*left_size[mode])); // ode_size + 
							// printf("updated total memory is %zu\n", total_memory);
						// }
					} else {
						#ifdef DEBUG_ENV
							printf("result_size=%d, mode_size=%d\n", n3, mode_size);
							printf("mode multiplied with is %d, ", mode);
							printf("tensor_ptr: ");
							print_to_console(input, block_size);
							// printf("Mul this block with vector[%d]\n", mode);
							printf("vector: ");
							print_to_console(vector, mode_size);
							printf("output (JUST BEFORE): ");
							print_to_console(output, mode_size);
						#endif
						cblas_dgemv(
							CblasRowMajor, 
							CblasNoTrans, 
							n3, mode_size, 
							alpha, 
							input, mode_size,
							vector, incx,
							beta, 
							output, incy);
						// if (vector_up == 0) {
							// total_memory += ((mode_size*n3) + (n3)); // mode_size
							// printf("updated total memory is %zu\n", total_memory);
						// }
					}

					#ifdef DEBUG_ENV
						printf("output (4 el): ");
						print_to_console(output, 4);
						if (mode != dim-1) {
							// print_to_console(output, fixed_right_size*left_size[mode]);
						} else {
							// print_to_console(output, n2);
						}
					#endif


					if (++iterations == dim-1) {
						break;
					}

					if (iterations % 2 == 0) {
						output = result_1->data;
						// reset_array(result_1->data, result_size, 0);
						input = result_2->data;
					} else {
						output = result_2->data;
						// reset_array(result_2->data, result_size, 0);
						input = result_1->data;
					}

					divisor *= mode_size;
					n3 = result_size / divisor;
					// reset_array(output, n3, 0.0);

				}

				// OKay perhaps the sizeof reset can be limited???
				// reset_array(result_1->data, result_1->size, 0.0);
				// reset_array(result_2->data, result_2->size, 0.0);

				#ifdef SINGLEBLOCK
					break;
				#endif

				if (++el == blocks) {
					break;
				}

				#ifndef SINGLEBLOCK
					tensor_ptr += block_size;
				#endif
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

				// Hard core stuff - recalculate each vector(!)
				// But we could possibly have a heuristic
				// For example only one is incremented(!)
				// The rest may arbitrarily change
				for (int d=0; d<dim; ++d) {
					vector_pointers[d] = vector_array[d]->data + (block_counter[d] * tensor->block_layout[d]);
				}

			}

			#ifdef NORMALIZE
				(void) normalize(vector_array[vector_up], global_vector_size);
			#endif

			#ifdef SINGLEVECTORUP
				break;
			#endif

		}
	}

	free(block_counter);
	free(block_counter_threshold);
	// printf("total_memory touched is %zu\n", total_memory);

}

// void
// pmBlockLibx(const struct tensor_storage * restrict tensor, struct lin_storage * restrict * restrict vector_array,
// 	struct lin_storage * restrict result_1, struct lin_storage * restrict result_2, const int iters) {

// 	// initialize LIBXSMM
// 	// int prefetch = LIBXSMM_PREFETCH_AUTO;
	
// 	#ifdef DEBUG_ENV
// 		printf("tvBlock\n");
// 	#endif

// 	const int dim = tensor->dim;

// 	size_t global_vector_size = tensor->layout[0];
// 	size_t blocks = 1;
// 	size_t mul_left[dim];
// 	size_t mul_mode[dim];
// 	mul_mode[dim-1] = 1;
// 	mul_left[dim-1] = 1;
// 	for (int d=dim-1; d>=0; --d) {
// 		size_t temp = (tensor->layout[d] + tensor->block_layout[d] - 1) / tensor->block_layout[d];
// 		if (d == dim-1) {
// 			mul_mode[d] = mul_left[d];
// 			mul_left[d] = temp;
// 		} else {
// 			mul_mode[d] = mul_left[d+1];
// 			mul_left[d] = mul_left[d+1] * temp;
// 		}
// 		blocks *= temp;
// 	}

// 	#ifdef DEBUG_ENV
// 		printf("mul left: ");
// 		print_to_console_sizet(mul_left, dim);
// 		printf("mul right: ");
// 		print_to_console_sizet(mul_mode, dim);
// 	#endif

// 	size_t block_size = 1;
// 	for (int d=0; d<dim; ++d) {
// 		block_size *= tensor->block_layout[d];
// 	}

// 	////////////// THIS CODE IS FROM THE TvSingle function
// 	size_t right_size[dim];
// 	size_t left_size[dim];
// 	left_size[0] = 1;
// 	right_size[dim-1] = 1;
// 	for (int d=1; d<dim; ++d) {
// 		left_size[d] = left_size[d-1] * tensor->block_layout[d];
// 	}	
// 	for (int d=dim-2; d>=0; --d) {
// 		right_size[d] = right_size[d+1] * tensor->block_layout[d];
// 	}	

// 	#ifdef DEBUG_ENV
// 		printf("left sizes vector: ");
// 		print_to_console_sizet(left_size, dim);
// 		printf("right sizes vector: ");
// 		print_to_console_sizet(right_size, dim);
// 	#endif

// 	// PROBLEM NON SQARE (!!!!)
// 	const MKL_INT mode_size = tensor->block_layout[0]; // vector_up is the "mode"?
// 	// const MKL_INT incx = 1; 
// 	// const MKL_INT incy = 1; 

// 	// const MKL_INT n2 = tensor->lin.size / mode_size;
// 	const MKL_INT result_size = block_size / mode_size;

// 	// double alpha = 1;
// 	// double beta = 1;
// 	////////////// THIS CODE IS FROM THE TvSingle function

// 	// printf("Important -> sizesof right size\n");
// 	// print_to_console_sizet(right_size, dim);

// 	// mode > vector_up -> right_size[dim-1]
// 	// mode < vector_up -> right_size[dim-2]
// 	libxsmm_dmmfunction kernel_left_high = libxsmm_dmmdispatch(right_size[dim-1], 1, mode_size, NULL, NULL, NULL, NULL, NULL, NULL, NULL); // &prefetch);
// 	libxsmm_dmmfunction kernel_left_low = libxsmm_dmmdispatch(right_size[dim-2], 1, mode_size, NULL, NULL, NULL, NULL, NULL, NULL, NULL); // &prefetch);
// 	libxsmm_dmmfunction kernel_left;

// 	// // that's for the fixed_right_size
// 	// // result_size -> that's for 4D, we have 3D sizes of results -> equal right_size 0,1,2
// 	// actually wrong,we onlydoright handmultiplicationonce (!) for mode = dim-1 which takes place only a fewtimes
// 	libxsmm_dmmfunction kernel_right[dim-1];
// 	for (int i=0; i < dim-1; ++i) {
// 		// printf("mode = %zu, left_size[]=%zu\n", i, left_size[i+1]);
// 		kernel_right[i] = libxsmm_dmmdispatch(1, left_size[i+1], mode_size, NULL, NULL, NULL, NULL, NULL, NULL, NULL); // &prefetch);
// 	}

// 	for (int j=0; j<iters; ++j) {

// 		#ifdef DEBUG_ENV
// 			printf("\niteration %d\n", j);
// 		#endif

// 		// vector up is the one being produced
// 		for (int vector_up=VECTOR_START; vector_up<dim; ++vector_up) {

// 			#ifdef DEBUG_ENV
// 			printf("vector_up=%d\n", vector_up);
// 			#endif

// 			size_t el = 0;

// 			// size_t vector_size = tensor->block_layout[vector_up]; // vector_size is the size of the block in that dim
// 			// to simply - improve this (size of each vector depends on the mode)

// 			double * restrict vector_pointers[dim]; // we have a set of pointers pointing to those vector memoriess
// 			const double * restrict tensor_ptr = tensor->lin.data; // reset the pointer to beginning
// 			for (int i=0; i<dim; ++i) {
// 				vector_pointers[i] = vector_array[i]->data;
// 			}

// 			while (1) {

// 				#ifdef DEBUG_ENV
// 					printf("block %zu, ", el);
// 				#endif
// 				// // process that block (with each vector except vector_up) producing vector_up
// 				// tvLoopedSingle(
// 				// 	tensor,
// 				// 	tensor_ptr, // current block pointer
// 				// 	vector_pointers, // it needs all "sub" vectors for this block (TODO)
// 				// 	result_1, // result vectors should be always the same (just temp)
// 				// 	result_2,
// 				// 	vector_up, // for now 0 (but should be the vector up!)
// 				// 	block_size
// 				// );

// 				// size_t divisor = 1;

// 				int iterations = 0;
// 				const double * restrict input = tensor_ptr;
// 					  double * restrict output = result_1->data;
// 				const double * restrict vector;

// 				reset_array(output, result_size, 0.0);
// 				// size_t n3 = result_size;

// 				size_t chosen_right_size = right_size[dim-1];
// 				kernel_left = kernel_left_high;

// 				int temp_dim = dim;
// 				for (int mode=dim-1; mode>=0; --mode) {
// 					if (mode == vector_up) {
// 						chosen_right_size = right_size[dim-2];
// 						kernel_left = kernel_left_low;
// 						continue;
// 					}

// 					#ifdef DEBUG_ENV
// 						printf("compute with mode %d\n", mode);
// 					#endif

// 					vector = vector_pointers[mode];
// 					if (iterations == dim-2) {
// 						output = vector_pointers[vector_up];
// 						// printf("Output is now a vector pointer vector up = %d\n", vector_up);
// 					}

// 					#ifdef DEBUG_ENV
// 							printf("tensor_ptr: ");
// 							print_to_console(input, block_size);
// 							// printf("Mul this block with vector[%d]\n", mode);
// 							printf("vector: ");
// 							print_to_console(vector, mode_size);
// 							printf("output (JUST BEFORE): ");
// 							print_to_console(output, mode_size);
// 					#endif
// 					// #ifdef DEBUG_ENV
// 					// 				if (iterations == dim-2) {
// 					// 					printf("Result before its produced:\n");
// 					// 					print_to_console(output, tensor->block_layout[vector_up]);
// 					// 				} else {
// 					// 					printf("Temporary result_1 storage (intermediate storage):");
// 					// 					print_to_console(output, tensor->block_layout[vector_up]);
// 					// 				}
// 					// 				printf("Mul this block with vector[%d]\n", mode);
// 					// 				printf("Vector used to multiply this block is: \n");
// 					// 				print_to_console(vector, mode_size);
// 					// #endif

// 					// size_t fixed_right_size = right_size[mode] / divisor;
// 					// printf("n3=%zu, mode is %zu, right_size mode is%zu, vecotr_up is %zu and fixedright size is %zu\n", n3, mode,right_size[mode], vector_up,  fixed_right_size);
// 					if (mode != --temp_dim) {
// 						#ifdef DEBUG_ENV
// 							printf("left_size: %zu, right_size: %zu\n", left_size[mode], chosen_right_size);
// 						#endif
// 						// printf("left hand side multiplication (mode %zu): %zu, right_size: %zu\n", mode, left_size[mode], chosen_right_size);
// 						for (size_t i=0; i<left_size[mode]; ++i) {
// 							// cblas_dgemv(
// 							// 	CblasColMajor, 
// 							// 	CblasNoTrans, 
// 							// 	fixed_right_size, mode_size,
// 							// 	alpha, 
// 							// 	next, fixed_right_size,
// 							// 	vector, incx,
// 							// 	beta, 
// 							// 	next_result, incy);
// 							// if (mode > vector_up) {
// 								const double * restrict next = input + i*mode_size*chosen_right_size;
// 								double * restrict next_result = output + i*chosen_right_size;
// 								kernel_left(next, vector, next_result, NULL, NULL, NULL);
// 							// } else {
// 							// 	const double * restrict next = input + i*mode_size*chosen_right_size;
// 							// 	double * restrict next_result = output + i*chosen_right_size;
// 							// 	kernel_left(next, vector, next_result, NULL, NULL, NULL);
// 							// }
// 						}
// 					} else {
// 						#ifdef DEBUG_ENV
// 							printf("values: result_size=%d, mode_Size=%d\n", left_size[mode], mode_size);
// 						#endif
//        					// printf("right hand side multiplication (mode %zu): n2=%zu, mode_size=%zu\n", mode, left_size[mode], mode_size);
// 						// cblas_dgemv(
// 						// 	CblasRowMajor, 
// 						// 	CblasNoTrans, 
// 						// 	n3, mode_size, 
// 						// 	alpha, 
// 						// 	input, mode_size,
// 						// 	vector, incx,
// 						// 	beta, 
// 						// 	output, incy);
// 						// kernel_right[mode-1](vector, input, output, NULL, NULL, NULL);
// 						kernel_right[mode-1](vector, input, output, NULL, NULL, NULL);
// 						// insert mode 
// 					}

// 					#ifdef DEBUG_ENV
// 						printf("output (4 el): ");
// 						print_to_console(output, 4);
// 						if (mode != dim-1) {
// 							// print_to_console(output, fixed_right_size*left_size[mode]);
// 						} else {
// 							// print_to_console(output, n2);
// 						}
// 					#endif

// 					if (++iterations == dim-1) {
// 						break;
// 					}

// 					if (iterations % 2 == 0) {
// 						output = result_1->data;
// 						// printf("Output is now result_1\n");
// 						input = result_2->data;
// 					} else {
// 						output = result_2->data;
// 						// printf("Output is now result_2\n");
// 						// reset_array(result_2->data, result_size, 0);
// 						input = result_1->data;
// 					}

// 					// divisor *= mode_size;
// 					// n3 = result_size / divisor;
// 					// reset_array(output, result_size, 0.0);
// 					reset_array(output, right_size[iterations], 0.0);

// 				}

// 				// OKay perhaps the sizeof reset can be limited???
// 				#ifdef DEBUG_ENV
// 					printf("we will reset the result memories to %zu (%zu / %zu)!\n", tensor->lin.size / mode_size,
// 						tensor->lin.size, mode_size);
// 				#endif
// 				// reset_array(result_1->data, result_1->size, 0);
// 				// reset_array(result_2->data, result_2->size, 0);

// 				if (++el == blocks) {
// 					break;
// 				}

// 				#ifndef SINGLEBLOCK
// 					tensor_ptr += block_size;
// 				#endif
// 				for (int d=0; d<dim; ++d) {
// 					if (el % mul_left[d] == 0) {
// 						#ifdef DEBUG_ENV
// 							printf("Resetting mode %d vector (to beginning of its memory)\n", d);
// 							#endif
// 							vector_pointers[d] = vector_array[d]->data;
// 					} else if (el % mul_mode[d] == 0) {
// 							vector_pointers[d] += mode_size;		
// 							#ifdef DEBUG_ENV
// 							printf("Move along mode %d vector by %zu elements (inc its memory)\n", d, mode_size);
// 							#endif	
// 					}	
// 				}

// 			}

// 			#ifdef NORMALIZE
// 				(void) normalize(vector_array[vector_up], global_vector_size);
// 			#endif

// 			#ifdef SINGLEVECTORUP
// 				break;
// 			#endif

// 		}
// 		// printf("\n");

// 	// #ifdef DEBUG_ENV
// 	// 	printf("normalizing all vectors after a single iteration\n");
// 	// 	printf("(normalized) vectors (4 el, %d iterations):\n", iters);
// 	// 	for (int i=0; i<dim; ++i) {
// 	// 		(void) normalize(vector_array[i], global_vector_size); // tensor->layout?
// 	// 		printf("%d: ", i);
// 	// 		print_to_console(vector_array[i]->data, 4);
// 	// 	}
// 	// #endif

// 	}

// }

void
pmMortonLibx(const struct tensor_storage * restrict tensor, struct lin_storage * restrict * restrict vector_array,
	struct lin_storage * restrict result_1, struct lin_storage * restrict result_2, const int iters) {
		
	#ifdef DEBUG_ENV
		printf("tvMorton\n");
	#endif

	const int dim = tensor->dim;

	size_t global_vector_size = tensor->layout[0];
	// Morton stuff (1)
	size_t * const block_counter = calloc(dim, sizeof(size_t));
	size_t * const block_counter_threshold = calloc(dim, sizeof(size_t));
	size_t blocks = 1;

	size_t * const mul = malloc(dim * sizeof(size_t));
	mul[dim-1] = 1;
	size_t max_block = 0;

	// size_t mul_left[dim];
	// size_t mul_mode[dim];
	// mul_mode[dim-1] = 1;
	// mul_left[dim-1] = 1;

	for (int d=dim-1; d>=0; --d) {
		size_t temp = (tensor->layout[d] + tensor->block_layout[d] - 1) / tensor->block_layout[d];

		// Morton only
		block_counter_threshold[d] = temp;
		if (block_counter_threshold[d] > max_block) {
			max_block = block_counter_threshold[d];
		}

		// if (d == dim-1) {
			// mul_mode[d] = mul_left[d];
			// mul_left[d] = temp;
		// } else {
			// mul_mode[d] = mul_left[d+1];
			// mul_left[d] = mul_left[d+1] * temp;
		// }

		// Morton only
		// if (d!=0) {
		// 	mul[d-1] = mul[d] * block_counter_threshold[d];
		// }
		blocks *= temp;
	}

	// #ifdef DEBUG_ENV
	// 	printf("mul left: ");
		// print_to_console_sizet(mul_left, dim);
		// printf("mul right: ");
		// print_to_console_sizet(mul_mode, dim);
	// #endif

	size_t block_size = 1;
	for (int d=0; d<dim; ++d) {
		block_size *= tensor->block_layout[d];
	}

	// Morton stuff (2)
	// const size_t morton_block_levels = log2(max_block)+1; // round-up (take front size_teger and add 1)
	// size_t * const morton_block_indices = calloc(morton_block_levels, sizeof(size_t));
	size_t mask;
	size_t level;
	size_t inc_game;
	size_t offset;
	// int block_diff;
	// double block_diff_log;

	////////////// THIS CODE IS FROM THE TvSingle function
	size_t right_size[dim];
	size_t left_size[dim];
	left_size[0] = 1;
	right_size[dim-1] = 1;
	for (int d=1; d<dim; ++d) {
		left_size[d] = left_size[d-1] * tensor->block_layout[d];
	}	
	for (int d=dim-2; d>=0; --d) {
		right_size[d] = right_size[d+1] * tensor->block_layout[d];
	}	

	#ifdef DEBUG_ENV
		printf("left sizes vector: ");
		print_to_console_sizet(left_size, dim);
		printf("right sizes vector: ");
		print_to_console_sizet(right_size, dim);
	#endif

	// PROBLEM NON SQARE (!!!!)
	const MKL_INT mode_size = tensor->block_layout[0]; // vector_up is the "mode"?
	const MKL_INT incx = 1; 
	const MKL_INT incy = 1; 

	// const MKL_INT n2 = tensor->lin.size / mode_size;
	const MKL_INT result_size = block_size / mode_size;

	double alpha = 1;
	const double betta = 0.0;

	////////////// THIS CODE IS FROM THE TvSingle function

	// WAIT (betta = 0) is in most cases because we dont want to add up to the result buffer but to the original vector(!)

	// mode > vector_up -> right_size[dim-1]
	// mode < vector_up -> right_size[dim-2]
	libxsmm_dmmfunction kernel_left_high = libxsmm_dmmdispatch(right_size[dim-1], 1, mode_size, NULL, NULL, NULL, NULL, &betta, NULL, NULL); // &prefetch);
	libxsmm_dmmfunction kernel_left_low = libxsmm_dmmdispatch(right_size[dim-2], 1, mode_size, NULL, NULL, NULL, NULL, &betta, NULL, NULL); // &prefetch);
	libxsmm_dmmfunction kernel_left_low_beta = libxsmm_dmmdispatch(right_size[dim-2], 1, mode_size, NULL, NULL, NULL, NULL, NULL, NULL, NULL); // &prefetch);

	libxsmm_dmmfunction kernel_left;

	// // that's for the fixed_right_size
	// // result_size -> that's for 4D, we have 3D sizes of results -> equal right_size 0,1,2
	// actually wrong,we onlydoright handmultiplicationonce (!) for mode = dim-1 which takes place only a fewtimes
	// libxsmm_dmmfunction kernel_right = libxsmm_dmmdispatch(1, result_size, mode_size, NULL, NULL, NULL, NULL, NULL, NULL, NULL); // &prefetch);

	libxsmm_dmmfunction kernel_right[dim-1];
	for (int i=1; i < dim-1; ++i) {
		kernel_right[i] = libxsmm_dmmdispatch(1, left_size[i+1], mode_size, NULL, NULL, NULL, NULL, &betta, NULL, NULL); // &prefetch);
	}
	kernel_right[0] = libxsmm_dmmdispatch(1, left_size[1], mode_size, NULL, NULL, NULL, NULL, NULL, NULL, NULL); // &prefetch);

	// printf("Important -> sizesof right size\n");
	// print_to_console_sizet(right_size, dim);

	for (int j=0; j<iters; ++j) {

		#ifdef DEBUG_ENV
			printf("\niteration %d\n", j);
		#endif

		// vector up is the one being produced
		for (int vector_up=VECTOR_START; vector_up<dim; ++vector_up) {

			reset_array(vector_array[vector_up]->data, vector_array[vector_up]->size, 0.0);

			#ifdef DEBUG_ENV
				printf("vector_up=%d\n", vector_up);
			#endif
			// printf("vector_up=%d\n", vector_up);

			size_t el = 0;

			// size_t vector_size = tensor->block_layout[vector_up]; // vector_size is the size of the block in that dim
			// to simply - improve this (size of each vector depends on the mode)

			double * restrict vector_pointers[dim]; // we have a set of pointers pointing to those vector memoriess
			const double * restrict tensor_ptr = tensor->lin.data; // reset the pointer to beginning
			for (int i=0; i<dim; ++i) {
				vector_pointers[i] = vector_array[i]->data;
			}

			// printf("BEFRpore\n");
			// print_to_console_sizet(block_counter, dim);
			memset(block_counter, 0, dim*sizeof(size_t));
			// print_to_console_sizet(block_counter, dim);

			while (1) {

				#ifdef DEBUG_ENV
					printf("block %zu, ", el);
				#endif
				// // process that block (with each vector except vector_up) producing vector_up
				// tvLoopedSingle(
				// 	tensor,
				// 	tensor_ptr, // current block pointer
				// 	vector_pointers, // it needs all "sub" vectors for this block (TODO)
				// 	result_1, // result vectors should be always the same (just temp)
				// 	result_2,
				// 	vector_up, // for now 0 (but should be the vector up!)
				// 	block_size
				// );

				// size_t divisor = 1;

				#ifdef DEBUG_ENV
					printf("Current tensor is:\n");
					print_to_console(tensor_ptr, 4); //block_size);
					printf("For example first element is: %f\n", tensor_ptr[0]);
					// int dim = tensor->dim;
				#endif

				int iterations = 0;

				const double * restrict input = tensor_ptr;
					  double * restrict output = result_1->data;
				const double * restrict vector;

				// reset_array(output, result_size, 0.0);
				// size_t n3 = result_size;
				size_t chosen_right_size = right_size[dim-1];
				kernel_left = kernel_left_high;

				int temp_dim = dim;
							
				for (int mode=dim-1; mode>=0; --mode) {

					if (mode == vector_up) {
						chosen_right_size = right_size[dim-2];
						kernel_left = kernel_left_low;
						// printf("this had happened\n");
						continue;
					}
					// printf("mode multiplied with is %d, ", mode);

					vector = vector_pointers[mode];
					if (iterations == dim-2) {
						// beta = 1;
						kernel_left = kernel_left_low_beta;
						// printf("iterations = %zu, ", dim-2);
						output = vector_pointers[vector_up];
					}

					#ifdef DEBUG_ENV
						if (iterations == dim-2) {
							// printf("Result before its produced:\n");
							// print_to_console(output, tensor->block_layout[vector_up]);
						} else {
							// printf("Temporary result_1 storage (intermediate storage):");
							// print_to_console(output, tensor->block_layout[vector_up]);
						}
						// printf("Mul this block with vector[%d]\n", mode);
						// printf("Vector used to multiply this block is: \n");
						// print_to_console(vector, mode_size);
					#endif

					// size_t fixed_right_size = right_size[mode] / divisor;
					// printf("n3=%zu, mode is %zu, right_size mode is%zu, vecotr_up is %zu and fixedright size is %zu", n3, mode,right_size[mode], vector_up,  fixed_right_size);

					if (mode != --temp_dim) {
						#ifdef DEBUG_ENV
						printf("left_size: %zu, right_size: %zu (times mode size %zu),\n", left_size[mode], chosen_right_size, mode_size);
						#endif
						// printf("left_size: %zu, right_size: %zu (times mode size %zu)\n", left_size[mode], chosen_right_size, mode_size);

						for (size_t i=0; i<left_size[mode]; ++i) {
							// const double * restrict next = input + i*mode_size*fixed_right_size;
							// double * restrict next_result = output + i*fixed_right_size;
							// cblas_dgemv(
							// 	CblasColMajor, 
							// 	CblasNoTrans, 
							// 	fixed_right_size, mode_size,
							// 	alpha, 
							// 	next, fixed_right_size,
							// 	vector, incx,
							// 	beta, 
							// 	next_result, incy);
							// if (mode > vector_up) {
							// 	const double * restrict next = input + i*mode_size*chosen_right_size;
							// 	double * restrict next_result = output + i*chosen_right_size;
							// 	kernel_left(next, vector, next_result, NULL, NULL, NULL);
							// } else {
								const double * restrict next = input + i*mode_size*chosen_right_size;
								double * restrict next_result = output + i*chosen_right_size;
								kernel_left(next, vector, next_result);//, NULL, NULL, NULL);
								// printf("kernel_left\n");
							// }

						}
						// printf(", the size of result = %zu\n", left_size[mode]*fixed_right_size);
					} else {
						#ifdef DEBUG_ENV
							printf("result_size: %d, mode_Size: %d\n", left_size[mode], mode_size);
							#endif
							// printf("result_size: %d, mode_Size: %d\n", left_size[mode], mode_size);
						// cblas_dgemv(
						// 	CblasRowMajor, 
						// 	CblasNoTrans, 
						// 	n3, mode_size, 
						// 	alpha, 
						// 	input, mode_size,
						// 	vector, incx,
						// 	beta, 
						// 	output, incy);
							// printf("mode-1=%zu\n", mode-1);
						kernel_right[mode-1](vector, input, output);//, NULL, NULL, NULL);
						// printf(", the size of result = %zu\n", n3);

					}

					#ifdef DEBUG_ENV
						printf("output (4 el): ");
						print_to_console(output, 4);
						if (mode != dim-1) {
							// print_to_console(output, fixed_right_size*left_size[mode]);
						} else {
							// print_to_console(output, n2);
						}
					#endif

					if (++iterations == dim-1) {
						// printf("mode = %zu, ", mode);
						// printf("iterations = %zu, so we break\n", dim-1);
						break;
					}
					
					// SWITCH BETWEEN SIDES:::: SIMPLY COMMENT ? UNCOMMENT THE BELOW
					// input = result_1->data;
					if (iterations % 2 == 0) {
						output = result_1->data;
						// reset_array(result_1->data, result_size, 0);
						input = result_2->data;
					} else {
						output = result_2->data;
						// reset_array(result_2->data, result_size, 0);
						input = result_1->data;
					}

					// divisor *= mode_size;
					// n3 = result_size / divisor;
					// reset_array(output, right_size[iterations], 0.0);

				}

				// OKay perhaps the sizeof reset can be limited???
				// reset_array(result_1->data, result_1->size, 0.0);
				// reset_array(result_2->data, result_2->size, 0.0);

				if (++el == blocks) {
					break;
				}

				#ifndef SINGLEBLOCK
					tensor_ptr += block_size;
				#endif
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

				// Hard core stuff - recalculate each vector(!)
				// But we could possibly have a heuristic
				// For example only one is incremented(!)
				// The rest may arbitrarily change
				for (int d=0; d<dim; ++d) {
					vector_pointers[d] = vector_array[d]->data + (block_counter[d] * tensor->block_layout[d]);
				}

			}

			#ifdef NORMALIZE
				(void) normalize(vector_array[vector_up], global_vector_size);
			#endif

			#ifdef SINGLEVECTORUP
				break;
			#endif

		}
	}

	free(block_counter);
	free(block_counter_threshold);

}

void
pmMortonLibxVms(const struct tensor_storage * restrict tensor, struct lin_storage * restrict * restrict vector_array,
	struct lin_storage * restrict result_1, struct lin_storage * restrict result_2, const int iters) {

	#ifdef DEBUG_ENV
		printf("tvMorton\n");
	#endif

	const int dim = tensor->dim;
	size_t global_vector_size = tensor->layout[0];

	// Morton stuff (1)
	size_t * const block_counter = calloc(dim, sizeof(size_t));
	size_t * const block_counter_threshold = calloc(dim, sizeof(size_t));
	size_t blocks = 1;

	size_t * const mul = malloc(dim * sizeof(size_t));
	mul[dim-1] = 1;
	size_t max_block = 0;

	for (int d=dim-1; d>=0; --d) {
		size_t temp = (tensor->layout[d] + tensor->block_layout[d] - 1) / tensor->block_layout[d];

		// Morton only
		block_counter_threshold[d] = temp;
		if (block_counter_threshold[d] > max_block) {
			max_block = block_counter_threshold[d];
		}
		blocks *= temp;
	}

	size_t block_size = 1;
	for (int d=0; d<dim; ++d) {
		block_size *= tensor->block_layout[d];
	}
	// Morton stuff (2)
	size_t mask;
	size_t level;
	size_t inc_game;
	size_t offset;

	////////////// THIS CODE IS FROM THE TvSingle function
	size_t right_size[dim];
	size_t left_size[dim];
	left_size[0] = 1;
	right_size[dim-1] = 1;
	for (int d=1; d<dim; ++d) {
		left_size[d] = left_size[d-1] * tensor->block_layout[d];
	}	
	for (int d=dim-2; d>=0; --d) {
		right_size[d] = right_size[d+1] * tensor->block_layout[d];
	}	

	// MYSHIT
	size_t vector_sizee;
	size_t right_sizee;
	size_t left_sizee = 0;
	size_t t = 0;
	size_t out_offset = 0;
	// MYSHIT FINISHED

	#ifdef DEBUG_ENV
		printf("left sizes vector: ");
		print_to_console_sizet(left_size, dim);
		printf("right sizes vector: ");
		print_to_console_sizet(right_size, dim);
	#endif

	// PROBLEM NON SQARE (!!!!)
	const MKL_INT mode_size = tensor->block_layout[0]; // vector_up is the "mode"?
	const MKL_INT incx = 1; 
	const MKL_INT incy = 1; 
	const MKL_INT result_size = block_size / mode_size;

	double alpha = 1;

	for (int j=0; j<iters; ++j) {

		#ifdef DEBUG_ENV
			printf("\niteration %d\n", j);
		#endif

		// vector up is the one being produced
		for (int vector_up=VECTOR_START; vector_up<dim; ++vector_up) {
			reset_array(vector_array[vector_up]->data, vector_array[vector_up]->size, 0.0);

			#ifdef DEBUG_ENV
			printf("vector_up=%d\n", vector_up);
			#endif

			size_t el = 0;

			double * restrict vector_pointers[dim]; // we have a set of pointers pointing to those vector memoriess
			const double * restrict tensor_ptr = tensor->lin.data; // reset the pointer to beginning
			for (int i=0; i<dim; ++i) {
				vector_pointers[i] = vector_array[i]->data;
			}

			memset(block_counter, 0, dim*sizeof(size_t));

			while (1) {

				#ifdef DEBUG_ENV
					printf("block %zu, ", el);
				#endif

				size_t divisor = 1;

				#ifdef DEBUG_ENV
					printf("Current tensor is:\n");
					print_to_console(tensor_ptr, 4); //block_size);
					printf("For example first element is: %f\n", tensor_ptr[0]);
					// int dim = tensor->dim;
				#endif

				int iterations = 0;

				const double * restrict input = tensor_ptr;
					  double * restrict output = result_1->data;
				const double * restrict vector;

				size_t n3 = result_size;

				int temp_dim = dim;

				double beta = 0;

				for (int mode=dim-1; mode>=0; --mode) {
					if (mode == vector_up) continue;
					vector = vector_pointers[mode];

					// t = 0;
					// vector_sizee = mode_size;
					size_t fixed_right_size = right_size[mode] / divisor;

					if (iterations == dim-2) {
						output = vector_pointers[vector_up];
						beta = 1;
					}

					if (mode != --temp_dim) {

						size_t next = 0;
						size_t out_offset = 0;
						for (size_t i=0; i<left_size[mode]; ++i) {
							if (beta == 0) {
								for (size_t j_frs=0; j_frs<fixed_right_size; ++j_frs) {
									output[j_frs+out_offset] = 0;
								}
							}
							for (size_t v=0; v<mode_size; ++v) {
								for (size_t j_frs=0; j_frs<fixed_right_size; ++j_frs) {
									output[j_frs+out_offset] += input[next++] * vector[v];
								}
							}
							out_offset += fixed_right_size;
						}

					} else {

						size_t next = 0;
						for(size_t i_n3=0; i_n3<n3; ++i_n3) {
							if (beta == 0) {
								output[i_n3] = 0;	
							}
							for (size_t j_mode_size=0; j_mode_size<mode_size; ++j_mode_size) {
								output[i_n3] += input[next++] * vector[j_mode_size];
							}
						}

					}
					
					// if (iterations == dim-2) {
					// 	output = vector_pointers[vector_up];
					// 	for (size_t i=0; i<left_sizee; ++i) {
					// 		for (size_t v=0; v<vector_sizee; ++v) {
					// 			for (size_t j=0; j<right_sizee; ++j) {
					// 				output[j+out_offset] += input[t++] * vector[v];
					// 			}
					// 		}
					// 		out_offset += right_sizee;
					// 	}
					// } else {
					// 	for (size_t i=0; i<left_sizee; ++i) {
					// 		for (size_t v=0; v<vector_sizee; ++v) {
					// 			for (size_t j=0; j<right_sizee; ++j) {
					// 				output[j+out_offset] = input[t++] * vector[v];
					// 			}
					// 		}
					// 		out_offset += right_sizee;
					// 	}
					// }

					// if (mode != --temp_dim) {
					// 	#ifdef DEBUG_ENV
					// 		printf("left_size: %zu, right_size: %zu\n", left_size[mode], fixed_right_size);
					// 		printf("mode multiplied with is %d, ", mode);
					// 		printf("tensor_ptr: ");
					// 		print_to_console(input, block_size);
					// 		// printf("Mul this block with vector[%d]\n", mode);
					// 		printf("vector: ");
					// 		print_to_console(vector, mode_size);
					// 		printf("output (JUST BEFORE): ");
					// 		print_to_console(output, mode_size);
					// 	#endif
					// 		printf("left_size: %zu, right_size: %zu, mode_size=%d\n", left_size[mode], fixed_right_size, mode_size);


					// } else {
					// 	#ifdef DEBUG_ENV
					// 		printf("result_size=%d, mode_size=%d\n", n3, mode_size);
					// 		printf("mode multiplied with is %d, ", mode);
					// 		printf("tensor_ptr: ");
					// 		print_to_console(input, block_size);
					// 		// printf("Mul this block with vector[%d]\n", mode);
					// 		printf("vector: ");
					// 		print_to_console(vector, mode_size);
					// 		printf("output (JUST BEFORE): ");
					// 		print_to_console(output, mode_size);
					// 	#endif
					// 		printf("result_size=%d, mode_size=%d\n", n3, mode_size);

					// 	// cblas_dgemv(
					// 	// 	CblasRowMajor, 
					// 	// 	CblasNoTrans, 
					// 	// 	n3, mode_size, 
					// 	// 	alpha, 
					// 	// 	input, mode_size,
					// 	// 	vector, incx,
					// 	// 	beta, 
					// 	// 	output, incy);
					// }

					#ifdef DEBUG_ENV
						printf("output (4 el): ");
						print_to_console(output, 4);
					#endif

					if (++iterations == dim-1) {
						break;
					}

					if (iterations % 2 == 0) {
						output = result_1->data;
						input = result_2->data;
					} else {
						output = result_2->data;
						input = result_1->data;
					}

					divisor *= mode_size;
					n3 = result_size / divisor;
				}

				#ifdef SINGLEBLOCK
					break;
				#endif

				if (++el == blocks) {
					break;
				}

				#ifndef SINGLEBLOCK
					tensor_ptr += block_size;
				#endif
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

				for (int d=0; d<dim; ++d) {
					vector_pointers[d] = vector_array[d]->data + (block_counter[d] * tensor->block_layout[d]);
				}

			}

			#ifdef NORMALIZE
				(void) normalize(vector_array[vector_up], global_vector_size);
			#endif

			#ifdef SINGLEVECTORUP
				break;
			#endif

		}
	}

	free(block_counter);
	free(block_counter_threshold);
}

void
pmMortonLibxSingle(const struct tensor_storage * restrict tensor, struct lin_storage * restrict * restrict vector_array,
	struct lin_storage * restrict result_1, struct lin_storage * restrict result_2, const int iters) {
		
	#ifdef DEBUG_ENV
		printf("tvMortonLibxSingle\n");
	#endif

	size_t executions = 0;

	const int dim = tensor->dim;

	size_t global_vector_size = tensor->layout[0];
	// Morton stuff (1)
	size_t * const block_counter = calloc(dim, sizeof(size_t));
	size_t * const block_counter_threshold = calloc(dim, sizeof(size_t));
	size_t blocks = 1;

	size_t * const mul = malloc(dim * sizeof(size_t));
	mul[dim-1] = 1;
	size_t max_block = 0;

	for (int d=dim-1; d>=0; --d) {
		size_t temp = (tensor->layout[d] + tensor->block_layout[d] - 1) / tensor->block_layout[d];

		// Morton only
		block_counter_threshold[d] = temp;
		if (block_counter_threshold[d] > max_block) {
			max_block = block_counter_threshold[d];
		}

		blocks *= temp;
	}

	size_t block_size = 1;
	for (int d=0; d<dim; ++d) {
		block_size *= tensor->block_layout[d];
	}

	// Morton stuff (2)
	// const size_t morton_block_levels = log2(max_block)+1; // round-up (take front size_teger and add 1)
	// size_t * const morton_block_indices = calloc(morton_block_levels, sizeof(size_t));
	size_t mask;
	size_t level;
	size_t inc_game;
	size_t offset;

	////////////// THIS CODE IS FROM THE TvSingle function
	size_t right_size[dim];
	size_t left_size[dim];
	left_size[0] = 1;
	right_size[dim-1] = 1;
	for (int d=1; d<dim; ++d) {
		left_size[d] = left_size[d-1] * tensor->block_layout[d];
	}	
	for (int d=dim-2; d>=0; --d) {
		right_size[d] = right_size[d+1] * tensor->block_layout[d];
	}	

	#ifdef DEBUG_ENV
		printf("left sizes vector: ");
		print_to_console_sizet(left_size, dim);
		printf("right sizes vector: ");
		print_to_console_sizet(right_size, dim);
	#endif

	// PROBLEM NON SQARE (!!!!)
	const MKL_INT mode_size = tensor->block_layout[0]; // vector_up is the "mode"?
	const MKL_INT incx = 1; 
	const MKL_INT incy = 1; 

	// const MKL_INT n2 = tensor->lin.size / mode_size;
	const MKL_INT result_size = block_size / mode_size;

	double alpha = 1;
	double beta = 1;
	////////////// THIS CODE IS FROM THE TvSingle function


	libxsmm_dmmfunction right_kernel[dim-1];
	libxsmm_dmmfunction left_kernel[dim-1];
	const double betta = 0.0;

	libxsmm_dmmfunction right_kernel_def[dim-1];
	libxsmm_dmmfunction left_kernel_def[dim-1];

	for (int i=0; i < dim-1; ++i) {
		right_kernel[i] = libxsmm_dmmdispatch(1, right_size[i], mode_size, NULL, NULL, NULL, NULL, &betta, NULL, NULL);
		left_kernel[i] = libxsmm_dmmdispatch(right_size[i], 1, mode_size, NULL, NULL, NULL, NULL, &betta, NULL, NULL);
		right_kernel_def[i] = libxsmm_dmmdispatch(1, right_size[i], mode_size, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
		left_kernel_def[i] = libxsmm_dmmdispatch(right_size[i], 1, mode_size, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	}
	// printf("Right size corresponding to the iterations: ");
	// print_to_console_sizet(right_size, dim);

	for (int j=0; j<iters; ++j) {

		#ifdef DEBUG_ENV
			printf("\niteration %d\n", j);
		#endif

		// vector up is the one being produced
		for (int vector_up=VECTOR_START; vector_up<dim; ++vector_up) {

			reset_array(vector_array[vector_up]->data, vector_array[vector_up]->size, 0.0);

			#ifdef DEBUG_ENV
			printf("vector_up=%d\n", vector_up);
			#endif

			size_t el = 0;

			double * restrict vector_pointers[dim]; // we have a set of pointers pointing to those vector memoriess
			const double * restrict tensor_ptr = tensor->lin.data; // reset the pointer to beginning
			for (int i=0; i<dim; ++i) {
				vector_pointers[i] = vector_array[i]->data;
			}

			memset(block_counter, 0, dim*sizeof(size_t));

			while (1) {

				#ifdef DEBUG_ENV
					printf("block %zu, ", el);
				#endif
				#ifdef DEBUG_ENV
					printf("Current tensor is:\n");
					print_to_console(tensor_ptr, 4); //block_size);
					printf("For example first element is: %f\n", tensor_ptr[0]);
				#endif

				int iterations = 0;

				const double * restrict input = tensor_ptr;
					  double * restrict output = result_1->data;
				const double * restrict vector;

				#ifdef RESETS_ENABLE
					reset_array(output, result_size, 0.0);
				#endif

				int right_mode = dim-1; // literally a mode to countdown from
				////////////////////////////////////////////////////
				// IN THIS CODE WE HAVE TWO ALTERNATIVE LOOPS (!) one without a weird continue statement:
				for (int iterations = 0; iterations < dim-1; ++iterations) {
					// we only need mode to know the "real_mode"
					// vector_up tells us how many iterations we will have on the left_kernel side

					if (iterations == dim-2) {

						output = vector_pointers[vector_up];

						if (iterations < vector_up) {
							vector = vector_pointers[iterations];
							#ifdef DEBUG_ENV
							printf("mode multiplied with is %d, ", iterations);
							printf("left hand side multiplication:, right size is the follwoing %zu\n", right_size[iterations]);
									printf("tensor_ptr: ");
									print_to_console(input, block_size);
									// printf("Mul this block with vector[%d]\n", mode);
									printf("vector: ");
									print_to_console(vector, mode_size);
									printf("output (JUST BEFORE): ");
									print_to_console(output, mode_size);
							#endif
							left_kernel_def[iterations](input, vector, output);//, NULL, NULL, NULL);
						} else {
							vector = vector_pointers[right_mode--];
							#ifdef DEBUG_ENV
							printf("mode multiplied with is %d, ", right_mode+1);
							printf("right hand side multiplication:, right size is the follwoing %zu\n", right_size[iterations]);
									printf("tensor_ptr: ");
									print_to_console(input, block_size);
									// printf("Mul this block with vector[%d]\n", mode);
									printf("vector: ");
									print_to_console(vector, mode_size);
									printf("output (JUST BEFORE): ");
									print_to_console(output, mode_size);
							#endif
							right_kernel_def[iterations](vector, input, output);//, NULL, NULL, NULL);
						}

					} else {

						// output = result_1->data;

						if (iterations < vector_up) {
							vector = vector_pointers[iterations];
							#ifdef DEBUG_ENV
							printf("mode multiplied with is %d, ", iterations);
							printf("left hand side multiplication:, right size is the follwoing %zu\n", right_size[iterations]);
									printf("tensor_ptr: ");
									print_to_console(input, block_size);
									// printf("Mul this block with vector[%d]\n", mode);
									printf("vector: ");
									print_to_console(vector, mode_size);
									printf("output (JUST BEFORE): ");
									print_to_console(output, mode_size);
							#endif
							left_kernel[iterations](input, vector, output);//, NULL, NULL, NULL);
						} else {
							vector = vector_pointers[right_mode--];
							#ifdef DEBUG_ENV
							printf("mode multiplied with is %d, ", right_mode+1);
							printf("right hand side multiplication:, right size is the follwoing %zu\n", right_size[iterations]);
									printf("tensor_ptr: ");
									print_to_console(input, block_size);
									// printf("Mul this block with vector[%d]\n", mode);
									printf("vector: ");
									print_to_console(vector, mode_size);
									printf("output (JUST BEFORE): ");
									print_to_console(output, mode_size);
							#endif
							right_kernel[iterations](vector, input, output);//, NULL, NULL, NULL);
						}

					}

					#ifdef DEBUG_ENV
						printf("output (4 el): ");
						print_to_console(output, 4);
					#endif

					// input = result_1->data;
					if (iterations % 2 == 0) {
						output = result_2->data;
						input = result_1->data;
					} else {
						output = result_1->data;
						input = result_2->data;
					}
					#ifdef PRINT_AUX
						executions += (right_size[iterations]*mode_size);
					#endif
					#ifdef RESETS_ENABLE
						// reset_array(output, right_size[iterations], 0.0);
					#endif
				}

				#ifdef SINGLEBLOCK
					break;
				#endif

				////////////////////////////////////////////////////

				if (++el == blocks) {
					break;
				}

				#ifndef SINGLEBLOCK
					tensor_ptr += block_size;
				#endif
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

				for (int d=0; d<dim; ++d) {
					vector_pointers[d] = vector_array[d]->data + (block_counter[d] * tensor->block_layout[d]);
				}

			}

			#ifdef NORMALIZE
				(void) normalize(vector_array[vector_up], global_vector_size);
			#endif

			#ifdef SINGLEVECTORUP
				break;
			#endif

		}
	}

	#ifdef PRINT_AUX
		printf("Total executions: %zu, block_size: %zu, ", executions, block_size);
	#endif

	free(block_counter);
	free(block_counter_threshold);

}

// THIS IS MEMORY OF LIBX WHERE WE USED BOTH BUFFERS (!!!!)

void
pmMortonLibxSinglePREVIOUS(const struct tensor_storage * restrict tensor, struct lin_storage * restrict * restrict vector_array,
	struct lin_storage * restrict result_1, struct lin_storage * restrict result_2, const int iters) {
		
	#ifdef DEBUG_ENV
		printf("tvMortonLibxSingle\n");
	#endif

	size_t executions = 0;

	const int dim = tensor->dim;

	size_t global_vector_size = tensor->layout[0];
	// Morton stuff (1)
	size_t * const block_counter = calloc(dim, sizeof(size_t));
	size_t * const block_counter_threshold = calloc(dim, sizeof(size_t));
	size_t blocks = 1;

	size_t * const mul = malloc(dim * sizeof(size_t));
	mul[dim-1] = 1;
	size_t max_block = 0;

	for (int d=dim-1; d>=0; --d) {
		size_t temp = (tensor->layout[d] + tensor->block_layout[d] - 1) / tensor->block_layout[d];

		// Morton only
		block_counter_threshold[d] = temp;
		if (block_counter_threshold[d] > max_block) {
			max_block = block_counter_threshold[d];
		}

		blocks *= temp;
	}

	size_t block_size = 1;
	for (int d=0; d<dim; ++d) {
		block_size *= tensor->block_layout[d];
	}

	// Morton stuff (2)
	// const size_t morton_block_levels = log2(max_block)+1; // round-up (take front size_teger and add 1)
	// size_t * const morton_block_indices = calloc(morton_block_levels, sizeof(size_t));
	size_t mask;
	size_t level;
	size_t inc_game;
	size_t offset;

	////////////// THIS CODE IS FROM THE TvSingle function
	size_t right_size[dim];
	size_t left_size[dim];
	left_size[0] = 1;
	right_size[dim-1] = 1;
	for (int d=1; d<dim; ++d) {
		left_size[d] = left_size[d-1] * tensor->block_layout[d];
	}	
	for (int d=dim-2; d>=0; --d) {
		right_size[d] = right_size[d+1] * tensor->block_layout[d];
	}	

	#ifdef DEBUG_ENV
		printf("left sizes vector: ");
		print_to_console_sizet(left_size, dim);
		printf("right sizes vector: ");
		print_to_console_sizet(right_size, dim);
	#endif

	// PROBLEM NON SQARE (!!!!)
	const MKL_INT mode_size = tensor->block_layout[0]; // vector_up is the "mode"?
	const MKL_INT incx = 1; 
	const MKL_INT incy = 1; 

	// const MKL_INT n2 = tensor->lin.size / mode_size;
	const MKL_INT result_size = block_size / mode_size;

	double alpha = 1;
	double beta = 1;
	////////////// THIS CODE IS FROM THE TvSingle function

	libxsmm_dmmfunction right_kernel[dim-1];
	libxsmm_dmmfunction left_kernel[dim-1];
	for (int i=0; i < dim-1; ++i) {
		right_kernel[i] = libxsmm_dmmdispatch(1, right_size[i], mode_size, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
		left_kernel[i] = libxsmm_dmmdispatch(right_size[i], 1, mode_size, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	}
	// printf("Right size corresponding to the iterations: ");
	// print_to_console_sizet(right_size, dim);

	for (int j=0; j<iters; ++j) {

		#ifdef DEBUG_ENV
			printf("\niteration %d\n", j);
		#endif

		// vector up is the one being produced
		for (int vector_up=VECTOR_START; vector_up<dim; ++vector_up) {

			#ifdef DEBUG_ENV
			printf("vector_up=%d\n", vector_up);
			#endif

			size_t el = 0;

			double * restrict vector_pointers[dim]; // we have a set of pointers pointing to those vector memoriess
			const double * restrict tensor_ptr = tensor->lin.data; // reset the pointer to beginning
			for (int i=0; i<dim; ++i) {
				vector_pointers[i] = vector_array[i]->data;
			}

			memset(block_counter, 0, dim*sizeof(size_t));

			while (1) {

				#ifdef DEBUG_ENV
					printf("block %zu, ", el);
				#endif
				#ifdef DEBUG_ENV
					printf("Current tensor is:\n");
					print_to_console(tensor_ptr, 4); //block_size);
					printf("For example first element is: %f\n", tensor_ptr[0]);
				#endif

				int iterations = 0;

				const double * restrict input = tensor_ptr;
					  double * restrict output = result_1->data;
				const double * restrict vector;

				#ifdef RESETS_ENABLE
					reset_array(output, result_size, 0.0);
				#endif

				int right_mode = dim-1; // literally a mode to countdown from
				////////////////////////////////////////////////////
				// IN THIS CODE WE HAVE TWO ALTERNATIVE LOOPS (!) one without a weird continue statement:
				for (int iterations = 0; iterations < dim-1; ++iterations) {
					// we only need mode to know the "real_mode"
					// vector_up tells us how many iterations we will have on the left_kernel side

					if (iterations == dim-2) {
						output = vector_pointers[vector_up];
					}
					if (iterations < vector_up) {
						vector = vector_pointers[iterations];
						#ifdef DEBUG_ENV
						printf("mode multiplied with is %d, ", iterations);
						printf("left hand side multiplication:, right size is the follwoing %zu\n", right_size[iterations]);
								printf("tensor_ptr: ");
								print_to_console(input, block_size);
								// printf("Mul this block with vector[%d]\n", mode);
								printf("vector: ");
								print_to_console(vector, mode_size);
								printf("output (JUST BEFORE): ");
								print_to_console(output, mode_size);
						#endif
						left_kernel[iterations](input, vector, output);//, NULL, NULL, NULL);
					} else {
						vector = vector_pointers[right_mode--];
						#ifdef DEBUG_ENV
						printf("mode multiplied with is %d, ", right_mode);
						printf("right hand side multiplication:, right size is the follwoing %zu\n", right_size[iterations]);
								printf("tensor_ptr: ");
								print_to_console(input, block_size);
								// printf("Mul this block with vector[%d]\n", mode);
								printf("vector: ");
								print_to_console(vector, mode_size);
								printf("output (JUST BEFORE): ");
								print_to_console(output, mode_size);
						#endif
						right_kernel[iterations](vector, input, output);//, NULL, NULL, NULL);
					}
					#ifdef DEBUG_ENV
						printf("output (4 el): ");
						print_to_console(output, 4);
					#endif
					if (iterations % 2 == 0) {
						output = result_2->data;
						input = result_1->data;
					} else {
						output = result_1->data;
						input = result_2->data;
					}
					#ifdef PRINT_AUX
						executions += (right_size[iterations]*mode_size);
					#endif
					#ifdef RESETS_ENABLE
						reset_array(output, right_size[iterations], 0.0);
					#endif
				}

				////////////////////////////////////////////////////

				if (++el == blocks) {
					break;
				}

				#ifndef SINGLEBLOCK
					tensor_ptr += block_size;
				#endif
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

				for (int d=0; d<dim; ++d) {
					vector_pointers[d] = vector_array[d]->data + (block_counter[d] * tensor->block_layout[d]);
				}

			}

			#ifdef NORMALIZE
				(void) normalize(vector_array[vector_up], global_vector_size);
			#endif

			#ifdef SINGLEVECTORUP
				break;
			#endif

		}
	}

	#ifdef PRINT_AUX
		printf("Total executions: %zu, block_size: %zu, ", executions, block_size);
	#endif

	free(block_counter);
	free(block_counter_threshold);

}

void
pmMortonSingle(const struct tensor_storage * restrict tensor, struct lin_storage * restrict * restrict vector_array,
	struct lin_storage * restrict result_1, struct lin_storage * restrict result_2, const int iters) {
		
	#ifdef DEBUG_ENV
		printf("tvMortonSingle\n");
	#endif

	const int dim = tensor->dim;

	size_t resets = 0;
	size_t executions = 0;

	size_t global_vector_size = tensor->layout[0];
	// Morton stuff (1)
	size_t * const block_counter = calloc(dim, sizeof(size_t));
	size_t * const block_counter_threshold = calloc(dim, sizeof(size_t));
	size_t blocks = 1;

	size_t * const mul = malloc(dim * sizeof(size_t));
	mul[dim-1] = 1;
	size_t max_block = 0;

	for (int d=dim-1; d>=0; --d) {
		size_t temp = (tensor->layout[d] + tensor->block_layout[d] - 1) / tensor->block_layout[d];

		// Morton only
		block_counter_threshold[d] = temp;
		if (block_counter_threshold[d] > max_block) {
			max_block = block_counter_threshold[d];
		}

		blocks *= temp;
	}

	size_t block_size = 1;
	for (int d=0; d<dim; ++d) {
		block_size *= tensor->block_layout[d];
	}
	// Morton stuff (2)
	// const size_t morton_block_levels = log2(max_block)+1; // round-up (take front size_teger and add 1)
	// size_t * const morton_block_indices = calloc(morton_block_levels, sizeof(size_t));
	size_t mask;
	size_t level;
	size_t inc_game;
	size_t offset;

	////////////// THIS CODE IS FROM THE TvSingle function
	size_t right_size[dim];
	size_t left_size[dim];
	left_size[0] = 1;
	right_size[dim-1] = 1;
	for (int d=1; d<dim; ++d) {
		left_size[d] = left_size[d-1] * tensor->block_layout[d];
	}	
	for (int d=dim-2; d>=0; --d) {
		right_size[d] = right_size[d+1] * tensor->block_layout[d];
	}	

	const MKL_INT mode_size = tensor->block_layout[0]; // vector_up is the "mode"?
	const MKL_INT incx = 1; 
	const MKL_INT incy = 1; 

	const MKL_INT result_size = block_size / mode_size;

	double alpha = 1;

	libxsmm_dmmfunction right_kernel[dim-1];
	libxsmm_dmmfunction left_kernel[dim-1];
	for (int i=0; i < dim-1; ++i) {
		right_kernel[i] = libxsmm_dmmdispatch(1, right_size[i], mode_size, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
		left_kernel[i] = libxsmm_dmmdispatch(right_size[i], 1, mode_size, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	}
	for (int j=0; j<iters; ++j) {

		#ifdef DEBUG_ENV
			printf("\niteration %d\n", j);
		#endif

		// vector up is the one being produced
		for (int vector_up=VECTOR_START; vector_up<dim; ++vector_up) {

			#ifdef DEBUG_ENV
			printf("vector_up=%d\n", vector_up);
			#endif

			size_t el = 0;

			reset_array(vector_array[vector_up]->data, vector_array[vector_up]->size, 0.0);

			double * restrict vector_pointers[dim]; // we have a set of pointers pointing to those vector memoriess
			const double * restrict tensor_ptr = tensor->lin.data; // reset the pointer to beginning
			for (int i=0; i<dim; ++i) {
				vector_pointers[i] = vector_array[i]->data;
			}

			memset(block_counter, 0, dim*sizeof(size_t));

			// size_t sum = 0;

			while (1) {

				int iterations = 0;

				const double * restrict input = tensor_ptr;
					  double * restrict output = result_1->data;
				const double * restrict vector;


				#ifdef PRINT_AUX
					resets += result_size;
				#endif
				int right_mode = dim-1; // literally a mode to countdown from

				double beta = 0;

				for (int iterations = 0; iterations < dim-1; ++iterations) {

					if (iterations == dim-2) {
						output = vector_pointers[vector_up];
						beta = 1;
					}

					if (iterations < vector_up) {
						vector = vector_pointers[iterations];
						#ifdef DEBUG_ENV
						printf("mode multiplied with is %d, ", iterations);
						printf("left hand side multiplication:, right size is the follwoing %zu\n", right_size[iterations]);
								printf("tensor_ptr: ");
								print_to_console(input, block_size);
								// printf("Mul this block with vector[%d]\n", mode);
								printf("vector: ");
								print_to_console(vector, mode_size);
								printf("output (JUST BEFORE): ");
								print_to_console(output, mode_size);
						#endif
						// left_kernel[iterations](input, vector, output, NULL, NULL, NULL);
						cblas_dgemv(
							CblasColMajor,
							CblasNoTrans, 
							right_size[iterations], mode_size,
							alpha, 
							input, right_size[iterations],
							vector, incx,
							beta, 
							output, incy);

					} else {
						vector = vector_pointers[right_mode--];
						#ifdef DEBUG_ENV
						printf("mode multiplied with is %d, ", right_mode+1);
						printf("right hand side multiplication:, right size is the follwoing %zu\n", right_size[iterations]);
								printf("tensor_ptr: ");
								print_to_console(input, block_size);
								// printf("Mul this block with vector[%d]\n", mode);
								printf("vector: ");
								print_to_console(vector, mode_size);
								printf("output (JUST BEFORE): ");
								print_to_console(output, mode_size);
						#endif
						// right_kernel[iterations](vector, input, output, NULL, NULL, NULL);
						cblas_dgemv(
							CblasRowMajor, 
							CblasNoTrans, 
							right_size[iterations], mode_size, 
							alpha, 
							input, mode_size,
							vector, incx,
							beta, 
							output, incy);
					}

					#ifdef DEBUG_ENV
						printf("output (4 el): ");
						print_to_console(output, 4);
					#endif

					if (iterations % 2 == 0) {
						output = result_2->data;
						input = result_1->data;
					} else {
						output = result_1->data;
						input = result_2->data;
					}

					#ifdef PRINT_AUX
						executions += (right_size[iterations]*mode_size);
						resets += right_size[iterations];
					#endif
					// sum += output[el];
					#ifdef RESETS_ENABLE
						reset_array(output, right_size[iterations], 0.0);
					#endif

				}

				#ifdef SINGLEBLOCK
					break;
				#endif

				if (++el == blocks) {
					break;
				}

				#ifndef SINGLEBLOCK
					tensor_ptr += block_size;
				#endif

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

				for (int d=0; d<dim; ++d) {
					vector_pointers[d] = vector_array[d]->data + (block_counter[d] * tensor->block_layout[d]);
				}

			}

			// vector_array[vector_up]->data += sum;

			#ifdef NORMALIZE
				(void) normalize(vector_array[vector_up], global_vector_size);
			#endif

			#ifdef SINGLEVECTORUP
				break;
			#endif

		}
	}

	#ifdef PRINT_AUX
		// printf("Total executions: %zu, block_size: %zu, ", executions, block_size);
		printf("Resets (model): %zu\n", resets);
	#endif

	free(block_counter);
	free(block_counter_threshold);

}

void
pmMortonSingleMvs(const struct tensor_storage * restrict tensor, struct lin_storage * restrict * restrict vector_array,
	struct lin_storage * restrict result_1, struct lin_storage * restrict result_2, const int iters) {
		
	#ifdef DEBUG_ENV
		printf("tvMortonSingleMvs\n");
	#endif

	const int dim = tensor->dim;

	size_t global_vector_size = tensor->layout[0];
	// Morton stuff (1)
	size_t * const block_counter = calloc(dim, sizeof(size_t));
	size_t * const block_counter_threshold = calloc(dim, sizeof(size_t));
	size_t blocks = 1;

	size_t * const mul = malloc(dim * sizeof(size_t));
	mul[dim-1] = 1;
	size_t max_block = 0;

	for (int d=dim-1; d>=0; --d) {
		size_t temp = (tensor->layout[d] + tensor->block_layout[d] - 1) / tensor->block_layout[d];

		// Morton only
		block_counter_threshold[d] = temp;
		if (block_counter_threshold[d] > max_block) {
			max_block = block_counter_threshold[d];
		}

		blocks *= temp;
	}

	size_t block_size = 1;
	for (int d=0; d<dim; ++d) {
		block_size *= tensor->block_layout[d];
	}

	// Morton stuff (2)
	// const size_t morton_block_levels = log2(max_block)+1; // round-up (take front size_teger and add 1)
	// size_t * const morton_block_indices = calloc(morton_block_levels, sizeof(size_t));
	size_t mask;
	size_t level;
	size_t inc_game;
	size_t offset;

	////////////// THIS CODE IS FROM THE TvSingle function
	size_t right_size[dim];
	size_t left_size[dim];
	left_size[0] = 1;
	right_size[dim-1] = 1;
	for (int d=1; d<dim; ++d) {
		left_size[d] = left_size[d-1] * tensor->block_layout[d];
	}	
	for (int d=dim-2; d>=0; --d) {
		right_size[d] = right_size[d+1] * tensor->block_layout[d];
	}	

	const MKL_INT mode_size = tensor->block_layout[0]; // vector_up is the "mode"?
	const MKL_INT incx = 1; 
	const MKL_INT incy = 1; 

	const MKL_INT result_size = block_size / mode_size;

	double alpha = 1;
	// double beta = 1;

	// libxsmm_dmmfunction right_kernel[dim-1];
	// libxsmm_dmmfunction left_kernel[dim-1];
	// for (int i=0; i < dim-1; ++i) {
	// 	right_kernel[i] = libxsmm_dmmdispatch(1, right_size[i], mode_size, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	// 	left_kernel[i] = libxsmm_dmmdispatch(right_size[i], 1, mode_size, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	// }
	for (int j=0; j<iters; ++j) {

		#ifdef DEBUG_ENV
			printf("\niteration %d\n", j);
		#endif

		// vector up is the one being produced
		for (int vector_up=VECTOR_START; vector_up<dim; ++vector_up) {
			reset_array(vector_array[vector_up]->data, vector_array[vector_up]->size, 0.0);

			#ifdef DEBUG_ENV
			printf("vector_up=%d\n", vector_up);
			#endif

			size_t el = 0;

			double * restrict vector_pointers[dim]; // we have a set of pointers pointing to those vector memoriess
			const double * restrict tensor_ptr = tensor->lin.data; // reset the pointer to beginning
			for (int i=0; i<dim; ++i) {
				vector_pointers[i] = vector_array[i]->data;
			}

			memset(block_counter, 0, dim*sizeof(size_t));

			while (1) {

				int iterations = 0;

				const double * restrict input = tensor_ptr;
					  double * restrict output = result_1->data;
				const double * restrict vector;

				// reset_array(output, result_size, 0.0);
				int right_mode = dim-1; // literally a mode to countdown from
				int left_mode = 0;
				double beta = 0;
				for (int iterations = 0; iterations < dim-1; ++iterations) {

					if (iterations == dim-2) {
						beta = 1;
						output = vector_pointers[vector_up];
					}

					if (right_mode > vector_up) {
						// printf("right mode: vector taken is %zu\n", right_mode);
						vector = vector_pointers[right_mode--];
						// right_kernel[iterations](vector, input, output, NULL, NULL, NULL);
						cblas_dgemv(
							CblasRowMajor, 
							CblasNoTrans, 
							right_size[iterations], mode_size, 
							alpha, 
							input, mode_size,
							vector, incx,
							beta, 
							output, incy);

					} else {
						// printf("left mode: vector taken is %d\n", left_mode);
						vector = vector_pointers[left_mode++];
						// left_kernel[iterations](input, vector, output, NULL, NULL, NULL);
						cblas_dgemv(
							CblasColMajor,
							CblasNoTrans, 
							right_size[iterations], mode_size,
							alpha, 
							input, right_size[iterations],
							vector, incx,
							beta, 
							output, incy);
					}

					#ifdef DEBUG_ENV
						printf("output (4 el): ");
						print_to_console(output, 4);
					#endif
					if (iterations % 2 == 0) {
						output = result_2->data;
						input = result_1->data;
					} else {
						output = result_1->data;
						input = result_2->data;
					}
					// reset_array(output, right_size[iterations], 0.0);
				}

				#ifdef SINGLEBLOCK
					break;
				#endif

				if (++el == blocks) {
					break;
				}

				#ifndef SINGLEBLOCK
					tensor_ptr += block_size;
				#endif
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

				for (int d=0; d<dim; ++d) {
					vector_pointers[d] = vector_array[d]->data + (block_counter[d] * tensor->block_layout[d]);
				}

			}

			#ifdef NORMALIZE
				(void) normalize(vector_array[vector_up], global_vector_size);
			#endif

			#ifdef SINGLEVECTORUP
				break;
			#endif

		}
	}

	free(block_counter);
	free(block_counter_threshold);

}


void
pmMortonLibxSingleMvs(const struct tensor_storage * restrict tensor, struct lin_storage * restrict * restrict vector_array,
	struct lin_storage * restrict result_1, struct lin_storage * restrict result_2, const int iters) {
		
	#ifdef DEBUG_ENV
		printf("tvMortonLibxSingleMvs\n");
	#endif

	const int dim = tensor->dim;

	size_t global_vector_size = tensor->layout[0];
	// Morton stuff (1)
	size_t * const block_counter = calloc(dim, sizeof(size_t));
	size_t * const block_counter_threshold = calloc(dim, sizeof(size_t));
	size_t blocks = 1;

	size_t * const mul = malloc(dim * sizeof(size_t));
	mul[dim-1] = 1;
	size_t max_block = 0;

	for (int d=dim-1; d>=0; --d) {
		size_t temp = (tensor->layout[d] + tensor->block_layout[d] - 1) / tensor->block_layout[d];

		// Morton only
		block_counter_threshold[d] = temp;
		if (block_counter_threshold[d] > max_block) {
			max_block = block_counter_threshold[d];
		}

		blocks *= temp;
	}

	size_t block_size = 1;
	for (int d=0; d<dim; ++d) {
		block_size *= tensor->block_layout[d];
	}

	// Morton stuff (2)
	// const size_t morton_block_levels = log2(max_block)+1; // round-up (take front size_teger and add 1)
	// size_t * const morton_block_indices = calloc(morton_block_levels, sizeof(size_t));
	size_t mask;
	size_t level;
	size_t inc_game;
	size_t offset;

	////////////// THIS CODE IS FROM THE TvSingle function
	size_t right_size[dim];
	size_t left_size[dim];
	left_size[0] = 1;
	right_size[dim-1] = 1;
	for (int d=1; d<dim; ++d) {
		left_size[d] = left_size[d-1] * tensor->block_layout[d];
	}	
	for (int d=dim-2; d>=0; --d) {
		right_size[d] = right_size[d+1] * tensor->block_layout[d];
	}	

	const MKL_INT mode_size = tensor->block_layout[0]; // vector_up is the "mode"?
	const MKL_INT incx = 1; 
	const MKL_INT incy = 1; 

	const MKL_INT result_size = block_size / mode_size;

	double alpha = 1;
	// double beta = 1;

	libxsmm_dmmfunction right_kernel[dim-1];
	libxsmm_dmmfunction left_kernel[dim-1];
	double const betta = 0.0;
	for (int i=0; i < dim-2; ++i) {
		right_kernel[i] = libxsmm_dmmdispatch(1, right_size[i], mode_size, NULL, NULL, NULL, NULL, &betta, NULL, NULL);
		left_kernel[i] = libxsmm_dmmdispatch(right_size[i], 1, mode_size, NULL, NULL, NULL, NULL, &betta, NULL, NULL);
	}
	right_kernel[dim-2] = libxsmm_dmmdispatch(1, right_size[dim-2], mode_size, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
	left_kernel[dim-2] = libxsmm_dmmdispatch(right_size[dim-2], 1, mode_size, NULL, NULL, NULL, NULL, NULL, NULL, NULL);

	for (int j=0; j<iters; ++j) {

		#ifdef DEBUG_ENV
			printf("\niteration %d\n", j);
		#endif

		// vector up is the one being produced
		for (int vector_up=VECTOR_START; vector_up<dim; ++vector_up) {
			reset_array(vector_array[vector_up]->data, vector_array[vector_up]->size, 0.0);

			#ifdef DEBUG_ENV
			printf("vector_up=%d\n", vector_up);
			#endif

			size_t el = 0;

			double * restrict vector_pointers[dim]; // we have a set of pointers pointing to those vector memoriess
			const double * restrict tensor_ptr = tensor->lin.data; // reset the pointer to beginning
			for (int i=0; i<dim; ++i) {
				vector_pointers[i] = vector_array[i]->data;
			}

			memset(block_counter, 0, dim*sizeof(size_t));

			while (1) {

				int iterations = 0;

				const double * restrict input = tensor_ptr;
					  double * restrict output = result_1->data;
				const double * restrict vector;

				// reset_array(output, result_size, 0.0);
				int right_mode = dim-1; // literally a mode to countdown from
				int left_mode = 0;
				for (int iterations = 0; iterations < dim-1; ++iterations) {

					if (iterations == dim-2) {
						output = vector_pointers[vector_up];
					}
					// } else {
					// 	output = result_1->data;
					// }

					if (right_mode > vector_up) {
						// printf("right mode: vector taken is %zu\n", right_mode);
						vector = vector_pointers[right_mode--];
						right_kernel[iterations](vector, input, output);//, NULL, NULL, NULL);
						// cblas_dgemv(
						// 	CblasRowMajor, 
						// 	CblasNoTrans, 
						// 	right_size[iterations], mode_size, 
						// 	alpha, 
						// 	input, mode_size,
						// 	vector, incx,
						// 	beta, 
						// 	output, incy);

					} else {
						// printf("left mode: vector taken is %d\n", left_mode);
						vector = vector_pointers[left_mode++];
						left_kernel[iterations](input, vector, output);//, NULL, NULL, NULL);
						// cblas_dgemv(
							// CblasColMajor,
							// CblasNoTrans, 
							// right_size[iterations], mode_size,
							// alpha,
							// input, right_size[iterations],
							// vector, incx,
							// beta, 
							// output, incy);
					}

					#ifdef DEBUG_ENV
						printf("output (4 el): ");
						print_to_console(output, 4);
					#endif
					
					if (iterations % 2 == 0) {
						output = result_2->data;
						input = result_1->data;
					} else {
						output = result_1->data;
						input = result_2->data;
					}
					// input = result_1->data;
					// reset_array(output, right_size[iterations], 0.0);
				}

				if (++el == blocks) {
					break;
				}

				#ifndef SINGLEBLOCK
					tensor_ptr += block_size;
				#endif
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

				for (int d=0; d<dim; ++d) {
					vector_pointers[d] = vector_array[d]->data + (block_counter[d] * tensor->block_layout[d]);
				}

			}

			#ifdef NORMALIZE
				(void) normalize(vector_array[vector_up], global_vector_size);
			#endif

			#ifdef SINGLEVECTORUP
				break;
			#endif

		}
	}

	free(block_counter);
	free(block_counter_threshold);
}

// Updates: 
// no needf or result hte temporary storages

void
pmMortonMyself(const struct tensor_storage * restrict tensor, struct lin_storage * restrict * restrict vector_array,
	struct lin_storage * restrict result_1, struct lin_storage * restrict result_2, const int iters) {
	
	// size_t total_memory = 0;

	#ifdef DEBUG_ENV
		printf("tvMorton\n");
	#endif

	const int dim = tensor->dim;

	size_t global_vector_size = tensor->layout[0];
	// Morton stuff (1)
	size_t * const block_counter = calloc(dim, sizeof(size_t));
	size_t * const block_counter_threshold = calloc(dim, sizeof(size_t));
	size_t blocks = 1;

	size_t * const mul = malloc(dim * sizeof(size_t));
	mul[dim-1] = 1;
	size_t max_block = 0;

	// size_t mul_left[dim];
	// size_t mul_mode[dim];
	// mul_mode[dim-1] = 1;
	// mul_left[dim-1] = 1;

	for (int d=dim-1; d>=0; --d) {
		size_t temp = (tensor->layout[d] + tensor->block_layout[d] - 1) / tensor->block_layout[d];

		// Morton only
		block_counter_threshold[d] = temp;
		if (block_counter_threshold[d] > max_block) {
			max_block = block_counter_threshold[d];
		}

		// if (d == dim-1) {
			// mul_mode[d] = mul_left[d];
			// mul_left[d] = temp;
		// } else {
			// mul_mode[d] = mul_left[d+1];
			// mul_left[d] = mul_left[d+1] * temp;
		// }

		// Morton only
		// if (d!=0) {
		// 	mul[d-1] = mul[d] * block_counter_threshold[d];
		// }
		blocks *= temp;
	}

	// #ifdef DEBUG_ENV
	// 	printf("mul left: ");
		// print_to_console_sizet(mul_left, dim);
		// printf("mul right: ");
		// print_to_console_sizet(mul_mode, dim);
	// #endif

	int block_size = 1;
	for (int d=0; d<dim; ++d) {
		block_size *= tensor->block_layout[d];
	}

	size_t executions = 0;
	// Morton stuff (2)
	// const size_t morton_block_levels = log2(max_block)+1; // round-up (take front size_teger and add 1)
	// size_t * const morton_block_indices = calloc(morton_block_levels, sizeof(size_t));
	size_t mask;
	size_t level;
	size_t inc_game;
	size_t offset;
	// int block_diff;
	// double block_diff_log;
	
	////////////// THIS CODE IS FROM THE TvSingle function
	size_t right_size[dim];
	size_t left_size[dim];
	left_size[0] = 1;
	right_size[dim-1] = 1;
	for (int d=1; d<dim; ++d) {
		left_size[d] = left_size[d-1] * tensor->block_layout[d];
	}	
	for (int d=dim-2; d>=0; --d) {
		right_size[d] = right_size[d+1] * tensor->block_layout[d];
	}	

	#ifdef DEBUG_ENV
		printf("left sizes vector: ");
		print_to_console_sizet(left_size, dim);
		printf("right sizes vector: ");
		print_to_console_sizet(right_size, dim);
	#endif

	// PROBLEM NON SQARE (!!!!)
	const MKL_INT mode_size = tensor->block_layout[0]; // vector_up is the "mode"?
	const MKL_INT incx = 1; 
	const MKL_INT incy = 1; 

	// const MKL_INT n2 = tensor->lin.size / mode_size;
	const MKL_INT result_size = block_size / mode_size;

	double alpha = 1;
	double beta = 1;
	////////////// THIS CODE IS FROM THE TvSingle function

	// do some memoization here
	int mul_results[dim];
	short coords2[dim];
	for (int mode=dim-1; mode>=0; --mode) {
		mul_results[mode] = ipow(mode_size, (dim-1-mode));
		coords2[mode] = 0;
	}

	// size_t total_memory = 0;

	for (int j=0; j<iters; ++j) {

		#ifdef DEBUG_ENV
			printf("\niteration %d\n", j);
		#endif

		// vector up is the one being produced
		for (int vector_up=VECTOR_START; vector_up<dim; ++vector_up) {

			#ifdef DEBUG_ENV
			printf("vector_up=%d\n", vector_up);
			#endif

			size_t block_el = 0;

			// size_t vector_size = tensor->block_layout[vector_up]; // vector_size is the size of the block in that dim
			// to simply - improve this (size of each vector depends on the mode)

			double * restrict vector_pointers[dim]; // we have a set of pointers pointing to those vector memoriess
			const double * restrict tensor_ptr = tensor->lin.data; // reset the pointer to beginning
			for (int i=0; i<dim; ++i) {
				vector_pointers[i] = vector_array[i]->data;
			}

			// printf("BEFRpore\n");
			// print_to_console_sizet(block_counter, dim);
			memset(block_counter, 0, dim*sizeof(size_t));
			// print_to_console_sizet(block_counter, dim);

			// size_t sum = 0;

			while (1) {

				#ifdef DEBUG_ENV
					printf("block %zu, ", block_el);
				#endif
				// // process that block (with each vector except vector_up) producing vector_up
				// tvLoopedSingle(
				// 	tensor,
				// 	tensor_ptr, // current block pointer
				// 	vector_pointers, // it needs all "sub" vectors for this block (TODO)
				// 	result_1, // result vectors should be always the same (just temp)
				// 	result_2,
				// 	vector_up, // for now 0 (but should be the vector up!)
				// 	block_size
				// );
				// size_t divisor = 1;

				#ifdef DEBUG_ENV
					printf("Current tensor is:\n");
					print_to_console(tensor_ptr, 4); //block_size);
					printf("For example first element is: %f\n", tensor_ptr[0]);
					// int dim = tensor->dim;
				#endif

				int iterations = 0;
				// const double * restrict input = tensor_ptr;
					  // double * restrict output = result_1->data;
				// const double * restrict vector;
				// reset_array(output, result_size, 0.0);
				// size_t n3 = result_size;
				// int temp_dim = dim;

				// rightmost: bump the guy if % 1==0 so everytime, or in absolute terms its result of [el]%3
				// then: bump the guy if % 3==0, or in absolute terms [el/3]%3, etc
				// formula is: 
				
				#ifdef DEBUG_ENV
					printf("tensor_ptr: ");
					print_to_console(tensor_ptr, block_size);
					// printf("Mul this block with vector[%d]\n", mode);
					printf("vectors:\n");
					for (int i=0; i<=dim-1; ++i) {
						if (i == vector_up) continue;
						printf("[%d]:       ", i);
						print_to_console(vector_pointers[i], mode_size);
					}
					printf("output (JUST BEFORE) [%d]: ", vector_up);
					print_to_console(vector_pointers[vector_up], mode_size);
					printf("\n block: %d, (vector_up=%d)\n", block_el, vector_up);
				#endif

				for (int el=0; el<block_size; ++el) {
					#ifdef PRINT_AUX
					++executions;
					#endif

					short temp_el = el % mode_size;

					// but we can use the truth THAT its a final addition, ah no not really....

					vector_pointers[0][temp_el] += tensor_ptr[el] * vector_pointers[1][temp_el]
					* vector_pointers[2][temp_el]
					* vector_pointers[3][temp_el];




					// vector_pointers[0][0] += tensor_ptr[el] * vector_pointers[1][0];



				}

				// if (vector_up == 0) {
					// for (int el=0; el<block_size; ++el) {
					// 	vector_pointers[0][el % mode_size] += tensor_ptr[el] * vector_pointers[1][el % mode_size];
						// vector_pointers[0][0] += (
						// 	tensor_ptr[el] * vector_pointers[1][0] * vector_pointers[2][0]
						// 	* vector_pointers[3][0] * vector_pointers[4][0] * vector_pointers[5][0]);
						// vector_pointers[vector_up][coords2[vector_up]] += tensor_ptr[el] * vector_pointers[1][coords2[1]] * vector_pointers[2][coords2[2]]
						// 	* vector_pointers[3][coords2[3]] * vector_pointers[4][coords2[4]] * vector_pointers[5][coords2[5]];
						// block_inc_fast(coords2, mode_size, dim-1);

						// for (int k=0; k<dim; ++k) {
						// 	printf("%d, ", coords2[k]);
						// }
						// printf("el done \n");
						// el % mode_size
						// (el / mode_size) % mode_size 
						// (el / mode_size / mode_size) % mode_size 

						// vector_pointers[0][(el / mode_size / mode_size / mode_size / mode_size / mode_size) % mode_size] += 
						// tensor_ptr[el] * 
						// vector_pointers[1][(el / mode_size / mode_size / mode_size / mode_size) % mode_size] * 
						// vector_pointers[2][(el / mode_size / mode_size / mode_size) % mode_size] * 
						// vector_pointers[3][(el / mode_size / mode_size) % mode_size] * 
						// vector_pointers[4][(el / mode_size) % mode_size] * 
						// vector_pointers[5][el % mode_size];
					// }
				// } else if (vector_up == 1) {
				// 	for (int el=0; el<block_size; ++el) {
				// 		vector_pointers[vector_up][coords2[vector_up]] += tensor_ptr[el] * vector_pointers[0][coords2[0]] * vector_pointers[2][coords2[2]];
				// 		block_inc_fast(coords2, mode_size, dim-1);
				// 	}
				// } else {
				// 	for (int el=0; el<block_size; ++el) {
				// 		vector_pointers[vector_up][coords2[vector_up]] += tensor_ptr[el] * vector_pointers[0][coords2[0]] * vector_pointers[1][coords2[1]];
				// 		block_inc_fast(coords2, mode_size, dim-1);
				// 	}
				// }









// alternative code to adapt

// size_t
// block_inc(size_t * const counters, const size_t * const thresholds, const size_t init_offset) {
// 	size_t offset = init_offset;
// 	while ( offset<=init_offset && (++counters[offset] == thresholds[offset])) {
// 		counters[offset--] = 0;
// 		// ALT IMPLEMENTATION
// 		//if (offset == -1) {
// 			//break;
// 		//}
// 	}
// 	return offset;
// }

				// old version (!)
					// we need a version in which the code is not varied by the dimensionality(!)
					// i.e. we always write to the same vector_ptr ???
					// then less code repepetitions (can be reused between vector_ups!)
					// OR
					// single statement then we can write

				// } else {

				// 	int temp_el = 0;
				// 	for (int el=0; el<block_size; ++el) {
				// 		double temp_result = 0;
				// 		temp_result += tensor_ptr[el];
				// 		// printf("el=%d\n", el);
				// 		// printf("out_index = %d\n", out_index);
				// 		// printf("%lf = %lf, \n", vector_pointers[vector_up][out_index],
				// 		for (int mode=0; mode<vector_up; ++mode) {
				// 			int in_index = (el / mul_results[mode]) % mode_size;
				// 			temp_result *= vector_pointers[mode][in_index];
				// 			// printf("%d, ", in_index);
				// 		}
				// 		for (int mode=vector_up+1; mode<dim; ++mode) {
				// 			int in_index = (el / mul_results[mode]) % mode_size;
				// 			temp_result *= vector_pointers[mode][in_index];
				// 			// printf("%d, ", in_index);
				// 		}
				// 		// printf("\n");
				// 		int out_index = (el / mul_results[vector_up]) % mode_size;
				// 		vector_pointers[vector_up][out_index] += temp_result;
				// 	}

				// }




				// if (vector_up == 0) total_memory += (block_size) + mode_size + mode_size;

					// vector_pointers[vector_up][out_index] *= tensor_ptr[el];
				// 	for (int mode=dim-1; mode>=0; --mode) {
				// 		if (mode == vector_up) {
				// 			continue;
				// 		} else {
				// 		// printf("%lf = %lf times %lf, \n",
				// 		// vector_pointers[vector_up][out_index]*vector_pointers[mode][in_index],
				// 		// vector_pointers[mode][in_index],
				// 		// vector_pointers[vector_up][out_index]);
				// 		// vector_pointers[vector_up][out_index] *= vector_pointers[mode][in_index];
				// 		int in_index = (el / mul_results[mode]) % mode_size;
				// 		temp_result *= vector_pointers[mode][in_index];
				// 		// printf("mode[%d], in_index = %d\n", mode, in_index);
				// 		}
				// 	}
				// 	int out_index = (el / mul_results[vector_up]) % mode_size;
				// 	vector_pointers[vector_up][out_index] += temp_result;
				// }

				// int temp_el = 0;
				// for (int el=0; el<block_size; ++el) {
				// 	double temp_result = 0;
				// 	// printf("el=%d\n", el);
					
				// 	// printf("out_index = %d\n", out_index);
				// 	// printf("%lf = %lf, \n", vector_pointers[vector_up][out_index],
				// 	temp_result += tensor_ptr[el];
				// 	// vector_pointers[vector_up][out_index] *= tensor_ptr[el];
				// 	for (int mode=dim-1; mode>=0; --mode) {
				// 		if (mode == vector_up) {
				// 			continue;
				// 		} else {
				// 		// printf("%lf = %lf times %lf, \n",
				// 		// vector_pointers[vector_up][out_index]*vector_pointers[mode][in_index],
				// 		// vector_pointers[mode][in_index],
				// 		// vector_pointers[vector_up][out_index]);
				// 		// vector_pointers[vector_up][out_index] *= vector_pointers[mode][in_index];
				// 		int in_index = (el / mul_results[mode]) % mode_size;
				// 		temp_result *= vector_pointers[mode][in_index];
				// 		// printf("mode[%d], in_index = %d\n", mode, in_index);
				// 		}
				// 	}
				// 	int out_index = (el / mul_results[vector_up]) % mode_size;
				// 	vector_pointers[vector_up][out_index] += temp_result;
				// }

					// if (++iterations == dim-1) {
					// 	break;
				// if (dim == 2) {
				// } else {

					// if (vector_up != dim-1) {

					// 	for (int el=0; el<block_size; ++el) {
					// 		// vector_pointers[vector_up][out_index] = tensor_ptr[el]*vector_pointers[dim-1][el % 3];
					// 		vector_pointers[vector_up][out_index] = tensor_ptr[el];
					// 		for (int mode=dim-1; mode>=0; --mode) {
					// 			temp_el = el / 3; // integer division (!)
					// 			if (mode == vector_up) {
					// 				out_index = temp_el % 3;
					// 				continue;
					// 			} else {
					// 				vector_pointers[vector_up][out_index] *= vector_pointers[mode][temp_el % 3];
					// 			}
					// 		}
					// 		// if (++iterations == dim-1) {
					// 		// 	break;
					// 		// }
					// 	}

					// } else {

					// 	for (int el=0; el<block_size; ++el) {
					// 		vector_pointers[vector_up][out_index] = tensor_ptr[el];
					// 		// vector_pointers[vector_up][out_index] = tensor_ptr[el]*vector_pointers[dim-2][(el/3) % 3];
					// 		for (int mode=dim-1; mode>=0; --mode) {
					// 			temp_el = el / 3; // integer division (!)
					// 			if (mode == vector_up) {
					// 				out_index = temp_el;
					// 				continue;
					// 			} else {
					// 				vector_pointers[vector_up][out_index % 3] *= vector_pointers[mode][temp_el % 3];
					// 			}
					// 		}
					// 	}
					// }
				// }
				// for (int el=0; el<result_size; ++el) {
				// 	vector_pointers[vector_up][el] = 1;
				// 	for (int mode=dim-1; mode>=0; --mode) {
				// 		vector_pointers[vector_up][?] *= vector_pointers[mode][?];
				// 	}
				// }

				#ifdef DEBUG_ENV
					// printf("output (4 el): ");
					print_to_console(vector_pointers[vector_up], 4);
				#endif

				if (++block_el == blocks) {
					break;
				}

				#ifndef SINGLEBLOCK
					tensor_ptr += block_size;
				#endif

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

				// Hard core stuff - recalculate each vector(!)
				// But we could possibly have a heuristic
				// For example only one is incremented(!)
				// The rest may arbitrarily change
				for (int d=0; d<dim; ++d) {
					vector_pointers[d] = vector_array[d]->data + (block_counter[d] * tensor->block_layout[d]);
				}

			}

			// vector_array[vector_up] += sum;

			#ifdef NORMALIZE
				(void) normalize(vector_array[vector_up], global_vector_size);
			#endif

			#ifdef SINGLEVECTORUP
				break;
			#endif

		}
	}

	#ifdef PRINT_AUX
		printf("Total executions: %zu, block_size: %zu, ", executions, block_size);
	#endif
	// printf("total_memory touched is %zu\n", total_memory);

	free(block_counter);
	free(block_counter_threshold);

}
















void
pmBlockSingleMvs(const struct tensor_storage * restrict tensor, struct lin_storage * restrict * restrict vector_array,
	struct lin_storage * restrict result_1, struct lin_storage * restrict result_2, const int iters) {
		
	#ifdef DEBUG_ENV
		printf("tvMortonSingleMvs\n");
	#endif

	const int dim = tensor->dim;

	size_t global_vector_size = tensor->layout[0];
	size_t blocks = 1;
	size_t mul_left[dim];
	size_t mul_mode[dim];
	mul_mode[dim-1] = 1;
	mul_left[dim-1] = 1;
	for (int d=dim-1; d>=0; --d) {
		size_t temp = (tensor->layout[d] + tensor->block_layout[d] - 1) / tensor->block_layout[d];
		if (d == dim-1) {
			mul_mode[d] = mul_left[d];
			mul_left[d] = temp;
		} else {
			mul_mode[d] = mul_left[d+1];
			mul_left[d] = mul_left[d+1] * temp;
		}
		blocks *= temp;
	}

	size_t block_size = 1;
	for (int d=0; d<dim; ++d) {
		block_size *= tensor->block_layout[d];
	}

	////////////// THIS CODE IS FROM THE TvSingle function
	size_t right_size[dim];
	size_t left_size[dim];
	left_size[0] = 1;
	right_size[dim-1] = 1;
	for (int d=1; d<dim; ++d) {
		left_size[d] = left_size[d-1] * tensor->block_layout[d];
	}	
	for (int d=dim-2; d>=0; --d) {
		right_size[d] = right_size[d+1] * tensor->block_layout[d];
	}	

	const MKL_INT mode_size = tensor->block_layout[0]; // vector_up is the "mode"?
	const MKL_INT incx = 1; 
	const MKL_INT incy = 1; 

	const MKL_INT result_size = block_size / mode_size;

	double alpha = 1;

	for (int j=0; j<iters; ++j) {

		#ifdef DEBUG_ENV
			printf("\niteration %d\n", j);
		#endif

		// vector up is the one being produced
		for (int vector_up=VECTOR_START; vector_up<dim; ++vector_up) {
			reset_array(vector_array[vector_up]->data, vector_array[vector_up]->size, 0.0);

			#ifdef DEBUG_ENV
			printf("vector_up=%d\n", vector_up);
			#endif

			size_t el = 0;

			double * restrict vector_pointers[dim]; // we have a set of pointers pointing to those vector memoriess
			const double * restrict tensor_ptr = tensor->lin.data; // reset the pointer to beginning
			for (int i=0; i<dim; ++i) {
				vector_pointers[i] = vector_array[i]->data;
			}

			// memset(block_counter, 0, dim*sizeof(size_t));

			while (1) {

				int iterations = 0;

				const double * restrict input = tensor_ptr;
					  double * restrict output = result_1->data;
				const double * restrict vector;

				// reset_array(output, result_size, 0.0);
				int right_mode = dim-1; // literally a mode to countdown from
				int left_mode = 0;
				double beta = 0;
				for (int iterations = 0; iterations < dim-1; ++iterations) {

					if (iterations == dim-2) {
						beta = 1;
						output = vector_pointers[vector_up];
					}

					if (right_mode > vector_up) {
						// printf("right mode: vector taken is %zu\n", right_mode);
						vector = vector_pointers[right_mode--];
						// right_kernel[iterations](vector, input, output, NULL, NULL, NULL);
						cblas_dgemv(
							CblasRowMajor, 
							CblasNoTrans, 
							right_size[iterations], mode_size, 
							alpha, 
							input, mode_size,
							vector, incx,
							beta, 
							output, incy);

					} else {
						// printf("left mode: vector taken is %d\n", left_mode);
						vector = vector_pointers[left_mode++];
						// left_kernel[iterations](input, vector, output, NULL, NULL, NULL);
						cblas_dgemv(
							CblasColMajor,
							CblasNoTrans, 
							right_size[iterations], mode_size,
							alpha, 
							input, right_size[iterations],
							vector, incx,
							beta, 
							output, incy);
					}

					#ifdef DEBUG_ENV
						printf("output (4 el): ");
						print_to_console(output, 4);
					#endif
					if (iterations % 2 == 0) {
						output = result_2->data;
						input = result_1->data;
					} else {
						output = result_1->data;
						input = result_2->data;
					}
					// reset_array(output, right_size[iterations], 0.0);
				}

				#ifdef SINGLEBLOCK
					break;
				#endif

				if (++el == blocks) {
					break;
				}

				#ifndef SINGLEBLOCK
					tensor_ptr += block_size;
				#endif
				for (int d=0; d<dim; ++d) {
					if (el % mul_left[d] == 0) {
						#ifdef DEBUG_ENV
							printf("Resetting mode %d vector (to beginning of its memory)\n", d);
							#endif
							vector_pointers[d] = vector_array[d]->data;
					} else if (el % mul_mode[d] == 0) {
							vector_pointers[d] += mode_size;		
							#ifdef DEBUG_ENV
							printf("Move along mode %d vector by %zu elements (inc its memory)\n", d, mode_size);
							#endif	
					}	
				}

			}

			#ifdef NORMALIZE
				(void) normalize(vector_array[vector_up], global_vector_size);
			#endif

			#ifdef SINGLEVECTORUP
				break;
			#endif

		}
	}

	// free(block_counter);
	// free(block_counter_threshold);

}




void
pmBlock(const struct tensor_storage * restrict tensor, struct lin_storage * restrict * restrict vector_array,
	struct lin_storage * restrict result_1, struct lin_storage * restrict result_2, const int iters) {
		
	#ifdef DEBUG_ENV
		printf("tvMortonSingleMvs\n");
	#endif

	const int dim = tensor->dim;

	size_t global_vector_size = tensor->layout[0];
	size_t blocks = 1;
	size_t mul_left[dim];
	size_t mul_mode[dim];
	mul_mode[dim-1] = 1;
	mul_left[dim-1] = 1;
	for (int d=dim-1; d>=0; --d) {
		size_t temp = (tensor->layout[d] + tensor->block_layout[d] - 1) / tensor->block_layout[d];
		if (d == dim-1) {
			mul_mode[d] = mul_left[d];
			mul_left[d] = temp;
		} else {
			mul_mode[d] = mul_left[d+1];
			mul_left[d] = mul_left[d+1] * temp;
		}
		blocks *= temp;
	}

	size_t block_size = 1;
	for (int d=0; d<dim; ++d) {
		block_size *= tensor->block_layout[d];
	}

	////////////// THIS CODE IS FROM THE TvSingle function
	size_t right_size[dim];
	size_t left_size[dim];
	left_size[0] = 1;
	right_size[dim-1] = 1;
	for (int d=1; d<dim; ++d) {
		left_size[d] = left_size[d-1] * tensor->block_layout[d];
	}	
	for (int d=dim-2; d>=0; --d) {
		right_size[d] = right_size[d+1] * tensor->block_layout[d];
	}	

	const MKL_INT mode_size = tensor->block_layout[0]; // vector_up is the "mode"?
	const MKL_INT incx = 1; 
	const MKL_INT incy = 1; 

	const MKL_INT result_size = block_size / mode_size;

	double alpha = 1;

	for (int j=0; j<iters; ++j) {

		#ifdef DEBUG_ENV
			printf("\niteration %d\n", j);
		#endif

		// vector up is the one being produced
		for (int vector_up=VECTOR_START; vector_up<dim; ++vector_up) {
			reset_array(vector_array[vector_up]->data, vector_array[vector_up]->size, 0.0);

			#ifdef DEBUG_ENV
			printf("vector_up=%d\n", vector_up);
			#endif

			size_t el = 0;

			double * restrict vector_pointers[dim]; // we have a set of pointers pointing to those vector memoriess
			const double * restrict tensor_ptr = tensor->lin.data; // reset the pointer to beginning
			for (int i=0; i<dim; ++i) {
				vector_pointers[i] = vector_array[i]->data;
			}

			// memset(block_counter, 0, dim*sizeof(size_t));

			while (1) {

				int iterations = 0;
				size_t n3 = result_size;
				int temp_dim = dim;
				const double * restrict input = tensor_ptr;
					  double * restrict output = result_1->data;
				const double * restrict vector;
				double beta = 0;
				size_t divisor = 1;
				for (int mode=dim-1; mode>=0; --mode) {
					if (mode == vector_up) continue;

					vector = vector_pointers[mode];
					if (iterations == dim-2) {
						beta = 1;
						output = vector_pointers[vector_up];
					}

					size_t fixed_right_size = right_size[mode] / divisor;
					if (mode != --temp_dim) {
						#ifdef DEBUG_ENV
							printf("left_size: %zu, right_size: %zu\n", left_size[mode], fixed_right_size);
							printf("mode multiplied with is %d, ", mode);
							printf("tensor_ptr: ");
							print_to_console(input, block_size);
							// printf("Mul this block with vector[%d]\n", mode);
							printf("vector: ");
							print_to_console(vector, mode_size);
							printf("output (JUST BEFORE): ");
							print_to_console(output, mode_size);
						#endif
						for (size_t i=0; i<left_size[mode]; ++i) {
							const double * restrict next = input + i*mode_size*fixed_right_size;
							double * restrict next_result = output + i*fixed_right_size;
							cblas_dgemv(
								CblasColMajor, 
								CblasNoTrans, 
								fixed_right_size, mode_size,
								alpha, 
								next, fixed_right_size,
								vector, incx,
								beta, 
								next_result, incy);
						}

					} else {
						#ifdef DEBUG_ENV
							printf("result_size=%d, mode_size=%d\n", n3, mode_size);
							printf("mode multiplied with is %d, ", mode);
							printf("tensor_ptr: ");
							print_to_console(input, block_size);
							// printf("Mul this block with vector[%d]\n", mode);
							printf("vector: ");
							print_to_console(vector, mode_size);
							printf("output (JUST BEFORE): ");
							print_to_console(output, mode_size);
						#endif
						cblas_dgemv(
							CblasRowMajor, 
							CblasNoTrans, 
							n3, mode_size, 
							alpha, 
							input, mode_size,
							vector, incx,
							beta, 
							output, incy);
					}

					#ifdef DEBUG_ENV
						printf("output (4 el): ");
						print_to_console(output, 4);
						if (mode != dim-1) {
							// print_to_console(output, fixed_right_size*left_size[mode]);
						} else {
							// print_to_console(output, n2);
						}
					#endif

					if (++iterations == dim-1) {
						break;
					}

					if (iterations % 2 == 0) {
						output = result_1->data;
						// reset_array(result_1->data, result_size, 0);
						input = result_2->data;
					} else {
						output = result_2->data;
						// reset_array(result_2->data, result_size, 0);
						input = result_1->data;
					}

					divisor *= mode_size;
					n3 = result_size / divisor;
				}

				// reset_array(output, result_size, 0.0);
				// int right_mode = dim-1; // literally a mode to countdown from
				// int left_mode = 0;
				// double beta = 0;
				// for (int iterations = 0; iterations < dim-1; ++iterations) {

				// 	if (iterations == dim-2) {
				// 		beta = 1;
				// 		output = vector_pointers[vector_up];
				// 	}

				// 	if (right_mode > vector_up) {
				// 		// printf("right mode: vector taken is %zu\n", right_mode);
				// 		vector = vector_pointers[right_mode--];
				// 		// right_kernel[iterations](vector, input, output, NULL, NULL, NULL);
				// 		cblas_dgemv(
				// 			CblasRowMajor, 
				// 			CblasNoTrans, 
				// 			right_size[iterations], mode_size, 
				// 			alpha, 
				// 			input, mode_size,
				// 			vector, incx,
				// 			beta, 
				// 			output, incy);

				// 	} else {
				// 		// printf("left mode: vector taken is %d\n", left_mode);
				// 		vector = vector_pointers[left_mode++];
				// 		// left_kernel[iterations](input, vector, output, NULL, NULL, NULL);
				// 		cblas_dgemv(
				// 			CblasColMajor,
				// 			CblasNoTrans, 
				// 			right_size[iterations], mode_size,
				// 			alpha, 
				// 			input, right_size[iterations],
				// 			vector, incx,
				// 			beta, 
				// 			output, incy);
				// 	}

				// 	#ifdef DEBUG_ENV
				// 		printf("output (4 el): ");
				// 		print_to_console(output, 4);
				// 	#endif
				// 	if (iterations % 2 == 0) {
				// 		output = result_2->data;
				// 		input = result_1->data;
				// 	} else {
				// 		output = result_1->data;
				// 		input = result_2->data;
				// 	}
				// 	// reset_array(output, right_size[iterations], 0.0);
				// }

				#ifdef SINGLEBLOCK
					break;
				#endif

				if (++el == blocks) {
					break;
				}

				#ifndef SINGLEBLOCK
					tensor_ptr += block_size;
				#endif
				for (int d=0; d<dim; ++d) {
					if (el % mul_left[d] == 0) {
						#ifdef DEBUG_ENV
							printf("Resetting mode %d vector (to beginning of its memory)\n", d);
							#endif
							vector_pointers[d] = vector_array[d]->data;
					} else if (el % mul_mode[d] == 0) {
							vector_pointers[d] += mode_size;		
							#ifdef DEBUG_ENV
							printf("Move along mode %d vector by %zu elements (inc its memory)\n", d, mode_size);
							#endif	
					}	
				}

			}

			#ifdef NORMALIZE
				(void) normalize(vector_array[vector_up], global_vector_size);
			#endif

			#ifdef SINGLEVECTORUP
				break;
			#endif

		}
	}

	// free(block_counter);
	// free(block_counter_threshold);

}


void
pmBlockLibx(const struct tensor_storage * restrict tensor, struct lin_storage * restrict * restrict vector_array,
	struct lin_storage * restrict result_1, struct lin_storage * restrict result_2, const int iters) {
		
	#ifdef DEBUG_ENV
		printf("tvMorton\n");
	#endif

	const int dim = tensor->dim;

	size_t global_vector_size = tensor->layout[0];
	size_t blocks = 1;
	size_t mul_left[dim];
	size_t mul_mode[dim];
	mul_mode[dim-1] = 1;
	mul_left[dim-1] = 1;
	for (int d=dim-1; d>=0; --d) {
		size_t temp = (tensor->layout[d] + tensor->block_layout[d] - 1) / tensor->block_layout[d];
		if (d == dim-1) {
			mul_mode[d] = mul_left[d];
			mul_left[d] = temp;
		} else {
			mul_mode[d] = mul_left[d+1];
			mul_left[d] = mul_left[d+1] * temp;
		}
		blocks *= temp;
	}

	size_t block_size = 1;
	for (int d=0; d<dim; ++d) {
		block_size *= tensor->block_layout[d];
	}

	// Morton stuff (2)
	// const size_t morton_block_levels = log2(max_block)+1; // round-up (take front size_teger and add 1)
	// size_t * const morton_block_indices = calloc(morton_block_levels, sizeof(size_t));
	size_t mask;
	size_t level;
	size_t inc_game;
	size_t offset;
	// int block_diff;
	// double block_diff_log;

	////////////// THIS CODE IS FROM THE TvSingle function
	size_t right_size[dim];
	size_t left_size[dim];
	left_size[0] = 1;
	right_size[dim-1] = 1;
	for (int d=1; d<dim; ++d) {
		left_size[d] = left_size[d-1] * tensor->block_layout[d];
	}	
	for (int d=dim-2; d>=0; --d) {
		right_size[d] = right_size[d+1] * tensor->block_layout[d];
	}	

	#ifdef DEBUG_ENV
		printf("left sizes vector: ");
		print_to_console_sizet(left_size, dim);
		printf("right sizes vector: ");
		print_to_console_sizet(right_size, dim);
	#endif

	// PROBLEM NON SQARE (!!!!)
	const MKL_INT mode_size = tensor->block_layout[0]; // vector_up is the "mode"?
	const MKL_INT incx = 1; 
	const MKL_INT incy = 1; 

	// const MKL_INT n2 = tensor->lin.size / mode_size;
	const MKL_INT result_size = block_size / mode_size;

	double alpha = 1;
	const double betta = 0.0;

	////////////// THIS CODE IS FROM THE TvSingle function

	// WAIT (betta = 0) is in most cases because we dont want to add up to the result buffer but to the original vector(!)

	// mode > vector_up -> right_size[dim-1]
	// mode < vector_up -> right_size[dim-2]
	libxsmm_dmmfunction kernel_left_high = libxsmm_dmmdispatch(right_size[dim-1], 1, mode_size, NULL, NULL, NULL, NULL, &betta, NULL, NULL); // &prefetch);
	libxsmm_dmmfunction kernel_left_low = libxsmm_dmmdispatch(right_size[dim-2], 1, mode_size, NULL, NULL, NULL, NULL, &betta, NULL, NULL); // &prefetch);
	libxsmm_dmmfunction kernel_left_low_beta = libxsmm_dmmdispatch(right_size[dim-2], 1, mode_size, NULL, NULL, NULL, NULL, NULL, NULL, NULL); // &prefetch);

	libxsmm_dmmfunction kernel_left;

	// // that's for the fixed_right_size
	// // result_size -> that's for 4D, we have 3D sizes of results -> equal right_size 0,1,2
	// actually wrong,we onlydoright handmultiplicationonce (!) for mode = dim-1 which takes place only a fewtimes
	// libxsmm_dmmfunction kernel_right = libxsmm_dmmdispatch(1, result_size, mode_size, NULL, NULL, NULL, NULL, NULL, NULL, NULL); // &prefetch);

	libxsmm_dmmfunction kernel_right[dim-1];
	for (int i=1; i < dim-1; ++i) {
		kernel_right[i] = libxsmm_dmmdispatch(1, left_size[i+1], mode_size, NULL, NULL, NULL, NULL, &betta, NULL, NULL); // &prefetch);
	}
	kernel_right[0] = libxsmm_dmmdispatch(1, left_size[1], mode_size, NULL, NULL, NULL, NULL, NULL, NULL, NULL); // &prefetch);

	// printf("Important -> sizesof right size\n");
	// print_to_console_sizet(right_size, dim);

	for (int j=0; j<iters; ++j) {

		#ifdef DEBUG_ENV
			printf("\niteration %d\n", j);
		#endif

		// vector up is the one being produced
		for (int vector_up=VECTOR_START; vector_up<dim; ++vector_up) {

			reset_array(vector_array[vector_up]->data, vector_array[vector_up]->size, 0.0);

			#ifdef DEBUG_ENV
				printf("vector_up=%d\n", vector_up);
			#endif
			// printf("vector_up=%d\n", vector_up);

			size_t el = 0;

			// size_t vector_size = tensor->block_layout[vector_up]; // vector_size is the size of the block in that dim
			// to simply - improve this (size of each vector depends on the mode)

			double * restrict vector_pointers[dim]; // we have a set of pointers pointing to those vector memoriess
			const double * restrict tensor_ptr = tensor->lin.data; // reset the pointer to beginning
			for (int i=0; i<dim; ++i) {
				vector_pointers[i] = vector_array[i]->data;
			}

			// printf("BEFRpore\n");
			// print_to_console_sizet(block_counter, dim);
			// memset(block_counter, 0, dim*sizeof(size_t));
			// print_to_console_sizet(block_counter, dim);

			while (1) {

				#ifdef DEBUG_ENV
					printf("block %zu, ", el);
				#endif
				// // process that block (with each vector except vector_up) producing vector_up
				// tvLoopedSingle(
				// 	tensor,
				// 	tensor_ptr, // current block pointer
				// 	vector_pointers, // it needs all "sub" vectors for this block (TODO)
				// 	result_1, // result vectors should be always the same (just temp)
				// 	result_2,
				// 	vector_up, // for now 0 (but should be the vector up!)
				// 	block_size
				// );

				// size_t divisor = 1;

				#ifdef DEBUG_ENV
					printf("Current tensor is:\n");
					print_to_console(tensor_ptr, 4); //block_size);
					printf("For example first element is: %f\n", tensor_ptr[0]);
					// int dim = tensor->dim;
				#endif

				int iterations = 0;

				const double * restrict input = tensor_ptr;
					  double * restrict output = result_1->data;
				const double * restrict vector;

				// reset_array(output, result_size, 0.0);
				// size_t n3 = result_size;
				size_t chosen_right_size = right_size[dim-1];
				kernel_left = kernel_left_high;

				int temp_dim = dim;
							
				for (int mode=dim-1; mode>=0; --mode) {

					if (mode == vector_up) {
						chosen_right_size = right_size[dim-2];
						kernel_left = kernel_left_low;
						// printf("this had happened\n");
						continue;
					}
					// printf("mode multiplied with is %d, ", mode);

					vector = vector_pointers[mode];
					if (iterations == dim-2) {
						// beta = 1;
						kernel_left = kernel_left_low_beta;
						// printf("iterations = %zu, ", dim-2);
						output = vector_pointers[vector_up];
					}

					#ifdef DEBUG_ENV
						if (iterations == dim-2) {
							// printf("Result before its produced:\n");
							// print_to_console(output, tensor->block_layout[vector_up]);
						} else {
							// printf("Temporary result_1 storage (intermediate storage):");
							// print_to_console(output, tensor->block_layout[vector_up]);
						}
						// printf("Mul this block with vector[%d]\n", mode);
						// printf("Vector used to multiply this block is: \n");
						// print_to_console(vector, mode_size);
					#endif

					// size_t fixed_right_size = right_size[mode] / divisor;
					// printf("n3=%zu, mode is %zu, right_size mode is%zu, vecotr_up is %zu and fixedright size is %zu", n3, mode,right_size[mode], vector_up,  fixed_right_size);

					if (mode != --temp_dim) {
						#ifdef DEBUG_ENV
						printf("left_size: %zu, right_size: %zu (times mode size %zu),\n", left_size[mode], chosen_right_size, mode_size);
						#endif
						// printf("left_size: %zu, right_size: %zu (times mode size %zu)\n", left_size[mode], chosen_right_size, mode_size);

						for (size_t i=0; i<left_size[mode]; ++i) {
							// const double * restrict next = input + i*mode_size*fixed_right_size;
							// double * restrict next_result = output + i*fixed_right_size;
							// cblas_dgemv(
							// 	CblasColMajor, 
							// 	CblasNoTrans, 
							// 	fixed_right_size, mode_size,
							// 	alpha, 
							// 	next, fixed_right_size,
							// 	vector, incx,
							// 	beta, 
							// 	next_result, incy);
							// if (mode > vector_up) {
							// 	const double * restrict next = input + i*mode_size*chosen_right_size;
							// 	double * restrict next_result = output + i*chosen_right_size;
							// 	kernel_left(next, vector, next_result, NULL, NULL, NULL);
							// } else {
								const double * restrict next = input + i*mode_size*chosen_right_size;
								double * restrict next_result = output + i*chosen_right_size;
								kernel_left(next, vector, next_result);//, NULL, NULL, NULL);
								// printf("kernel_left\n");
							// }

						}
						// printf(", the size of result = %zu\n", left_size[mode]*fixed_right_size);
					} else {
						#ifdef DEBUG_ENV
							printf("result_size: %d, mode_Size: %d\n", left_size[mode], mode_size);
							#endif
							// printf("result_size: %d, mode_Size: %d\n", left_size[mode], mode_size);
						// cblas_dgemv(
						// 	CblasRowMajor, 
						// 	CblasNoTrans, 
						// 	n3, mode_size, 
						// 	alpha, 
						// 	input, mode_size,
						// 	vector, incx,
						// 	beta, 
						// 	output, incy);
							// printf("mode-1=%zu\n", mode-1);
						kernel_right[mode-1](vector, input, output);//, NULL, NULL, NULL);
						// printf(", the size of result = %zu\n", n3);

					}

					#ifdef DEBUG_ENV
						printf("output (4 el): ");
						print_to_console(output, 4);
						if (mode != dim-1) {
							// print_to_console(output, fixed_right_size*left_size[mode]);
						} else {
							// print_to_console(output, n2);
						}
					#endif

					if (++iterations == dim-1) {
						// printf("mode = %zu, ", mode);
						// printf("iterations = %zu, so we break\n", dim-1);
						break;
					}
					
					// SWITCH BETWEEN SIDES:::: SIMPLY COMMENT ? UNCOMMENT THE BELOW
					// input = result_1->data;
					if (iterations % 2 == 0) {
						output = result_1->data;
						// reset_array(result_1->data, result_size, 0);
						input = result_2->data;
					} else {
						output = result_2->data;
						// reset_array(result_2->data, result_size, 0);
						input = result_1->data;
					}

					// divisor *= mode_size;
					// n3 = result_size / divisor;
					// reset_array(output, right_size[iterations], 0.0);

				}

				if (++el == blocks) {
					break;
				}

				#ifndef SINGLEBLOCK
					tensor_ptr += block_size;
				#endif
				for (int d=0; d<dim; ++d) {
					if (el % mul_left[d] == 0) {
						#ifdef DEBUG_ENV
							printf("Resetting mode %d vector (to beginning of its memory)\n", d);
							#endif
							vector_pointers[d] = vector_array[d]->data;
					} else if (el % mul_mode[d] == 0) {
							vector_pointers[d] += mode_size;		
							#ifdef DEBUG_ENV
							printf("Move along mode %d vector by %zu elements (inc its memory)\n", d, mode_size);
							#endif	
					}	
				}

			}

			#ifdef NORMALIZE
				(void) normalize(vector_array[vector_up], global_vector_size);
			#endif

			#ifdef SINGLEVECTORUP
				break;
			#endif

		}
	}

	// free(block_counter);
	// free(block_counter_threshold);

}