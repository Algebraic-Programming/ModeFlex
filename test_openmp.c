#define _GNU_SOURCE // HT requirements; sched_getcpu(3) is glibc-specific (see the man page)
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
#include <omp.h>
#include <mkl.h>
#include <sched.h>
#ifdef LIKWID_PERFMON
#include <likwid.h>
#else
#define LIKWID_MARKER_INIT
#define LIKWID_MARKER_THREADINIT
#define LIKWID_MARKER_SWITCH
#define LIKWID_MARKER_REGISTER(regionTag)
#define LIKWID_MARKER_START(regionTag)
#define LIKWID_MARKER_STOP(regionTag)
#define LIKWID_MARKER_CLOSE
#define LIKWID_MARKER_GET(regionTag, nevents, events, time, count)
#endif

// #define COMPLETEMORTON
#define RANDOMIZE

#define HEADERMULTICORE "timestamp,elapsed,response,success,dim,mode,n,block_n,algo_id,std,mworst,mbest,perf,mbuf,mkl,omp,git_hash,b_array,p_array,n_array,operation\n"

#define MAX_1 		134217728
#define MAX_5 		671088640
#define MAX_10 		1342177280
#define MAX_25 		3355443200
#define MAX_37pol 	5033164800
#define MAX_50 		6710886400
#define MAX_75 		10066329600
#define MAX_100 	13421772800
#define MAX_100_x 	13799808000
#define MAX_150 	20132659200
#define MAX_200 	26843545600
#define MAX_200_x 	26873856000 
#define MAX_300 	40265318400
#define MAX_400 	53687091200

#define MIN 1210000
// #define MIN 6553600 // 50MB
// #define MAX 2684354560 // 20GB
// #define MAX 26843545
// #define MAX 5368709120 // 40GB
// #define MAX 8053063680 // 60GB

#if (TEST_ENV == 1)
	#define TENSIZE MAX_1
	#define SINGLESIZE 10
#else
	#define TENSIZE MAX_100_x
	#define SINGLESIZE 12
#endif

// size_t
// set_p_array(struct array p_array, const size_t dim) {

// 	const int p_size = 1;

// 	#if (TEST_ENV == 1)
// 		printf("INFO: TEST_ENV=1: p_array set to {p_size, p_size, 1...}\n");
// 		for(int i=0;i<dim;++i) p_array.a[i] = 2;
// 		p_array.a[0] = p_size;
// 	#else
// 	switch (p_size) {
// 		case 8:
// 			switch (dim) {
// 				case 2:
// 					p_array = (struct array){{p_size, 8}};
// 					break;
// 				case 3:
// 					p_array = (struct array){{p_size, 2, 4}};
// 					break;
// 				case 4:
// 					p_array = (struct array){{p_size, 2, 1, 4}};
// 					break;
// 				case 5:
// 					// p_array = (struct array){{p_size, 1, 2, 2, 2}}; // loosing mode-obliousness (mode=1?)
// 					p_array = (struct array){{p_size, 4, 1, 3, 2}}; // 16 steps instead of 8
// 					break;
// 				default:
// 					printf("ERROR: We cannot handle striping for more than dim=5 (unless we adopt p^{d-1}");
// 					exit(-1);
// 					break;
// 			}
// 			break;
// 		case 4:
// 			switch (dim) {
// 				case 2:
// 					p_array = (struct array){{p_size, 4}};
// 					break;
// 				case 3:
// 					p_array = (struct array){{p_size, 2, 2}};
// 					break;
// 				case 4:
// 					p_array = (struct array){{p_size, 2, 2, 2}};
// 					break;
// 				case 5:
// 					// p_array = (struct array){{p_size, 1, 2, 2, 2}}; // loosing mode-obliousness (mode=1?)
// 					p_array = (struct array){{p_size, 2, 2, 2, 2}}; // 16 steps instead of 8
// 					break;
// 				case 6:
// 					p_array = (struct array){{p_size, 3, 2, 2, 2, 2}};
// 					break;
// 				default:
// 					printf("ERROR: We cannot handle striping for more than dim=5 (unless we adopt p^{d-1}");
// 					exit(-1);
// 					break;
// 			}
// 			break;
// 		case 12:
// 			switch (dim) {
// 				case 2:
// 					p_array = (struct array){{p_size, 14}};
// 					break;
// 				case 3:
// 					p_array = (struct array){{p_size, 4, 4}};
// 					break;
// 				case 4:
// 					p_array = (struct array){{p_size, 2, 2, 3}};
// 					break;
// 				case 5:
// 					p_array = (struct array){{p_size, 3, 2, 2, 1}};
// 					break;
// 				case 6:
// 					p_array = (struct array){{p_size, 2, 2, 2, 2, 2}};
// 					break;
// 				default:
// 					printf("ERROR: We cannot handle striping for more than dim=5 (unless we adopt p^{d-1}");
// 					exit(-1);
// 					break;
// 			}
// 			break;
// 		case 15:
// 			switch (dim) {
// 				case 2: 
// 					p_array = (struct array){{p_size, 15}};
// 					break;
// 				case 3:
// 					p_array = (struct array){{p_size, 5, 3}};
// 					break;
// 				case 4:
// 					p_array = (struct array){{p_size, 2, 3, 3}};
// 					break;
// 				case 5:
// 					p_array = (struct array){{p_size, 2, 2, 2, 2}};
// 					break;
// 				case 6:
// 					p_array = (struct array){{p_size, 2, 2, 2, 2, 2}};
// 					break;
// 			}
// 			break;
// 		case 16:
// 			switch (dim) {
// 				case 2: 
// 					p_array = (struct array){{p_size, 16}};
// 					break;
// 				case 3:
// 					p_array = (struct array){{p_size, 4, 4}};
// 					break;
// 				case 4:
// 					p_array = (struct array){{p_size, 4, 2, 2}};
// 					break;
// 				case 5:
// 					p_array = (struct array){{p_size, 2, 2, 2, 2}};
// 					break;
// 				case 6:
// 					p_array = (struct array){{p_size, 2, 2, 2, 2, 2}}; // Going slightly above, should be fine?
// 					break;
// 			}
// 			break;
// 			break;
// 		case 30:
// 			switch (dim) {
// 				case 2: 
// 					p_array = (struct array){{p_size, 30}};
// 					break;
// 				case 3:
// 					p_array = (struct array){{p_size, 5, 3}};
// 					break;
// 				case 4:
// 					p_array = (struct array){{p_size, 5, 3, 2}};
// 					break;
// 				case 5:
// 					p_array = (struct array){{p_size, 4, 2, 2, 2}};
// 					break;
// 				case 6:
// 					p_array = (struct array){{p_size, 2, 2, 2, 2, 2}};
// 					break;
// 				default:
// 					break;
// 			}
// 			break;
// 		case 28:
// 			switch (dim) {
// 				case 2: 
// 					p_array = (struct array){{p_size, 28}};
// 					break;
// 				case 3:
// 					p_array = (struct array){{p_size, 7, 4}};
// 					break;
// 				case 4:
// 					p_array = (struct array){{p_size, 7, 2, 2}};
// 					break;
// 				case 5:
// 					p_array = (struct array){{p_size, 4, 2, 2, 2}}; // 32 instead of 28
// 					break;
// 				case 6:
// 					p_array = (struct array){{p_size, 2, 2, 2, 2, 2}}; // 32 instead of 28
// 					break;
// 				default:
// 					break;
// 			}
// 			break;
// 		default:
// 			// printf("Hitting default 1\n");
// 			switch (dim) {
// 				default:
// 					printf("INFO: No setting for this number of processors.\n");
// 					// Fit in the whole array with p_size at each dimenion (!)
// 					for (int i=0; i<(int)dim; ++i) p_array.a[i] = 1;
// 					p_array.a[0] = p_size;
// 					p_array.a[1] = p_size;
// 					break;
// 			}
// 			break;
// 	}
// 	#endif

int test_openmp(int argc, char ** argv) {
    // Init Marker API in serial region once in the beginning
    LIKWID_MARKER_INIT;
    #pragma omp parallel
    {
        // Each thread must add itself to the Marker API, therefore must be
        // in parallel region
        LIKWID_MARKER_THREADINIT;
        // Optional. Register region name
        // LIKWID_MARKER_REGISTER("example");
    }
    
	// numa_set_localalloc(); is called just after main() in tests_all.c
	
	int p_size;
    #pragma omp parallel
    #pragma omp single
    	p_size = omp_get_num_threads();

	int pin_array[p_size];
	printf("Pinning: ");
	#pragma omp parallel
	{
        int thread_num = omp_get_thread_num();
		#pragma omp for ordered schedule(static,1)
	    for (int t=0; t<p_size; ++t)
	    {
	        assert( t==thread_num );
	        #pragma omp ordered
	        {
		        int cpu_num = sched_getcpu();
		        pin_array[t] = cpu_num;
	        	printf("%d ", cpu_num);
	        }
	    }
	}
	printf("\nTotal number of threads is %d\n", p_size);

	// PARAMTERIZING THE CODE
	size_t tensize_max = (size_t) TENSIZE; // Unless it gets overwritten
	int algo_max_par = 2;
	int algo_max_par_unfold = 1;

	// Sequential algorithms only on the single socket run (!)
	#ifdef SINGLESOCKET
		int algo_max_seq = 3;
	#else
		int algo_max_seq = 0;
	#endif
	int dim_min = 2;
	int dim_max = 4;
	switch (argc) {
		case 4:
			sscanf (*(argv+argc--), "%zu", &tensize_max);
		case 3:
			sscanf (*(argv+argc--), "%d", &algo_max_par);	
		case 2:
			sscanf (*(argv+argc--), "%d", &dim_max);
			sscanf (*(argv+argc--), "%d", &dim_min);
			break;
	}
	// #if (TEST_ENV == 1)
	// 	if (dim_max > 4) {
	// 		printf("WARNING: dim_max reduced to 4 dimensions; Otherwise we have impossible to meet condition!\n");
	// 		dim_max = 4;
	// 	}
	// #endif
	if (algo_max_par > 5) {
		printf("WARNING: algo_max_par reduced to 2; No more algorithms available!\n");
		algo_max_par = 5;
	}
	printf("Params: dim_min=%d, dim_max=%d, algo_max_par=%d, p_size=%d, tensize_max=%zu\n", dim_min, dim_max, algo_max_par, p_size, tensize_max);

	TVM algorithms[] = {
		tvm_power_sync_p,
		tvm_power_sync_mkl_p,
		tvMortonMulticore,
		tvm_ppower_sync_mkl_p,
		tvm_block_major_input_aligned_output_aligned_BLAS_POWERS_v3_mine,
		tvMortonMulticoreMkl,
		tvm_ppower_sync_mkl_p,
		tvm_ppower_sync_mkl,
		tvm_ppower_sync,
		tvMortonMulticoreMkl,
		block_morton_block_unfold,
		tvm_vector_major_BLAS_col_mode_multicore2,
	};

	TVM algorithms_unfold[] = {
		// tvm_vector_major_BLAS_col_mode_multicore,
		tvm_vector_major_BLAS_col_mode_multicore3
	};

	TVM algorithms_seq[] = {
		tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3,
		tvm_morton_block_major_input_aligned_output_aligned_BLAS_POWERS_3_libx,
		tvm_vector_major_BLAS_col_mode
	};

	// BENCHMARKING: Figuring out the filename
	FILE * file;
	if (TEST_ENV == 1) {
		printf("INFO: test_powermethod.c: TEST ENV\n");
	} else {
		char hostname[1024];
		gethostname(hostname, 1024);
		struct timespec time;
		clock_gettime(CLOCK_MONOTONIC, &time);
		char filename[BUFSIZE];
		printf("INFO: test_powermethod.c: BENCHMARKING ENV\n");
		snprintf(filename, BUFSIZE, "%s/%s_%.0f_dimmin_%d_dimmax_%d_nmin_%d_nmax_%d_modemin_%d_modemax_%d_blockn_%d_%s.csv", FOLDER, hostname, timespec_to_microseconds(time), dim_min, dim_max, 0, 0, 0, 0, 0, "bench");
		file = fopen(filename, "w"); // r+ if you want to update without deleteing it
		if (file == NULL) {
			perror("ERROR: Error opening file.\n");
		}
		fprint_array_int(file, pin_array, p_size);
		fprintf(file, "\n");
		fprintf(file, HEADERMULTICORE);
		fflush(file);
	}

	// GET tensor, result and vector memories (overallocating)
	// tensor_size = max
	// result_size is max when d is highest; thus, result_size = tensor_size / n, where n = round_down(tensor_size, 1.0/dim)
	// vector_size is max when d is 2; thus, vector_size = tensor_size / n, where n = round_down(tensor_size, 1.0/2.0)
	size_t tensor_size = tensize_max;
	size_t result_size;
	if (ceil(tensize_max / floor(pow(tensize_max, 1.0/(double)dim_max))) * 2 > tensize_max) { // FIX: if 2 2... we ended up with 16, 32 modes while both were alloc'ed 24
		result_size = tensize_max;
	} else {
		result_size = ceil(tensize_max / floor(pow(tensize_max, 1.0/(double)dim_max))) * 2;
	}
	// We had to add this for 5D experiments of ivy2s (!!!) otherwise perhaps redundant (!)
	// result_size = 559872000;
	result_size = 953120000;
	size_t vector_size = ceil(tensize_max / floor(pow(tensize_max, 1.0/(double)dim_min))) * 2;
	// size_t result_size = ceil(tensize_max / floor(pow(tensize_max, 1.0/(double)dim_max)));
	// size_t vector_size = ceil(tensize_max / floor(pow(tensize_max, 1.0/(double)dim_min)));
	printf("INFO: Max sizes (for allocation): tensor=%zu, result=%zu, vector=%zu\n", tensor_size, result_size, vector_size);

	// New functions allocate storage for dim_max (for layout, layout2, block_layout and layout_perm)
	printf("INFO: Allocating tensor...\n");
	struct tensor_storage * const tensor = gen_block_tensor_clean_safe(1, &tensor_size, &tensor_size, dim_max, 0); 
	struct tensor_storage * const tensor2 = gen_block_tensor_clean_safe(1, &tensor_size, &tensor_size, dim_max, 1); // Local alloc only (!)
    tensor->lin.p_size = p_size; // Important to store the correct parallelism in the tensor (!)
    tensor2->lin.p_size = p_size;
	printf("INFO: Allocating result...\n");
	struct tensor_storage * const result = gen_block_tensor_clean_safe(1, &result_size, &result_size, dim_max, 0); 
	printf("INFO: Allocating vector...\n");
	struct lin_storage * const vector = gen_vector_seeded_safe(vector_size, 0, 0); 

	// Minimize the jobs on the stack; Reuse these arrays
	size_t layout[dim_max], block_layout[dim_max], layout2[dim_max], p_sizes[dim_max];
	// Structure as a wrapper around the array
	struct array block_array, p_array; // you may use the copy assignment operator by means of compound literals
	// Assume absolute dim_max_max of 1

	for (int dim = dim_min; dim <= dim_max; ++dim) {
		printf("	DIM %d\n", dim);

		// Update the dimensions of objects
		tensor->dim = (size_t) dim;
		tensor2->dim = (size_t) dim;
		result->dim = tensor->dim-1;

		// Bottom-up strategy: first choose the block_n, then choose stripe_n, and finally choose n
		// Create one problem at a time: first, unequal stripe_n
		switch (dim) {
			case 2:
				block_array = (struct array){{44, 124, 570}}; // 10% (2.6MB): 572; 13MB: 1276; -- try also 512, 570=2.48
				break;
			case 3:
				block_array = (struct array){{12, 24, 68}}; // 10% (2.5MB): 68; 12.5MB: 116; -- try also ???, 68=2.43
				break;
			case 4:
				block_array = (struct array){{6, 10, 22}}; 	// 10% (2.7MB): 24; 10MB: 34; -- try also 24, 22=1.86, 24=2.63
				break;
			case 5:
				block_array = (struct array){{4, 6, 12}}; 	// 10% (2MB): 12; 8.5MB: 16; -- try also 12, 14=4.39, 12=2.05
				break;
			case 6:
				block_array = (struct array){{3, 4, 8}}; 	// 10% (2MB): 8; 8MB: 10;
				break;
			case 7:
				block_array = (struct array){{2, 4, 6}}; 	// 10% (2.2MB): 6; 6.6MB: 7;
				break;
			case 8:
				block_array = (struct array){{2, 3, 5}}; 	// 10% (3MB): 5; 
				break;
			case 9:
				block_array = (struct array){{2, 3, 4}}; 	// 10% (2MB): 4; 15MB: 5;
				break;
			case 10:
				block_array = (struct array){{2, 4}}; 		// 10% (?): ?; (8MB): 4
				break;
		}
		// #if (TEST_ENV == 1)
			// printf("INFO: TEST_ENV=1: simplify block test cases to be only 2 (or not?).\n");
			// block_array = (struct array){{2,2,2}};
		// #endif
		printf("INFO: BLOCK test cases: ");
		print_to_console_sizet(block_array.a, 3);
		// Instantiate an array based on this hard-coded sizes for this given dimension

		for (int bi=2; bi<=2; ++bi) {

			// printf("INFO: %d set of block layouts\n", bi);
			// BLOCK LAYOUT
			if (block_array.a[bi] == 0) {
				printf("INFO: Safety feature: We do not want to run with 0\n");
				continue; // Safety feature since we do not rely on block_array_size
			}
			// #if (TEST_ENV == 1)
			// 	printf("INFO: Test_env: continue on L3 block\n");
			// 	if (bi>1) continue;	// We test only L1 and L2-sized tensors(!)
			// #else
			// 	printf("INFO: Benchmarking env, continue on blocks other than L3\n");
			// 	if (bi<1) continue;	// We benchmark only L3-sized tensors(!)
			// #endif

			#if (TEST_ENV == 1)
				#ifdef RANDOMIZE
				// printf("INFO: block_layout randomized\n");
				// randomize_array_sizet(block_layout, dim, block_array.a[0]);
				// #else
				printf("INFO: block_layout = 2\n");
				for (int i=0; i<dim; ++i) block_layout[i] = 1;
				#endif
			#else
				for (int j=0; j<dim; ++j) block_layout[j] = block_array.a[bi]; // Get block_layout correctly represent the block layout
			#endif

			// HERE QUICK FIX (!)
			// 4 socket machine
			switch (dim) {
			case 4:
				block_layout[dim-1] = 12;
			break;
			case 5:
				block_layout[dim-1] = 6;
			break;
			}
			// // 8 socket machine
			// switch (dim) {
			// case 3:
			// 	block_layout[0] = 34;
			// 	block_layout[dim-1] = 34;
			// break;
			// case 4:
			// 	block_layout[0] = 12;
			// 	block_layout[dim-1] = 12;
			// break;
			// case 5:
			// 	block_layout[0] = 6;
			// 	block_layout[1] = 10;
			// 	block_layout[dim-2] = 10;
			// 	block_layout[dim-1] = 6;
			// break;
			// }

			size_t total_block_size = get_size(block_layout, dim);
			// Feed it into the tensor memory (and print it)
			fill_array_sizet_except_mode(block_layout, tensor->block_layout, tensor->dim+1, tensor->dim);
			fill_array_sizet_except_mode(block_layout, tensor2->block_layout, tensor2->dim+1, tensor2->dim);
			printf("INFO: BLOCKS: ");
			print_to_console_sizet(tensor->block_layout, tensor->dim);

			// Change n of the tensor (!)
			for (int test_n=0; test_n <= 0; ++test_n) {

				size_t tensize_max_temp = tensize_max / ((int) pow(2,test_n));
				printf("   N = %d\n", test_n);

				for (int parameter_p=0; parameter_p <= 0; ++parameter_p) { // Changes the distribution of the tensor (!)

					tensor->p = parameter_p;
					printf("INFO: DISTRIBUTIONS: tensor (%d)\n", tensor->p);

					int reference_block_n = tensor->block_layout[0];
					// Make block length along that dimension "special"
					#if (TEST_ENV == 0)
						// if (tensor->block_layout[tensor->p] > 1) tensor->block_layout[tensor->p] = tensor->block_layout[tensor->p]/2;
						// for (int i=0; i<dim; ++i) tensor->block_layout[i] = 1;
						// tensor->block_layout[0] = 4;
						// tensor->block_layout[dim-2] = SINGLESIZE;
						// tensor->block_layout[dim-1] = SINGLESIZE;
					#endif

					// #ifdef SYNCHONREMOTE
					// for (size_t pp=1; pp<=max_pp; ++pp) {
					// #endif
					for (int pp=1; pp>=0; --pp) {
						
						// We go through different p-arrays
						// assert(p_size == 10);
						switch (pp) {

						case 0: // 0-sync case
							for (int i=0; i<dim; ++i) p_array.a[i] = 1;
							p_array.a[tensor->p] = p_size;
							assert(get_size(p_array.a, dim) == (size_t) p_size);
						break;

						case 1: // p-1-sync case
							switch (p_size) {

							case 4: // #ifdef SYNCHONREMOTE
								p_array = (struct array){{4, 2, 2}};
							break;

							// The following 2 are for single-socket and multi-socket ivy2 machine
							case 10:
								switch (dim) {
								case 2:
									p_array = (struct array){{p_size, p_size}};
								break;
								case 3:
									p_array = (struct array){{p_size, 5, 2}};
								break;
								case 4:
									p_array = (struct array){{p_size, 5, 2, 1}};
								break;
								case 5:
									p_array = (struct array){{p_size, 5, 1, 2, 1}};
								break;
								}
							break;

							case 20:
								switch (dim) {
								case 2: 
									p_array = (struct array){{p_size, p_size}};
								break;
								case 3:
									p_array = (struct array){{p_size, 5, 4}};
								break;
								case 4:
									p_array = (struct array){{p_size, 5, 2, 2}};
								break;
								case 5:
									p_array = (struct array){{p_size, 5, 1, 2, 2}};
								break;
								}
							break;

							// The following is for single-socket ivy4s and ivy8s

							case 15:
								switch (dim) {
								case 2: 
									p_array = (struct array){{p_size, p_size}};
								break;
								case 3:
									p_array = (struct array){{p_size, 5, 3}};
								break;
								case 4:
									p_array = (struct array){{p_size, 5, 3, 1}};
								break;
								case 5:
									p_array = (struct array){{p_size, 5, 3, 1, 1}};
								break;
								}
							break;

							case 60:
								switch (dim) {
								case 2:
									p_array = (struct array){{p_size, p_size}};
								break;
								case 3:
									p_array = (struct array){{p_size, 10, 6}};
								break;
								case 4:
									p_array = (struct array){{p_size, 5, 3, 4}};
								break;
								case 5:
									p_array = (struct array){{p_size, 5, 3, 2, 2}};
								break;
								}
							break;

							case 120:
								switch (dim) {
								case 2:
									p_array = (struct array){{p_size, p_size}};
								break;
								case 3:
									p_array = (struct array){{p_size, 10, 12}};
								break;
								case 4:
									p_array = (struct array){{p_size, 5, 3, 8}};
								break;
								case 5:
									p_array = (struct array){{p_size, 5, 3, 2, 4}};
								break;
								}
							break;

							default:
								for (int i=0; i<dim; ++i) p_array.a[i] = p_size;
							break;	

							}
							// ERROR: Violated p-assumption; We have less than p stripes per partition (threads are STALLED)
							assert(get_size(p_array.a, dim) >= (size_t) p_size*p_size);
						break;
						}

						// PREPARING THE TENSOR (!)
						printf("INFO: SETTING TENSOR[1]\n");
						assert(p_array.a[tensor->p] == (size_t) p_size);

						printf("   INFO: (IMPORTANT -- p_size array!!!) STRIPES: ");
						print_to_console_sizet(p_array.a, dim);
						// int stripes = get_size(p_array.a, dim);

						// This code assumes SQUARE block sizes (otherwise the total size would be calculate differently)
						// double n = pow(tensize_max_temp, 1.0/(double)(dim));
						// printf("INFO: We have %f per dim (total tensize is %zu).\n", n, tensize_max_temp);
						// // Get the p_size dimension out of the way
						// double temp_double = n / (double)(tensor->block_layout[tensor->p]*p_array.a[tensor->p]);
						// int temp;
						// if (round(temp_double) - temp_double < 1e-6) {
						// 	temp = round(temp_double);
						// } else {
						// 	temp = (int) (n / (double)(tensor->block_layout[tensor->p]*p_array.a[tensor->p])); // truncate the double (down)
						// }
						// if (temp == 0) {
						// 	temp = 1; // assume temp must fit (!)
						// 	size_t adjusted_tensize = (tensize_max / ((int) pow(2,test_n))) / (double) (p_array.a[tensor->p]*tensor->block_layout[tensor->p]);
						// 	printf("INFO: Adjusted tensize is %zu\n", adjusted_tensize);
						// 	n = pow(adjusted_tensize, 1.0/(double)(dim-1));
						// 	// readjust the n for the guys below
						// 	printf("INFO: (After) We have reduced it %f per dim (total tensize is %zu).\n", n, adjusted_tensize);

						// }
						// layout2[tensor->p] = temp * tensor->block_layout[tensor->p];
						// layout[tensor->p] = p_array.a[tensor->p] * layout2[tensor->p];

						// If pp=0, then we can reuse already computed layout so just get layout2 (!)
						// layout2 = layout / p_array.a

						// if (pp == 1) {

							struct array size_array;
							// switch(dim) {
							// 	case 2:
							// 		size_array = (struct array){{68400, 68400}}; 
							// 	break;
							// 	case 3:
							// 		size_array = (struct array){{1360, 1700, 1632}};
							// 	break;
							// 	case 4:
							// 		size_array = (struct array){{6440, 220, 220, 220}}; 
							// 	break;
							// 	case 5:
							// 		size_array = (struct array){{240, 60, 48, 48, 60}};
							// 	break;
							// }

							#if (TEST_ENV == 1)
							// important that for this test, p == 3 (!)
							assert(p_size == 3);
							switch(dim) {
								case 2:
									size_array = (struct array){{1710, 1710}};
								break;
								case 3:
									size_array = (struct array){{204, 204, 204}};
								break;
								case 4:
									size_array = (struct array){{66, 66, 66, 66}};
								break;
								// case 5:
								// 	size_array = (struct array){{720, 60, 36, 24, 720}};
								// break;
							}

							#else
							// This works on 4 sockets (!)
							switch(dim) {
								case 2:
									size_array = (struct array){{68400, 68400}};
								break;
								case 3:
									size_array = (struct array){{4080, 680, 4080}};
								break;
								case 4:
									size_array = (struct array){{1320, 110, 132, 720}};
								break;
								case 5:
									size_array = (struct array){{720, 60, 36, 24, 360}};
								break;
							}
							// This works on 2 sockets (!)
							// switch(dim) {
							// 	case 2:
							// 		size_array = (struct array){{45600, 45600}};
							// 	break;
							// 	case 3:
							// 		size_array = (struct array){{1360, 1360, 1360}};
							// 	break;
							// 	case 4:
							// 		size_array = (struct array){{440, 110, 88, 440}};
							// 	break;
							// 	case 5:
							// 		size_array = (struct array){{240, 60, 36, 24, 240}};
							// 	break;
							// }
							// This works on 8 sockets (!)
							// switch(dim) {
							// 	case 2:
							// 		size_array = (struct array){{136800, 136800}};
							// 	break;
							// 	case 3:
							// 		size_array = (struct array){{4080, 680, 4080}};
							// 	break;
							// 	case 4:
							// 		size_array = (struct array){{1440, 110, 66, 1440}};
							// 	break;
							// 	case 5:
							// 		size_array = (struct array){{720, 50, 36, 20, 720}};
							// 	break;
							// }
							#endif

							printf("INFO: BLOCKS (in a stripe): ");
							for (int j=0; j<dim; ++j) {
								layout[j] = size_array.a[j];
								layout2[j] = layout[j] / p_array.a[j];
								assert(p_array.a[j] * layout2[j] == layout[j]);
								printf("%d, ", (int) (layout2[j] / tensor->block_layout[j]));
							}

							// // This is where we are most restrictive -- compute this (!)
							// printf("INFO: BLOCKS (in a stripe): ");
							// for (int j=0; j<dim; ++j) {
							// 	if (j == tensor->p) continue;
							// 	// printf("size_per_dim = %f\n", n);
							// 	// cast is broken (truncation)
							// 	double temp_double = n / (double)(tensor->block_layout[j]*p_array.a[j]);
							// 	int temp;
							// 	if (round(temp_double) - temp_double < 1e-6) {
							// 		temp = round(temp_double);
							// 	} else {
							// 		temp = (int) (n / (double)(tensor->block_layout[j]*p_array.a[j])); // truncate the double (down)
							// 	}
							// 	if (temp == 0) {
							// 		printf("ERROR: we could not fit even one block (%zu) times p_array (%zu) in n (%f)\n",
							// 			tensor->block_layout[j], p_array.a[j], n);
							// 	}
							// 	assert(temp != 0);
							// 	#ifdef COMPLETEMORTON
							// 	// Idea: make sure temp is a power of 2 (otherwise the morton-curve is unbalanced;)
							// 	int temp_rounded = 1;
							// 	while (temp > 1) {
							// 		temp >>= 1;
							// 		temp_rounded <<= 1;
							// 	}
							// 	// printf("temp = %d rounded to power of 2 is %d\n", temp, temp_rounded);
							// 	temp = temp_rounded;
							// 	#endif
							// 	layout2[j] = temp * tensor->block_layout[j];
							// 	layout[j] = p_array.a[j] * layout2[j];
							// 	printf("%d, ", temp);
							// } printf("\n");
						// } else {
							// Simply use aleady computed sizes in the previous iteration (!)
							// Simply modify layout2 (!)
							// printf("INFO: BLOCKS (in a stripe): ");
							// for (int j=0; j<dim; ++j) {
							// 	if (j == tensor->p) continue;
							// 	layout2[j] = layout[j] / p_array.a[j];
							// 	assert(p_array.a[j] * layout2[j] == layout[j]);
							// 	printf("%d, ", temp);
							// } printf("\n");
						// }

						// Feed it into the tensor memory (and print it)
						fill_array_sizet_except_mode(layout, tensor->layout, tensor->dim+1, tensor->dim);
						fill_array_sizet_except_mode(layout2, tensor->layout2, tensor->dim+1, tensor->dim);

						// fill_array_sizet_except_mode(p_sizes, tensor->lin.p_sizes, tensor->dim+1, tensor->dim);
						printf("INFO: BLOCK_LAYOUT: ");
						print_to_console_sizet(tensor->block_layout, tensor->dim);
						printf("INFO: LAYOUT2 (STRIPES): ");
						print_to_console_sizet(tensor->layout2, tensor->dim);
						printf("INFO: LAYOUT: ");
						print_to_console_sizet(tensor->layout, tensor->dim);

						tensor->lin.size = get_size(tensor->layout, tensor->dim);
						for (int i=0; i<dim; ++i) assert(tensor->layout[i] != 0);

						// This checks whether we will need tensor2 in this benchmark run
						// Which we will need if this is the 0-sync situation
						if (pp == 0 && parameter_p == 0) {

							tensor2->p = dim-1-parameter_p;
							assert(tensor->p != tensor2->p);
							#if (TEST_ENV == 0)
								// if (tensor2->block_layout[tensor2->p] > 1) tensor2->block_layout[tensor2->p] = tensor2->block_layout[tensor2->p]/2;
								// for (int i=0; i<dim; ++i) tensor2->block_layout[i] = 1;
								// tensor2->block_layout[dim-2] = SINGLESIZE;
								// tensor2->block_layout[dim-1] = SINGLESIZE;
								// Fix, otherwise the sizes of tensor2 WILL NOT match tensor
								// if (dim >= 3) {
									// tensor2->block_layout[dim-3] = SINGLESIZE;
									// tensor2->block_layout[tensor2->p] = 4;
								// }
							#endif
							printf("INFO: DISTRIBUTIONS: tensor2 (%d)\n", tensor2->p);

							printf("INFO: SETTING TENSOR[2]\n");

							// ALl we need to do is just recalculate the number of blocks
							// DO NOT MODIFY TENSOR SIZE (!)
							// DO NOT MODIFY P_ARRAY (!) of tensor (???)
							// ok, actually it maybe different because now we have to distribute along rightmost mode...

							size_t save_me = p_array.a[tensor2->p];
							p_array.a[tensor2->p] = p_array.a[tensor->p];
							p_array.a[tensor->p] = save_me;

							for (int j=0; j<dim; ++j) {
								if (j == tensor2->p) continue;
								assert(p_array.a[j] == 1);
							}

							printf("INFO: BLOCKS (in a stripe): ");
							for (int j=0; j<dim; ++j) {
								layout2[j] = layout[j] / p_array.a[j];
								assert(p_array.a[j] * layout2[j] == layout[j]);
								printf("%d, ", (int) (layout2[j] / tensor->block_layout[j]));
							}

							// Swap p_arrays (!)
							// Just swap tensor2->p with tensor->p (!)

							// You just need to readjust -- do not modify the layout (!)
							// layout2[tensor2->p] = layout[tensor2->p] / p_array.a[tensor2->p];
							// layout2[tensor->p] = layout[tensor->p] / p_array.a[tensor->p];

							// save_me = layout[tensor2->p];
							// layout[tensor2->p] = layout[tensor->p];
							// layout[tensor->p] = save_me;

							// for (int i=0; i<dim; ++i) p_array.a[i] = 1;
							// p_array.a[tensor2->p] = p_size;
							assert(get_size(p_array.a, dim) == (size_t) p_size);
							assert(p_array.a[tensor2->p] == (size_t) p_size);

							printf("INFO: STRIPES: ");
							print_to_console_sizet(p_array.a, dim);
							// stripes = get_size(p_array.a, dim);
							
							// if (pp == 1) {
							// 	// This code assumes SQUARE block sizes (otherwise the total size would be calculate differently)
							// 	double n = pow(tensize_max_temp, 1.0/(double)(dim));
							// 	// Get the p_size dimension out of the way
							// 	double temp_double = n / (double)(tensor2->block_layout[tensor2->p]*p_array.a[tensor2->p]);
							// 	int temp;
							// 	if (round(temp_double) - temp_double < 1e-6) {
							// 		temp = round(temp_double);
							// 	} else {
							// 		temp = (int) (n / (double)(tensor2->block_layout[tensor2->p]*p_array.a[tensor2->p])); // truncate the double (down)
							// 	}
							// 	if (temp == 0) {
							// 		temp = 1; // assume temp must fit (!)
							// 		size_t adjusted_tensize = (tensize_max / ((int) pow(2,test_n))) / (double) (p_array.a[tensor2->p]*tensor2->block_layout[tensor2->p]);
							// 		printf("INFO: Adjusted tensize is %zu\n", adjusted_tensize);
							// 		n = pow(adjusted_tensize, 1.0/(double)(dim-1));
							// 		// readjust the n for the guys below
							// 		printf("INFO: (After) We have reduced it %f per dim (total tensize is %zu).\n", n, adjusted_tensize);

							// 	}
							// 	layout2[tensor2->p] = temp * tensor2->block_layout[tensor2->p];
							// 	layout[tensor2->p] = p_array.a[tensor2->p] * layout2[tensor2->p];

							// 	// This is where we are most restrictive -- compute this (!)
							// 	printf("INFO: BLOCKS (in a stripe): ");
							// 	for (int j=0; j<dim; ++j) {
							// 		if (j == tensor2->p) continue;
							// 		// printf("size_per_dim = %f\n", n);
							// 		// cast is broken (truncation)
							// 		double temp_double = n / (double)(tensor2->block_layout[j]*p_array.a[j]);
							// 		int temp;
							// 		if (round(temp_double) - temp_double < 1e-6) {
							// 			temp = round(temp_double);
							// 		} else {
							// 			temp = (int) (n / (double)(tensor2->block_layout[j]*p_array.a[j])); // truncate the double (down)
							// 		}
							// 		assert(temp != 0);
							// 		#ifdef COMPLETEMORTON
							// 		// Idea: make sure temp is a power of 2 (otherwise the morton-curve is unbalanced;)
							// 		int temp_rounded = 1;
							// 		while (temp > 1) {
							// 			temp >>= 1;
							// 			temp_rounded <<= 1;
							// 		}
							// 		// printf("temp = %d rounded to power of 2 is %d\n", temp, temp_rounded);
							// 		temp = temp_rounded;
							// 		#endif
							// 		layout2[j] = temp * tensor2->block_layout[j];
							// 		layout[j] = p_array.a[j] * layout2[j];
							// 		printf("%d, ", temp);
							// 	} printf("\n");
							// } else {
							// }

							// printf("INFO: BLOCKS (in a stripe): ");
							// for (int j=0; j<dim; ++j) {
							// 	layout[j] = size_array.a[j];
							// 	layout2[j] = layout[j] / p_array.a[j];
							// 	assert(p_array.a[j] * layout2[j] == layout[j]);
							// 	printf("%d, ", (int) layout2[j] / tensor->block_layout[j]);
							// }

							// Feed it into the tensor memory (and print it)
							fill_array_sizet_except_mode(layout, tensor2->layout, tensor2->dim+1, tensor2->dim);
							fill_array_sizet_except_mode(layout2, tensor2->layout2, tensor2->dim+1, tensor2->dim);

							// fill_array_sizet_except_mode(p_sizes, tensor2->lin.p_sizes, tensor2->dim+1, tensor2->dim);
							printf("INFO: BLOCK_LAYOUT: ");
							print_to_console_sizet(tensor2->block_layout, tensor2->dim);
							printf("INFO: LAYOUT2 (STRIPES): ");
							print_to_console_sizet(tensor2->layout2, tensor2->dim);
							printf("INFO: LAYOUT: ");
							print_to_console_sizet(tensor2->layout, tensor2->dim);

							tensor2->lin.size = get_size(tensor2->layout, tensor2->dim);
							for (int i=0; i<dim; ++i) assert(tensor2->layout[i] != 0);
							// Prepare the tensor ONCE for all the modes (!)
							// printf("INFO: P_SIZE of tensor2: %d\n", tensor2->lin.p_size);

							// Assert this and we are fine (!)
							// Fuck -- why do we need the 2? I just keep this one below...
							// assert(tensor->layout[tensor->p] == tensor2->layout[tensor2->p]);
							for (int i=0; i<dim; ++i) {
								if (i == tensor->p || i == tensor2->p) {
									continue;
								}
								// This will not be true anymore, after adding unequal block sizes
								// Hence added assertion below about the total size
								// assert(tensor->layout[i] == tensor2->layout[i]);
							}
							assert(get_size(tensor->layout, dim) == get_size(tensor2->layout, dim));
							// assert(tensor->layout[tensor->p] == tensor2->layout[tensor2->p]);
							// assert(tensor2->layout[tensor->p] == tensor->layout[tensor2->p]);

						}
						// printf("INFO: P_SIZE of tensor: %d\n", tensor->lin.p_size);

						// Tensor2 will be used for mode tensor->p in which case we will go back to tensor2 (!) distribution
						// PROBLEM: we cannot really sucessfully test this (!)

						// For a particular mode: determines the result (sizes!) and the vector (size!);
						// necessary for correctness in BOTH test and benchmark scenario
						// PREPARES the result only if it's TEST_ENV
						for (int mode=0; mode<dim; ++mode) {
							printf("         MODE %d\n", mode);

							struct tensor_storage * runtime_tensor = tensor;
							if (mode == tensor->p && pp == 0 && parameter_p == 0) {
								runtime_tensor = tensor2;
								printf("INFO: runtime_tensor = tensor2\n");
							} else {
								printf("INFO: runtime_tensor = tensor\n");
							}
							
							fill_array_sizet_except_mode(runtime_tensor->block_layout, result->block_layout, runtime_tensor->dim, mode);
							fill_array_sizet_except_mode(runtime_tensor->layout, result->layout, runtime_tensor->dim, mode);
							fill_array_sizet_except_mode(runtime_tensor->layout2, result->layout2, runtime_tensor->dim, mode);
							result->lin.size = get_size(result->layout, result->dim);
							vector->size = runtime_tensor->layout[mode];
							printf("INFO: Tensor_size=%zu (max=%zu), result_size=%zu (max=%zu), vector_size=%zu (max=%zu)\n", runtime_tensor->lin.size, tensor_size, runtime_tensor->lin.size/vector->size, result_size, vector->size, vector_size);
							assert(runtime_tensor->lin.size <= tensor_size);
							assert(runtime_tensor->lin.size/vector->size <= result_size);
							assert(vector->size <= vector_size);
							runtime_tensor->lin.data = tensor->lin.data;
							
							// FIRST CATEGORY OF ALGORITHMS (preparing the result FOR THEM is different)

							// The following is a sequential run to produce correct result
							#if (TEST_ENV == 1) && !defined(LIKWID_PERFMON)
								omp_set_num_threads(1);
								mkl_set_num_threads(1);
								printf("           MODEL_SEQ_ALGORITHM (LIKWID_PERFMON not defined; We proceed with verification tests!)\n");
								reset_array(result->lin.data, result->lin.size, 0.0);
								// TENSOR -> RESULT 
								tvm_vector_major_BLAS_col_mode(runtime_tensor, vector, &result->lin, mode);
								// printf("hey guys, so we produced a simple result and it's...\n");
								// print_to_console(result->lin.data, result->lin.size);
								// TENSOR -> MASTER_TENSOR (blocked)
								block_morton_block_unfold(runtime_tensor, vector, &runtime_tensor->lin, runtime_tensor->p);
								// RESULT -> MASTER_RESULT (blocked)
								if (mode == runtime_tensor->p) {
									// printf("INFO: Result is distributed along mode 0\n");
									block_morton_block_unfold(result, vector, &result->lin, 0);
								} else if (runtime_tensor->p < mode) {
									// printf("INFO: Result is distributed along mode %zu\n", runtime_tensor->p);
									block_morton_block_unfold(result, vector, &result->lin, runtime_tensor->p);
								} else {
									// printf("INFO: Result is distributed along mode %zu\n", runtime_tensor->p-1);
									block_morton_block_unfold(result, vector, &result->lin, runtime_tensor->p-1);
								}
								// Prepare the vector (!) elements might be misaligned
								if (mode == runtime_tensor->p) {
									omp_set_num_threads(p_size);
									size_t next = 0;
									#pragma omp parallel
									{	
										int tid = omp_get_thread_num();
										int nthreads = omp_get_num_threads();
										int partition_size = vector_size / nthreads;
										int logical_partition_size = runtime_tensor->layout[runtime_tensor->p] / p_size;
										#pragma omp for ordered schedule(static,1)
									    for (int t=0; t<nthreads; ++t)
									    {
									        assert( t==tid );
									        #pragma omp ordered
									        {
												int el = 0;
												for (int pv=0; pv<logical_partition_size; ++pv) {
													vector->local_data[tid][el++] = vector->data[next++];
												}
									        }
								        }
									}
								}
							#endif

							// The following is a parallel run (either for test or benchmark)
							for (int i=0; i<algo_max_par; ++i) {

								#ifdef TESTP
								// IDEA OF COMPREHENSIVE TESTS (but only if time allows!)
								// If it's 0-sync do it only if mode is different than tensor->p
								// if ((pp == 0) && (mode == tensor->p)) {
								// 	continue;
								// }
								#endif

								printf("            PAR_ALGORITHM %d\n", i);
								omp_set_num_threads(p_size);
								mkl_set_num_threads(1); // Must set this here because the algorithm might have messed this up (!)

								#if (TEST_ENV == 1) // Reset the result in case it's been polluted by the previous parallel algorithm
								#if !defined(LIKWID_PERFMON)
									if (mode == runtime_tensor->p) {
										reset_array(result->lin.data, result->lin.size, 0.0);
									} else {
										reset_array_double_locally(&result->lin);
									}
									algorithms[i](runtime_tensor, vector, &result->lin, mode);
									if (mode == runtime_tensor->p) { // VERIFY THE RESULT (policy depends on the mode)
										// Idea: Comparison depends on the mode
										// printf("INFO: Verifying against result->data\n");
										if (!verify_data_master_data_results(&result->lin)) {
											printf("We found an error!\n");
											exit(-1);
										} else {
											// printf("INFO: Algorithm passed the test.\n");
										}
										// if (!verify_data_parallel_results(&result->lin, 1)) {
										// 	printf("We found an error!\n");
										// 	exit(-1);
										// }
									} else {
										// Idea: Comparison depends on the mode
										// printf("INFO: Verifying against result->local_data[tid]\n");
										if (!verify_data_parallel_results(&result->lin, 1)) {
											printf("We found an error!\n");
											exit(-1);
										} else {
											// printf("INFO: Algorithm passed the test.\n");
										}
									}
								#else
									LIKWID_MARKER_START(toString(algorithms[i]));
									algorithms[i](runtime_tensor, vector, &result->lin, mode);
									LIKWID_MARKER_STOP(toString(algorithms[i]));
								#endif

								#else
								measure_multicore(
									algorithms[i], runtime_tensor, vector, &result->lin, mode,
									file, runtime_tensor->layout[mode], reference_block_n, total_block_size);
								fflush(file);
								#endif
							} // loop over algorithms

							// DO THE BELOW ONLY IF pp == 1
							if (pp == 1) {

								// The following is a sequential run to produce correct result (for unfold algorithms)
								#if (TEST_ENV == 1) && !defined(LIKWID_PERFMON)
									omp_set_num_threads(1);
									mkl_set_num_threads(1);
									reset_array(result->lin.data, result->lin.size, 0.0);
									tvm_vector_major_BLAS_col_mode(runtime_tensor, vector, &result->lin, mode);
									// printf("hey guys, so we produced a simple result (again) and it's...\n");
									// print_to_console(result->lin.data, result->lin.size);
									// copy result->lin.data into result->lin.master_data
									memcpy(result->lin.master_data, result->lin.data, result->lin.size * sizeof(DTYPE));
									// printf("hey guys, so we copied into master data and it's....\n");
									// print_to_console(result->lin.master_data, result->lin.size);
								#endif

								// The following is a parallel run (either for test or benchmark)
								for (int i=0; i<algo_max_par_unfold; ++i) {
									printf("            PAR_ALGORITHM_UNFOLD %d\n", i);
									omp_set_num_threads(p_size);
									mkl_set_num_threads(1); // Must set this here because the algorithm might have messed this up (!)
									#if (TEST_ENV == 1) // Reset the result in case it's been polluted by the previous parallel algorithm
									#if !defined(LIKWID_PERFMON)
										reset_array(result->lin.data, result->lin.size, 0.0);
										algorithms_unfold[i](runtime_tensor, vector, &result->lin, mode);
										// printf("hey guys, so we produced a simple result (with the unfold algo) and it's...\n");
										// print_to_console(result->lin.data, result->lin.size);
										printf("INFO: Verifying against result->data\n");
										if (!verify_data_master_data_results(&result->lin)) {
											printf("We found an error!\n");
											exit(-1);
										} else {
											// printf("INFO: Algorithm passed the test.\n");
										}
									#else
										LIKWID_MARKER_START(toString(algorithms_unfold[i]));
										algorithms_unfold[i](runtime_tensor, vector, &result->lin, mode);
										LIKWID_MARKER_STOP(toString(algorithms_unfold[i]));
									#endif
									#else
									measure_multicore(
										algorithms_unfold[i], runtime_tensor, vector, &result->lin, mode,
										file, runtime_tensor->layout[mode], reference_block_n, total_block_size);
									fflush(file);
									#endif
								} // loop over algorithms

							}

							// We only run the baselines for pp = 1
							if (pp == 1) {
								// // The following is a run over sequential algorithms
								for (int i=0; i<algo_max_seq; ++i) {
									printf("            SEQ_ALGORITHM %d\n", i);
									omp_set_num_threads(1);
									mkl_set_num_threads(1); // Must set this here because the algorithm might have messed this up (!)
									#if (TEST_ENV == 1)
									#if defined(LIKWID_PERFMON)
									LIKWID_MARKER_START(toString(algorithms_seq[i]));
									algorithms_seq[i](tensor, vector, &result->lin, mode);
									LIKWID_MARKER_STOP(toString(algorithms_seq[i]));
									#endif
									#else 
									measure(
										algorithms_seq[i], runtime_tensor, vector, &result->lin, mode,
										file, runtime_tensor->layout[mode], tensor->block_layout[mode], total_block_size);
									fflush(file);
									#endif
								} // loop over algorithms					
							}


						} // loop over modes

					} // loop over kinds of p_arrays

				} // loop over P

			} // test_n (check different ns)

		} // loop over blocks 

	} // loop over dims

	// Set the sizes back correctly in case we use numa to free (!)
	tensor->lin.size = tensor_size;
	result->lin.size = result_size;
	vector->size = vector_size;

	// FREE TENSOR MEMORY, local_allocation ? 0:both, 1:local, 2:interleaved
	omp_set_num_threads(p_size); // Very important that we bring back the p_size to free the local tensors
	printf("INFO: Free the tensor memory\n");
	free_tensor_storage_safe(tensor, 0);
	tensor2->lin.data = NULL;
	free_tensor_storage_safe(tensor2, 0);
	printf("INFO: Free the result memory\n");
	free_tensor_storage_safe(result, 0);
	printf("INFO: Free the vector memory\n");
	free_lin_storage_safe(vector, 0);

    // Close Marker API and write results to file for further evaluation done by likwid-perfctr
    LIKWID_MARKER_CLOSE;

	// CLOSING THE FILE
	if (file != NULL) {
		fclose(file);
	}

	return 0;

}
