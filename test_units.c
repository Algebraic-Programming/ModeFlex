#include <algorithms.h>
#include <gen_utils.h> // for randomize array int 
#include <gen_data.h> // get_vector, gen_vector, gen_block_tensor
#include <file_utils.h> // for save_to_file
#include <test.h> // for inline functions
#include <rand_utils.h>
#include <stdlib.h> // for free
#include <string.h>
#include <assert.h>

#define TEST_SIZE 10000
typedef void (*TVM)(); // #include <bench_utils.h>

void
treat_as_dtype(DTYPE * const source, DTYPE * const dst, void (*f)()) {
	// Copy source into blocks repeatedly
	// Send a loop (!) of unaligned accesses
	// int i = 0;
	for(size_t block_size = 2; block_size <20; ++block_size) {
	// block_size = 3;
	// while (i++ < 10) {
		// size_t block_size = rand_int(10, 50);
		printf("Block size is %zu.\n", block_size);
		short cond = (TEST_SIZE % block_size);
		size_t ROUNDED_TEST_SIZE = (TEST_SIZE/block_size) * block_size;

		for (size_t i=0; i<ROUNDED_TEST_SIZE; i+=block_size) {
			// printf("Call for block with i=%d\n", i);//with pointers at dst+i=%p and source+i=%p\n", i, (void *)dst+i, (void *)source+i);
			f(dst, source+i, block_size);
			if (memcmp(source+i, dst, block_size*sizeof(DTYPE)) == 0) {
				// printf("Source/Dst (correct):\n");
				// print_to_console(source+i, block_size);
				// print_to_console(dst, block_size);
				// printf("Memory regions are identical.\n");
			} else {
				printf("Memory regions are DIFFERENT. Printing source and destination.\n");
				printf("Source/Dst (ERROR):\n");
				print_to_console(source+i, block_size);
				print_to_console(dst, block_size);
				exit(-1);
			}
		}
		// Extra last step if unequal block sizes
		if (cond) {
			size_t i = ROUNDED_TEST_SIZE;
			f(dst, source+i, cond);
			if (memcmp(source+i, dst, cond*sizeof(DTYPE)) == 0) {
				// printf("Source/Dst (correct):\n");
				// print_to_console(source+i, cond);
				// print_to_console(dst, cond);
				// printf("Memory regions are identical.\n");
			} else {
				printf("Memory regions are DIFFERENT. Printing source and destination.\n");
				printf("Source/Dst (ERROR):\n");
				print_to_console(source+i, cond);
				print_to_console(dst, cond);
				exit(-1);
			}
		}
	}
	// Only guarantee: first write is aligned (?)
	// Read block -> block_size is not always moving us to aligned boundary
	// Writing part of block -> (totally) out of alignment but we can maybe pad it (for vector size)
}

void
treat_as_char(DTYPE * const source, DTYPE * const dst, void (*f)()) {
	// Copy source into blocks repeatedly
	// Send a loop (!) of unaligned accesses
	int i = 0;
	while (i++ < 10) {
		size_t block_size = rand_int(10, 50);
		printf("Block size is %zu.\n", block_size);
		short cond = (TEST_SIZE % block_size);
		size_t ROUNDED_TEST_SIZE = (TEST_SIZE/block_size) * block_size;

		for (size_t i=0; i<ROUNDED_TEST_SIZE; i+=block_size) {
			// printf("Call for block with i=%d\n", i);//with pointers at dst+i=%p and source+i=%p\n", i, (void *)dst+i, (void *)source+i);
			f(dst, source+i, block_size*sizeof(DTYPE));
			if (memcmp(source+i, dst, block_size*sizeof(DTYPE)) == 0) {
				// printf("Source/Dst (correct):\n");
				// print_to_console(source+i, block_size);
				// print_to_console(dst, block_size);
				// printf("Memory regions are identical.\n");
			} else {
				printf("Memory regions are DIFFERENT. Printing source and destination.\n");
				printf("Source/Dst (ERROR):\n");
				print_to_console(source+i, block_size);
				print_to_console(dst, block_size);
				exit(-1);
			}
		}
		// Extra last step if unequal block sizes
		if (cond) {
			// printf("Call for the last block\n");
			f(dst, source+i, cond*sizeof(DTYPE));
			if (memcmp(source+i, dst, cond*sizeof(DTYPE)) == 0) {
				// printf("Source/Dst (correct):\n");
				// print_to_console(source+i, cond);
				// print_to_console(dst, cond);
				// printf("Memory regions are identical.\n");
			} else {
				printf("Memory regions are DIFFERENT. Printing source and destination.\n");
				printf("Source/Dst (ERROR):\n");
				print_to_console(source+i, cond);
				print_to_console(dst, cond);
				exit(-1);
			}
		}
	}
	// Only guarantee: first write is aligned (?)
	// Read block -> block_size is not always moving us to aligned boundary
	// Writing part of block -> (totally) out of alignment but we can maybe pad it (for vector size)
}

int
test_units() {

	DTYPE * source_memory = get_aligned_memory(TEST_SIZE * sizeof(DTYPE), ALIGNMENT);
	DTYPE * dst_memory = get_aligned_memory(TEST_SIZE * sizeof(DTYPE), ALIGNMENT);
	// based on malloc and NOT calloc
	// zero-out the source

	// hence use memset to "use" the memory
	memset(source_memory, 0, TEST_SIZE * sizeof(DTYPE));
	if (print_status("gen_array_int", source_memory)) {
		for (size_t i=0; i<TEST_SIZE; ++i) {
			source_memory[i] = rand_int(2,10);
		}
	}
	// printf("SOURCE:\n");
	// print_to_console(source_memory, TEST_SIZE);

	const int memcpy_dtype = 3;
	TVM memcpy_dtype_algo[3] = {
		nontemp_memcpy_by_2,
		nontemp_memcpy,
		CopyWithSSEPrefetchNT
		// CopyWithAVX,
		// CopyWithAVXNT
	};

	const int memcpy_char = 1;
	TVM memcpy_char_algo[1] = {
		memcpy // memcpy treats the memory as pure char* so we have to give all sizes mutliplied by sizeof(DTYPE) for the test to work(!)
	};

	for (int i=0; i<memcpy_dtype; ++i) {
		memset(dst_memory, 0, TEST_SIZE * sizeof(DTYPE));
		treat_as_dtype(source_memory, dst_memory, memcpy_dtype_algo[i]);
	}

	for (int i=0; i<memcpy_char; ++i) {
		memset(dst_memory, 0, TEST_SIZE * sizeof(DTYPE));
		treat_as_char(source_memory, dst_memory, memcpy_char_algo[i]);
	}

	// 0 - contents of both memory regions are identical
	free(source_memory);
	free(dst_memory);

	return 1;
}
