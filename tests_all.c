#include <stdio.h>
#include <stdlib.h>
#include <test.h>
#include <unistd.h>
// for the folder checking/creation
#include <sys/types.h>
#include <sys/stat.h>
#include <numa.h>

// *argv is AN array of pointers to the strings
// argv is a pointer to AN array of pointers
int main( int argc, char *argv[] ) {

	/** Initialize the library; pay for setup cost at a specific point. */
	void libxsmm_init(void);

	numa_set_localalloc(); // We set this just after main()
		
	struct stat st = {0};
	if (stat(FOLDER, &st) == -1) {
		printf("Directory %s doesn't exist, create it\n", FOLDER);
		if (mkdir(FOLDER, 0777) == -1) {
			printf("Error creating directory! Exit the program.\n");
			exit(-1);
		}
	}
	
	// *argv dereferences to get another pointer inside that array
	printf("Script name: %s\n", *(argv) );
	// we do -- on argc to say we used this element
	--argc;

	// argc is given by the system (as argv is)
	if (argc == 0) {
		// provide default params
	} else if (argc == 2) {
		// dim, dim_max
	} else if (argc == 4) {
		// dim, n
	} else if (argc == 3) {
		// dim, dim, max_algo
	} else if (argc == 5) {
		// dim, n, block_n
	} else if (argc == 6) {
		// dim, n, mode
	} else if (argc == 7) {
		// dim, n, mode, block_n
	} else {
		printf("wrong number of params! exiting...\n");
		exit(EXIT_FAILURE);
	}
	
	// OFFTOPIC
	// if we want another element we must put brackets
	if (argc != 0) {
		printf("First argument: %s\n", *(argv+1) );
	}
	// this gets the first element in array and moves it to 1
	// so it prints the name without first character
	printf("Script name (without 2 chars): %s\n\n", (*argv)+2 );
	// why this works?
	// only because ++ has precedence EVEN over the pointer dereference!
	//while(argc--) {
		//printf("%s\n", *argv++);
	//}
	// END OF OFFTOPIC
	
	// Worth noting potential pitfalls and hidden cases:
	// 1) Case when block_n is higher than n: We quit that by not running such cases (blockmode: test4_multi_n)
	// 2) for blockmode, we compare with qsort: better method would be:
	// blockmode result storage depends on the mode -> I had a test like this:
	// if (mode==0) result_mode=0 else result_mode=mode-1 

	printf("%*s", 20, "test_units\n\n");
	// test_units();

	printf("%*s", 20, "test0_powers\n\n");
	// test0_powers(argc, argv);

	printf("%*s", 20, "test0_ht\n\n");
	// test0_ht(argc, argv);
	
	printf("%*s", 20, "test0_ht_multicore\n\n");
	// test0_ht_multicore(argc, argv);

	printf("%*s", 20, "test1\n\n");
	test1(argc, argv);

	printf("%*s", 20, "test_openmp\n\n");
	// test_openmp(argc, argv);

	printf("%*s", 20, "test3_multi_blockn\n\n");
	// test3_multi_blockn(argc, argv);

	printf("%*s", 20, "test4_multi_n\n\n");
	// test4_multi_n(argc, argv);

	printf("%*s", 20, "test5_multi_all\n\n");
	// test5_multi_all(argc, argv);

	printf("%*s", 20, "test_tmm\n\n");
	// test_tmm(argc, argv);

	printf("%*s", 20, "test_powermethod\n\n");
	// test_powermethod(argc, argv);

	/** De-initialize the library and free internal memory (optional). */
	void libxsmm_finalize(void);
	
	return 0;
}
