#include <stdio.h>
#include <scenario.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <bench_utils.h>

int
main( int argc, char *argv[] ) {

	/** Initialize the library; pay for setup cost at a specific point. */
	void libxsmm_init(void);

	struct stat st = {0};
	if (stat(FOLDER, &st) == -1) {
		printf("Directory %s doesn't exist, create it\n", FOLDER);
		if (mkdir(FOLDER, 0777) == -1) {
			printf("Error creating directory! Exit the program.\n");
			exit(-1);
		}
	}

	--argc;

	printf("%*s", 20, "scenario0\n\n");
	scenario0(argc, argv);
	printf("\n");

	// printf("%*s", 20, "scenario0\n\n");
	// scenario_tmm(argc, argv);
	// printf("\n");

	// printf("%*s", 20, "scenario0stream\n\n");
	// scenario0stream(argc, argv);
	// printf("\n");

	// printf("%*s", 20, "scenario_powermethod\n\n");
	// scenario_powermethod(argc, argv);
	// printf("\n");

#if 0
	printf("%*s", 20, "scenario1\n\n");
	scenario1(argc, argv);
	printf("\n");
#endif

	/** De-initialize the library and free internal memory (optional). */
	void libxsmm_finalize(void);

	return 0;
}
