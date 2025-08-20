#include <rand_utils.h>
#include <stdlib.h> // For random(), RAND_MAX
#include <sys/time.h> // For gettimeofday, part of POSIX
#include <stdio.h>
#include <math.h>

// rand() and srand() are part of the standard library of C
// random() and srandom() are part of POSIX and not available on Windows
/* Numerical Recipes book: "be very, very suspicious of a system-supplied rand(); System-supplied rand()s are almost always linear congruential generators */
// In GNU library, RAND_MAX is 2147483647 but portable code cannot assume more than 32767

//  C or C++ offer random(), which returns a long integer in the range 0 to RAND_MAX, so ((double)random())/RAND_MAX is a double that lies in the range 0.0 to 1.0. 

// Generates an integer between [min,max)
DTYPE
rand_int(int min, int max) {
	// http://www.eternallyconfuzzled.com/arts/jsw_art_rand.aspx
	// math implementation: http://stackoverflow.com/questions/2509679/how-to-generate-a-random-number-from-within-a-range
	double r;
	r = rand_double();

	//printf("WHAT DO YOU THINK????\n");
	int range = max-min;
	// printf("%lf\n", r*range);
	return (DTYPE) (r*range) + min;
}

int
rand_binary() {
	double r = rand_double();
	return round(r);
}

// Ensures values between 0 and 1
double
rand_double() {
	// printf("random()=%lf\n", (double)random());
	// printf("Random is: %f\n", (double)random()/(RAND_MAX+1.0));
	return (double)random()/(RAND_MAX+1.0);
}

double
rand_double_full() {
	return (double)random();
}

void
set_seed(int seed) {
	srandom(seed);
}

int
set_random_seed() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	int seed = (int) (tv.tv_sec + tv.tv_usec);
	srandom(seed);
	return seed;
}
