#include<time.h>

#define MILLION 1000000L
#define BILLION 1000000000L
#define THOUSAND 1000.0

struct timespec
timespec_diff(const struct timespec start, const struct timespec end) {
	struct timespec temp;
	if ((end.tv_nsec-start.tv_nsec) < 0) {
		temp.tv_sec = (double)end.tv_sec - start.tv_sec - 1;
		temp.tv_nsec = BILLION + (double)end.tv_nsec - start.tv_nsec;
	} else {
		temp.tv_sec = (double)end.tv_sec-start.tv_sec;
		temp.tv_nsec = (double)end.tv_nsec-(double)start.tv_nsec;
	}
	return (const struct timespec) temp;
}

double
timespec_to_microseconds(const struct timespec diff) {
	// printf("for now, MILLION*diff.tv_sec=%.17g, diff.tv_nsec/THOUSAND=%.17g\n", MILLION*diff.tv_sec, diff.tv_nsec/THOUSAND);
	// printf("and now sum = %.17g\n",MILLION*diff.tv_sec + diff.tv_nsec/THOUSAND);
	return (MILLION*diff.tv_sec + diff.tv_nsec/THOUSAND);
}

 /**
  * timeval_to_ns - Convert timeval to nanoseconds
  * @ts:         pointer to the timeval variable to be converted
  *
  * Returns the scalar nanosecond representation of the timeval
  * parameter.
  */


double
timespec_to_seconds(const struct timespec diff) {
	return (diff.tv_sec + diff.tv_nsec/BILLION);
}

