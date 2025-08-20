#define ONE_OVER_BILLION 1E-9
// Implementation of functions defined in file_utils.h
#include <stdio.h>
#include <string.h>
#include <structures.h>
#include <stdlib.h>
// all intrinsics?
#include <immintrin.h>
#include <omp.h>

void
print256_num(__m256d var) {
    DTYPE *val = (DTYPE*) &var;
    printf("Numerical: %f %f %f %f \n", 
           val[0], val[1], val[2], val[3]);
}

void
print_to_console(DTYPE * ptr, size_t size) {
	for (size_t el=0; el<size; ++el) {

		// DTYPE DEPENDENT
		// for now: selected double!
		printf("el[%ld]=%lf, ", el, (double) *(ptr+el));

		// DTYPE DEPENDENT

	}
	printf("\n");
}

void
print_to_console_local(const struct lin_storage * storage) {

	#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		int nthreads = omp_get_num_threads();
		size_t partition_size = storage->size / nthreads;
		printf("partition_size = %zu / %d = %zu\n", storage->size, nthreads, storage->size / nthreads); 
		
		#pragma omp for ordered schedule(static,1)
	    for (int t=0; t<nthreads; ++t)
	    {
	        #pragma omp ordered
	        {
	        	printf("Thread %d printing output partition of size %zu\n", tid, partition_size);
	        	print_to_console(storage->local_data[tid], partition_size);
	        }
        }
    }
}

void
print_to_console_double(double * ptr, size_t size) {
	for (size_t el=0; el<size; ++el) {
		//printf("el[%ld]=%lf, ", el, *(ptr+el));
		printf("el[%ld]=%23.16e, ", el, *(ptr+el));
	}
	printf("\n");
}

void
print_to_console_int(int * ptr, size_t size) {
	for (size_t el=0; el<size; ++el) {
		printf("el[%ld]=%d, ", el, *(ptr+el));
	}
	printf("\n");
}

void
print_to_console_sizet(const size_t * const ptr, const size_t size) {
	for (size_t el=0; el<size; ++el) {
		printf("el[%ld]=%zu, ", el, *(ptr+el));
	}
	printf("\n");
}

void
print_exec_time(const char* fun, const struct timespec start, const struct timespec stop) {
	double exec_time = (stop.tv_sec - start.tv_sec) + 
		(double)(stop.tv_nsec - start.tv_nsec) * (double)ONE_OVER_BILLION;
	printf("%s: %fs\n", fun, exec_time);
}

void
save_to_file(const char const * filename, char* mode, DTYPE * ptr, size_t size) {	
	FILE *f = fopen(filename, mode);

	if (f) {
		// Save the array-like object consisting of int
		for (size_t el=0; el<size; ++el) {
			// DTYPE DEPENDENT
			fprintf(f, "%.2f", (double) *(ptr+el));
			fprintf(f, ",");
		}
	} else {
		printf("Error: File could not be opened.");
	}

	if (ferror(f)) {
		printf("Error while writing to a file.\n");
	}

	if (f != NULL) {
		fclose(f);
	}
}

void
fprint_array_sizet(FILE * const file_handle, const struct array p_array, const size_t size) {
	for(int i=0; i<(int) size; ++i) {
		(void) fprintf(file_handle, "%zu;", p_array.a[i]);
	}
}

void
fprint_ptr_sizet(FILE * const file_handle, const size_t * const p_array, const size_t size) {
	for(int i=0; i<(int) size; ++i) {
		(void) fprintf(file_handle, "%zu;", p_array[i]);
	}
}

void
fprint_array_int(FILE * const file_handle, const int * array, const size_t size) {
	for(int i=0; i<(int) size; ++i) {
		(void) fprintf(file_handle, "%d;", array[i]);
	}
}
