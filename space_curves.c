#include<stdio.h>
#include<stdlib.h>
#include<file_utils.h>
#include<structures.h>

int
morton_inc_struct_2(size_t * const counters, const size_t * const thresholds, const size_t init_offset, const size_t mode, const size_t levels) {

	int mask = 1;
	size_t level = 0;
	int inc_game = 1;
	size_t offset = init_offset;
	while (inc_game) {
	if (counters[offset] & mask) {
		counters[offset] &= ~mask;
		if (offset == 0) {
			mask <<= 1;
			level += 1;
			offset = init_offset;
		} else {
			offset -= 1;
		}
	} else {
		if ((counters[offset] | mask) >= thresholds[offset]) {
			if (offset == 0) {
				mask <<= 1;
				level += 1;
				offset = init_offset;
			} else if (offset > levels) {
				return -1;
			} else {
				offset -= 1;
			}
		} else {
			inc_game = 0;
		}
	}
	}
	counters[offset] |= mask;
	if (offset == mode) {
		return level;
	} else {
		return 0;
	}
	
}

void
morton_inc(size_t * counters, size_t * thresholds, size_t init_offset) {

	// try an idea: just do % with appropriate multiple of 2

	// we do all in one loop!
	//int last_set_to_zero = -1;
	int mask = 1;
	int inc_game = 1;
	
	// start with highest dimension
	size_t offset = init_offset;

	// if lowest dimension & mask is 0 means we found already the bit 
	// lowest dimension = itself | mask
	// otherwise while
	while (inc_game) {
	if (counters[offset] & mask) {
		// we already know that the lowest bit is not 0
		// modify that counter (we must reset that 1)
		counters[offset] &= ~mask;
		// mul of 2 should be zero at this point
		// but we are the last dimension
		if (offset == 0) {
			// go to the first dimension but mask<<=1
			mask <<= 1;
			offset = init_offset;
		} else {
			// go to the next dimension!
			offset -= 1;
		}
	} else {
		// we want to increment this dimension but it could be out of bounds
		// correct me: but I think equality is okay
		if ((counters[offset] | mask) >= thresholds[offset]) {
			// do the same as above (go to next dim)
			if (offset == 0) {
				mask <<= 1;
				offset = init_offset;
			} else {
				offset -= 1;
			}
			// all we need to do is prevent from the loop from stopping
			// (this will not add the mask -> which will make us out of bounds)
		} else {
			// natural termination
			inc_game = 0;
		}
	}
	}
	// we found the 0 to be switched to 1
	counters[offset] |= mask;
}

void
morton_inc_struct(size_t * const counters, const size_t * const thresholds, const size_t init_offset, struct morton_data * const morton_meta) {

	// try an idea: just do % with appropriate multiple of 2
	// we do all in one loop!
	//int last_set_to_zero = -1;
	int mask = 1;
	size_t level = 0;
	int inc_game = 1;
	
	// start with highest dimension
	size_t offset = init_offset;

	// if lowest dimension & mask is 0 means we found already the bit 
	// lowest dimension = itself | mask
	// otherwise while
	while (inc_game) {
	if (counters[offset] & mask) {
		// we already know that the lowest bit is not 0
		// modify that counter (we must reset that 1)
		//printf("WE PLAID THE INC GAME\n");
		counters[offset] &= ~mask;
		// mul of 2 should be zero at this point
		// but we are the last dimension
		if (offset == 0) {
			// go to the first dimension but mask<<=1
			mask <<= 1;
			level += 1;
			offset = init_offset;
		} else {
			// go to the next dimension!
			offset -= 1;
		}
	} else {
		// we want to increment this dimension but it could be out of bounds
		// correct me: but I think equality is okay
		//printf("counters[offset] | mask) = %d\n", counters[offset] | mask);
		//printf("thresholds[offset] = %d\n", thresholds[offset]);
		if ((counters[offset] | mask) >= thresholds[offset]) {
			//printf("WE ARE OUT OF BOUNDS\n");
			// do the same as above (go to next dim)
			if (offset == 0) {
				mask <<= 1;
				level += 1;
				offset = init_offset;
			} else {
				offset -= 1;
			}
			// all we need to do is prevent from the loop from stopping
			// (this will not add the mask -> which will make us out of bounds)
		} else {
			// natural termination
			inc_game = 0;
		}
	}
	}
	// we found the 0 to be switched to 1
	counters[offset] |= mask;

	morton_meta->dim = offset;
	morton_meta->level = level;
	//morton_meta->mask = mask;
}

size_t
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
