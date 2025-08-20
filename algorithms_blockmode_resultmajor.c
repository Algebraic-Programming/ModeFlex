#include <algorithms.h>
#include <rand_utils.h>
#include <file_utils.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <mkl.h>

void
tvm_blockmode_resultmajor_input_aligned(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {
	size_t last_dim_index = tensor->dim-1;
	size_t * mul = malloc(tensor->dim * sizeof(size_t));
	mul[last_dim_index] = 1;

	size_t block_size = tensor->block_layout[0];
	size_t right_block_size = 1;
	for (size_t i=last_dim_index; i!=0; --i) {
		mul[i-1] = mul[i] * tensor->layout[tensor->layout_perm[i]];
		block_size *= tensor->block_layout[i];
		if (i>mode) {
			right_block_size *= tensor->block_layout[i];
			//printf("RBS=%d i=%d mode=%d \n", right_block_size, i, mode);
		}
	}
	//printf("block_size = %d\n", block_size);
	size_t vector_size = vector->size;
	size_t right_size = mul[mode]; 
	size_t left_size = tensor->lin.size / vector_size / right_size;
	
	size_t RBS = right_block_size;
	size_t vector_block_size = tensor->block_layout[mode];
	size_t VBS = vector_block_size;
	size_t left_block_size = block_size / vector_block_size / right_block_size;
	//printf("LBS=%d, VBS=%d, RBS=%d\n", left_block_size, VBS, RBS);
	size_t global_t = 0;
	size_t t = 0;
	size_t out_offset = 0;
	size_t next = 0;
	size_t left_offset = 1;
	if (mode != 0) {
		left_offset = mul[mode-1];
	}
	// optimization variables
	size_t calc1 = vector_block_size * right_size;
	size_t last_ii = (left_size / left_block_size) * left_block_size;
	size_t last_vv = (vector_size / vector_block_size) * vector_block_size;
	size_t last_jj = (right_size / right_block_size) * right_block_size;
	//printf("calc1=%d, last_ii=%d, last_v=%d, last_j=%d\n", calc1, last_ii, last_vv, last_jj);

	for (size_t ii=0; ii<left_size; ii+=left_block_size) {
		//printf("ii=%d\n", ii);
		if (ii==last_ii) {
			left_block_size = left_size % ii;
		}
		right_block_size = RBS;
		for (size_t jj=0; jj<right_size; jj+=right_block_size) {
			//printf("jj=%d\n", jj);
			if (jj==last_jj) {
				right_block_size = right_size % jj;
			}

			global_t = 0;
			vector_block_size = VBS;
			for (size_t vv=0; vv<vector_size; vv+=vector_block_size) {
				//printf("vv=%d\n", vv);
				if (vv==last_vv) {
					vector_block_size = vector_size % vv;
				}	
				out_offset = ii*right_size+jj;
				for (size_t i=0; i<left_block_size; ++i) {
					t = global_t + (i+ii)*left_offset + jj;
					for (size_t v=0; v<vector_block_size; ++v) {
						for (size_t j=0; j<right_block_size; ++j) {
							//printf("block: vec=%d, out=%d, tensor=%d\n", v+vv, j+out_offset, t+j);
							result_tensor->data[out_offset+j] += 
								tensor->lin.data[next++] * vector->data[v+vv];
						}
						t += right_size;
					}
					out_offset += right_size;
				}
				global_t += calc1; // we won't enter the loop anyway (replaced global_t = vv*right_size (before for jj loop)
			}
		}
	}
	free(mul);
}

void
tvm_blockmode_resultmajor_input_aligned_output_aligned(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {
	size_t last_dim_index = tensor->dim-1;
	size_t * mul = malloc(tensor->dim * sizeof(size_t));
	mul[last_dim_index] = 1;

	size_t block_size = tensor->block_layout[0];
	size_t right_block_size = 1;
	for (size_t i=last_dim_index; i!=0; --i) {
		mul[i-1] = mul[i] * tensor->layout[tensor->layout_perm[i]];
		block_size *= tensor->block_layout[i];
		if (i>mode) {
			right_block_size *= tensor->block_layout[i];
			//printf("RBS=%d i=%d mode=%d \n", right_block_size, i, mode);
		}
	}
	//printf("block_size = %d\n", block_size);
	size_t vector_size = vector->size;
	size_t right_size = mul[mode]; 
	size_t left_size = tensor->lin.size / vector_size / right_size;
	
	size_t RBS = right_block_size;
	size_t vector_block_size = tensor->block_layout[mode];
	size_t VBS = vector_block_size;
	size_t left_block_size = block_size / vector_block_size / right_block_size;
	//printf("LBS=%d, VBS=%d, RBS=%d\n", left_block_size, VBS, RBS);
	size_t global_t = 0;
	//size_t local_offset = 0;
	size_t previous_out_count = 0;
	size_t t = 0;
	//size_t out = 0;
	size_t reset_me_to = 0;
	//size_t global_out = 0;
	size_t out_offset = 0;
	size_t out_count = 0;
	size_t next = 0;
	size_t left_offset = 1;
	if (mode != 0) {
		left_offset = mul[mode-1];
	}
	// optimization variables
	//size_t calc1 = vector_block_size * right_size;
	size_t last_ii = (left_size / left_block_size) * left_block_size;
	size_t last_vv = (vector_size / vector_block_size) * vector_block_size;
	size_t last_jj = (right_size / right_block_size) * right_block_size;
	//printf("calc1=%d, last_ii=%d, last_v=%d, last_j=%d\n", calc1, last_ii, last_vv, last_jj);

	for (size_t ii=0; ii<left_size; ii+=left_block_size) {
		//printf("ii=%d\n", ii);
		if (ii==last_ii) {
			left_block_size = left_size % ii;
		}
		right_block_size = RBS;
		for (size_t jj=0; jj<right_size; jj+=right_block_size) {
			//printf("jj=%d\n", jj);
			if (jj==last_jj) {
				right_block_size = right_size % jj;
			}
			global_t = 0;
			vector_block_size = VBS;

			for (size_t vv=0; vv<vector_size; vv+=vector_block_size) {
				//printf("vv=%d\n", vv);
				if (vv==last_vv) {
					vector_block_size = vector_size % vv;
				}	
				if (vv==0) {
					//printf("we must inc the global_out now\n");
					//global_out = ii * right_size;
					reset_me_to = previous_out_count;

				} else {		
					// reset output everytime we move asize_t vector block part
					//printf("reset occurs\n");
					//out = global_out;
					previous_out_count = reset_me_to;
					//out_count = 0;
				}
				out_offset = ii*right_size+jj;
				for (size_t i=0; i<left_block_size; ++i) {
					t = global_t + (i+ii)*left_offset + jj;
					//out = global_out;
					for (size_t v=0; v<vector_block_size; ++v) {
						//printf("new v\n");
						out_count = previous_out_count;
						for (size_t j=0; j<right_block_size; ++j) {
							//printf("new j\n");
							///printf("block: vec=%d, out=%d, n_out=%d, out_count=%d, tensor=%d\n", v+vv, j+out_offset, global_out, out_count, t+j);
							result_tensor->data[out_offset+j] += 
								tensor->lin.data[next++] * vector->data[v+vv];
							out_count++;
						}
						// v - always resets the output
						t += right_size;
					}
					previous_out_count = out_count;
					out_offset += right_size;
				}
				//global_t += calc1; // we won't enter the loop anyway (replaced global_t = vv*right_size (before for jj loop)
			}
		}
		//printf("YES WE START NEW ii!!!");
		//reset_me_to = 0;
		previous_out_count = 0;
	}
	free(mul);
}

