#include <algorithms.h>
#include <rand_utils.h>
#include <file_utils.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <mkl.h>

//////////////////////////////////////////////////////////////////////////////////////
// MODE BASED BLOCK algorithms

// Input: unfold, Output: unfold (like with other _major algorithms)
void
tvm_blockmode_major(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

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
	//printf("v_s=%d, r_s=%d, l_s=%d\n", vector_size, right_size, left_size);
	size_t RBS = right_block_size;
	size_t vector_block_size = tensor->block_layout[mode];
	size_t VBS = vector_block_size;
	size_t left_block_size = block_size / vector_block_size / right_block_size;
	//printf("LBS=%d, VBS=%d, RBS=%d\n", left_block_size, VBS, RBS);
	size_t global_t = 0;
	size_t t = 0;
	size_t counter = 0;
	size_t out_offset = 0;
	size_t left_offset = 1;
	if (mode != 0) {
		left_offset = mul[mode-1];
	}
	//printf("Left_offset=%d\n", left_offset);
	// optimization variables
	size_t calc1 = vector_block_size * right_size;
	size_t last_ii = (left_size / left_block_size) * left_block_size;
	size_t last_vv = (vector_size / vector_block_size) * vector_block_size;
	size_t last_jj = (right_size / right_block_size) * right_block_size;

	//printf("calc1=%d, last_ii=%d, last_v=%d, last_j=%d\n", calc1, last_ii, last_vv, last_jj);
	for (size_t ii=0; ii<left_size; ii+=left_block_size) {
		//printf("ii=%d\n", ii);
		if (ii==last_ii) {
			//printf("here ii\n");
			left_block_size = left_size % ii;
		}
		global_t = 0;
		vector_block_size = VBS;
		for (size_t vv=0; vv<vector_size; vv+=vector_block_size) {
			//printf("vv=%d\n", vv);
			if (vv==last_vv) {
				//printf("vec_b_size=%d, vec_size=%d\n", vector_block_size, vector_size);
				//printf("here vv\n");
				vector_block_size = vector_size % vv;
			}
			right_block_size = RBS;
			for (size_t jj=0; jj<right_size; jj+=right_block_size) {
				//printf("jj=%d\n", jj);
				if (jj==last_jj) {
					//printf("here jj\n");
					right_block_size = right_size % jj;
				}
				out_offset = ii*right_size+jj;
				//printf("new block...\n");
				for (size_t i=0; i<left_block_size; ++i) {
					//printf("i=%d\n", i);
					t = global_t + (i+ii)*left_offset + jj;
					//printf("t = global_t + (i+ii)*left_offset + jj (%d = %d + (%d+%d)*%d + %d)\n",
							//t, global_t, i, ii, left_offset, jj);
					for (size_t v=0; v<vector_block_size; ++v) {
						//printf("v=%d\n", v);
						for (size_t j=0; j<right_block_size; ++j) {
							//if (counter:wq
							//printf("t=%d, j=%d\n", t,j);
							//printf("out=%d = tensor=%d * vec=%d\n", out_offset+j, t+j, v+vv);
							result_tensor->data[out_offset+j] += 
								tensor->lin.data[t+j] * vector->data[v+vv];
							counter++;	
						}
						t += right_size;
					}
					out_offset += right_size;
				}
			}
			global_t += calc1; // we won't enter the loop anyway (replaced global_t = vv*right_size (before for jj loop)
			//printf("GLOBAL T IS UPDATED\n");
		}
	}
	free(mul);
}

void
tvm_blockmode_major_input_aligned(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

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

	size_t vector_size = vector->size;
	size_t right_size = mul[mode]; 
	size_t left_size = tensor->lin.size / vector_size / right_size;

	// right_block_size is figured out above
	size_t RBS = right_block_size;
	size_t vector_block_size = tensor->block_layout[mode];
	size_t VBS = vector_block_size;
	size_t left_block_size = block_size / vector_block_size / right_block_size;

	size_t global_t = 0;
	size_t t = 0;
	size_t next = 0;

	size_t out_offset = 0;
	size_t left_offset = 1;
	if (mode != 0) {
		left_offset = mul[mode-1];
	}

	// optimization variables
	size_t calc1 = vector_block_size * right_size;
	size_t last_ii = (left_size / left_block_size) * left_block_size;
	size_t last_vv = (vector_size / vector_block_size) * vector_block_size;
	size_t last_jj = (right_size / right_block_size) * right_block_size;

	for (size_t ii=0; ii<left_size; ii+=left_block_size) {
		if (ii==last_ii) {
			left_block_size = left_size % ii;
		}
		global_t = 0;
		vector_block_size = VBS;
		//printf("\nleft_block_size ====== %d\n", left_block_size);
		for (size_t vv=0; vv<vector_size; vv+=vector_block_size) {
			if (vv==last_vv) {
				vector_block_size = vector_size % vv;
			}
			right_block_size = RBS;
			//printf("\nvector_block_size ====== %d\n", vector_block_size);
			// this is the loop which will cover all contiguous blocks (currently in memory) in unfold storage
			for (size_t jj=0; jj<right_size; jj+=right_block_size) {
				if (jj==last_jj) {
					right_block_size = right_size % jj;
				}
				out_offset = ii*right_size+jj;
				//printf("\nright_block_size ====== %d\n", right_block_size);
				//printf("out_offset=%d \n", out_offset);
				for (size_t i=0; i<left_block_size; ++i) {
					t = global_t + (i+ii)*left_offset + jj;
					for (size_t v=0; v<vector_block_size; ++v) {
						for (size_t j=0; j<right_block_size; ++j) {
							//printf("block: vec=%d, out=%d, tensor=%d\n", v+vv, j+out_offset, next);
							//printf("out_offset = %d, jj = %d\n", out_offset, jj);
							//printf("block: vec=%d, out=%d, ten=%d\n", v+vv, out_offset+j, next);
							result_tensor->data[out_offset+j] += 
								tensor->lin.data[next++] * vector->data[v+vv];
						}
						t += right_size;
					}
					out_offset += right_size;
				}
			}
			global_t += calc1; // we won't enter the loop anyway (replaced global_t = vv*right_size (before for jj loop)
		}
	}
	
	//printf("SIEMA \n");
	free(mul);
}

void
tvm_blockmode_major_BLAS_input_aligned_output_aligned(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

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
	//printf("v_size=%d, r_size=%d, l_size=%d\n", vector_size, right_size, left_size);
	
	size_t RBS = right_block_size;
	size_t vector_block_size = tensor->block_layout[mode];
	size_t VBS = vector_block_size;
	size_t left_block_size = block_size / vector_block_size / right_block_size;
	//p//rsize_tf("LBS=%d, VBS=%d, RBS=%d\n", left_block_size, VBS, RBS);
	
	//size_t global_t = 0;
	//size_t t = 0;
	size_t next = 0;
	//size_t new_out = 0;
	size_t temp_new_out = 0;
	size_t super_temp_new_out = 0;

	//size_t out_offset = 0;
	//size_t left_offset = 1;
	//if (mode != 0) {
		//left_offset = mul[mode-1];
	//}

	// optimization variables
	//size_t calc1 = vector_block_size * right_size;
	size_t last_ii = (left_size / left_block_size) * left_block_size;
	size_t last_vv = (vector_size / vector_block_size) * vector_block_size;
	size_t last_jj = (right_size / right_block_size) * right_block_size;
	//printf("calc1=%d, last_ii=%d, last_v=%d, last_j=%d\n", calc1, last_ii, last_vv, last_jj);

	for (size_t ii=0; ii<left_size; ii+=left_block_size) {
		//printf("ii=%d\n", ii);
		if (ii==last_ii) {
			//if (ii==0) printf("LOL\n");
			left_block_size = left_size % ii;
		}
		//global_t = 0;
		vector_block_size = VBS;

		for (size_t vv=0; vv<vector_size; vv+=vector_block_size) {
			//printf("vv=%d\n", vv);
			if (vv==last_vv) {
				//if (vv==0) printf("LOL\n");
				vector_block_size = vector_size % vv;
			}
			right_block_size = RBS;

			//printf("reset occurs\n");
			for (size_t jj=0; jj<right_size; jj+=right_block_size) {
				if (jj==0) {
					//printf("RESET OCCURS!\n");
					//new_out = ii*right_size;
					temp_new_out = ii*right_size;//new_out;

				}
				//printf("jj=%d\n", jj);
				if (jj==last_jj) {
					//if (jj==0) printf("LOL\n");
					right_block_size = right_size % jj;
				}
				//out_offset = ii*right_size+jj;
				//printf("RESET OCCURS\n");
				for (size_t i=0; i<left_block_size; ++i) {
					//size_t reset_new_out = new_out + right_block_size;
					//t = global_t + (i+ii)*left_offset + jj;
					//printf("out_offset=%d\n", out_offset);
					//printf("global_t=%d\n", global_t);
					//printf("(i+ii)=%d\n", i+ii);

					//for (size_t v=0; v<vector_block_size; ++v) {
						super_temp_new_out = temp_new_out;

						//size_t //temp_new_out = new_out + i*right_block_size;
						// vector change - reset the temp out
						//temp_new_out = jj;

						// Only this loop is contiguous (???)

						//for (size_t j=0; j<right_block_size; ++j) {
							const double alpha = 1;
							const double beta = 1;
							const MKL_INT incx = 1;
							const MKL_INT incy = 1;

							//const MKL_INT n = result_tensor->size;
							const MKL_INT n = right_block_size * vector_block_size;
							// IMHO: this should bve like htis, but then i tbreaks something lese
							// if its just rihgt_block_size it works in more cases

							// wit n equal this, its breaks earlier but I think this is the size of matrix , so... 


							//const MKL_INT mode = vector->size;
							const MKL_INT mode = vector_block_size;

							printf("Size of matrix: %d BY %d \n", n, mode);
							printf("Run with tensor + next(%zu), result + super_temp(%zu), vector + vv(%zu)\n", 
									next, super_temp_new_out, vv);
							cblas_dgemv(
								CblasRowMajor, // const CBLAS_LAYOUT
								CblasNoTrans, // const CBLAS_TRANSPOSE
								n, mode, // const MKL_size_t (s)
								alpha, // const double
								tensor->lin.data + next, mode, // const double*, const MKL_size_t
								vector->data + vv, incx, // const double*, const MKL_size_t
								beta, // const float
								result_tensor->data + super_temp_new_out, incy); // const double*, const MKL_size_t
							
							printf("result now is\n");
							print_to_console(result_tensor->data, result_tensor->size);

							next += right_block_size * vector_block_size;

							

							//result_tensor->data[super_temp_new_out++] +=
								//tensor->lin.data[next++] * vector->data[v+vv];
						//}
						//t += right_size;

					//}

					//out_offset += right_size;
					//printf("incrementing reste_new_out\n");
						temp_new_out += right_block_size;
						printf("NEXT...\n");
					//printf("temp_new_out = %d\n", temp_new_out);
				}
			}
			//global_t += calc1; // we won't enter the loop anyway (replaced global_t = vv*right_size (before for jj loop)
		}
		//printf("INCREASE THE II- that's when reset occurs\n");
	}	
	free(mul);
	//printf("SIEMA \n");
}

void
tvm_blockmode_major_input_aligned_output_aligned(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

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
	//printf("v_size=%d, r_size=%d, l_size=%d\n", vector_size, right_size, left_size);
	
	size_t RBS = right_block_size;
	size_t vector_block_size = tensor->block_layout[mode];
	size_t VBS = vector_block_size;
	size_t left_block_size = block_size / vector_block_size / right_block_size;
	//p//rsize_tf("LBS=%d, VBS=%d, RBS=%d\n", left_block_size, VBS, RBS);
	
	//size_t global_t = 0;
	//size_t t = 0;
	size_t next = 0;
	//size_t new_out = 0;
	size_t temp_new_out = 0;
	size_t super_temp_new_out = 0;

	//size_t out_offset = 0;
	//size_t left_offset = 1;
	//if (mode != 0) {
		//left_offset = mul[mode-1];
	//}

	// optimization variables
	//size_t calc1 = vector_block_size * right_size;
	size_t last_ii = (left_size / left_block_size) * left_block_size;
	size_t last_vv = (vector_size / vector_block_size) * vector_block_size;
	size_t last_jj = (right_size / right_block_size) * right_block_size;
	//printf("calc1=%d, last_ii=%d, last_v=%d, last_j=%d\n", calc1, last_ii, last_vv, last_jj);

	for (size_t ii=0; ii<left_size; ii+=left_block_size) {
		//printf("ii=%d\n", ii);
		if (ii==last_ii) {
			//if (ii==0) printf("LOL\n");
			left_block_size = left_size % ii;
		}
		//global_t = 0;
		vector_block_size = VBS;

		for (size_t vv=0; vv<vector_size; vv+=vector_block_size) {
			//printf("vv=%d\n", vv);
			if (vv==last_vv) {
				//if (vv==0) printf("LOL\n");
				vector_block_size = vector_size % vv;
			}
			right_block_size = RBS;

			//printf("reset occurs\n");
			for (size_t jj=0; jj<right_size; jj+=right_block_size) {
				if (jj==0) {
					//printf("RESET OCCURS!\n");
					//new_out = ii*right_size;
					temp_new_out = ii*right_size;//new_out;

				}
				//printf("jj=%d\n", jj);
				if (jj==last_jj) {
					//if (jj==0) printf("LOL\n");
					right_block_size = right_size % jj;
				}
				//out_offset = ii*right_size+jj;
				//printf("RESET OCCURS\n");
				for (size_t i=0; i<left_block_size; ++i) {
					//size_t reset_new_out = new_out + right_block_size;
					//t = global_t + (i+ii)*left_offset + jj;
					//printf("out_offset=%d\n", out_offset);
					//printf("global_t=%d\n", global_t);
					//printf("(i+ii)=%d\n", i+ii);
					for (size_t v=0; v<vector_block_size; ++v) {
						super_temp_new_out = temp_new_out;
						//size_t //temp_new_out = new_out + i*right_block_size;
						// vector change - reset the temp out
						//temp_new_out = jj;
						for (size_t j=0; j<right_block_size; ++j) {
							//printf("block: vec=%d, ten=%d, out=%d\n", 
								//v+vv, next, super_temp_new_out);
							//next++;
							//super_temp_new_out++;
							//result_tensor->data[out_offset+j] += 

							printf("Run with tensor + next(%zu), result + super_temp(%zu), vector + v + vv(%zu)\n",
									next, super_temp_new_out, v+vv);

							//printf("block: vec=%d, out=%d, ten=%d\n", v+vv, super_temp_new_out, next);
							result_tensor->data[super_temp_new_out++] +=
								tensor->lin.data[next++] * vector->data[v+vv];

						}

						print_to_console(result_tensor->data, result_tensor->size);

						//t += right_size;
					}
					//out_offset += right_size;
					//printf("incrementing reste_new_out\n");
					temp_new_out += right_block_size;
					//printf("temp_new_out = %d\n", temp_new_out);
				}
			}
			//global_t += calc1; // we won't enter the loop anyway (replaced global_t = vv*right_size (before for jj loop)
		}
		//printf("INCREASE THE II- that's when reset occurs\n");
	}	
	free(mul);
	//printf("SIEMA \n");
}

void
tvm_blockmode_major_input_aligned_output_aligned_2(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

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
		}
	}
	size_t vector_size = vector->size;
	size_t right_size = mul[mode]; 
	size_t left_size = tensor->lin.size / vector_size / right_size;
	
	size_t RBS = right_block_size;
	size_t vector_block_size = tensor->block_layout[mode];
	size_t VBS = vector_block_size;
	size_t left_block_size = block_size / vector_block_size / right_block_size;
	size_t next = 0;
	size_t temp_new_out = 0;
	size_t super_temp_new_out = 0;

	size_t last_ii = (left_size / left_block_size) * left_block_size;
	size_t last_vv = (vector_size / vector_block_size) * vector_block_size;
	size_t last_jj = (right_size / right_block_size) * right_block_size;

	for (size_t ii=0; ii<left_size; ii+=left_block_size) {
		if (ii==last_ii) {
			left_block_size = left_size % ii;
		}
		vector_block_size = VBS;
		for (size_t vv=0; vv<vector_size; vv+=vector_block_size) {
			if (vv==last_vv) {
				vector_block_size = vector_size % vv;
			}
			right_block_size = RBS;
			temp_new_out = ii*right_size;
			for (size_t jj=0; jj<right_size; jj+=right_block_size) {
				if (jj==last_jj) {
					right_block_size = right_size % jj;
				}
				for (size_t i=0; i<left_block_size; ++i) {
					for (size_t v=0; v<vector_block_size; ++v) {
						super_temp_new_out = temp_new_out;
						for (size_t j=0; j<right_block_size; ++j) {
							//printf("block: vec=%d, out=%d, ten=%d\n", v+vv, super_temp_new_out, next);
							result_tensor->data[super_temp_new_out++] +=
								tensor->lin.data[next++] * vector->data[vv+v];
						}
					}
					temp_new_out += right_block_size;
				}
			}
		}
	}	
	free(mul);
}

//////////////////////////////////////////////////////////////////////////
// Mode based blocking: result major variant

void
tvm_blockmode_resultmajor(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

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
							//printf("block: vec=%d, out=%d\n", v+vv, j+out_offset);
							result_tensor->data[out_offset+j] += 
								tensor->lin.data[t+j] * vector->data[v+vv];
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

