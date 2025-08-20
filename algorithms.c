#include <algorithms.h>
#include <rand_utils.h>
#include <file_utils.h>
#include <string.h>
#include <stdlib.h>
// #include <blis.h>
#include <math.h>
#include <mkl.h>
#include <gen_utils.h>
#include <smmintrin.h>
#include <immintrin.h>
#include <libxsmm.h>

/*
void
tvm_taco(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result, const size_t mode) {

	int A1_size, A2_size, A3_size, A4_size, A5_size, A6_size, A7_size, A8_size, A9_size, A10_size, y1_size, y2_size, y3_size, y4_size, y5_size, y6_size, y7_size, y8_size, y9_size;
	size_t dim = tensor->dim;

  double* restrict y_vals = (double*)(result->data);
  double* restrict A_vals = (double*)(tensor->lin.data);
  double* restrict x_vals = (double*)(vector->data);

  for (size_t d=0; d<dim; ++d) {
    switch(d) {
      case 0:
      	A1_size = tensor->layout[0];
        break;
      case 1:
        A2_size = tensor->layout[1];
        break;
      case 2:
        A3_size = tensor->layout[2];
        break;
      case 3:
        A4_size = tensor->layout[3];
        break;
      case 4:
        A5_size = tensor->layout[4];
        break;
      case 5:
        A6_size = tensor->layout[5];
        break;
      case 6:
        A7_size = tensor->layout[6];
        break;
      case 7:
        A8_size = tensor->layout[7];
        break;
      case 8:
        A9_size = tensor->layout[8];
        break;
      case 9:
        A10_size = tensor->layout[9];
        break;
    }
  }

  int res = -1;
  for (size_t d=0; d<dim; ++d) {
  	++res;
    if (d != mode) {
      switch(res) {
        case 0:
      	  y1_size = tensor->layout[d];
          break;
        case 1:
      	  y2_size = tensor->layout[d];
          break;
        case 2:
      	  y3_size = tensor->layout[d];
          break;
        case 3:
      	  y4_size = tensor->layout[d];
          break;
        case 4:
      	  y5_size = tensor->layout[d];
          break;
        case 5:
      	  y6_size = tensor->layout[d];
          break;
        case 6:
      	  y7_size = tensor->layout[d];
          break;
        case 7:
      	  y8_size = tensor->layout[d];
          break;
        case 8:
      	  y9_size = tensor->layout[d];
          break;
      }
    } else {
      --res;
    }
  }

  if (dim == 2) {
  	
	  if (mode == 0) {
	    for (size_t iA = 0; iA < A1_size; iA++) {
	      double ti = x_vals[iA];
	      for (size_t kA = 0; kA < A2_size; kA++) {
	        size_t pA2 = (iA * A2_size) + kA;
	        y_vals[kA] = y_vals[kA] + (A_vals[pA2] * ti);
	      }
	    }
	  } else {
	    for (size_t jA = 0; jA < A1_size; jA++) {
	      double ti = 0;
	      for (size_t iA = 0; iA < A2_size; iA++) {
	        size_t pA2 = (jA * A2_size) + iA;
	        ti += A_vals[pA2] * x_vals[iA];
	      }
	      y_vals[jA] = ti;
	    }
	  }

	} else if (dim == 3) {
	  if (mode == 0) {
			  for (size_t iA = 0; iA < A1_size; iA++) {
			    double ti = x_vals[iA];
			    for (size_t jA = 0; jA < A2_size; jA++) {
			      size_t pA2 = (iA * A2_size) + jA;
			      double tj = ti;
			      for (size_t kA = 0; kA < A3_size; kA++) {
			        size_t pA3 = (pA2 * A3_size) + kA;
			        size_t py2 = (jA * y2_size) + kA;
			        y_vals[py2] = y_vals[py2] + (A_vals[pA3] * tj);
			      }
			    }
			  }
	  } else if (mode == 1) {
		  for (size_t jA = 0; jA < A1_size; jA++) {
		    for (size_t iA = 0; iA < A2_size; iA++) {
		      size_t pA2 = (jA * A2_size) + iA;
		      double ti = x_vals[iA];
		      for (size_t kA = 0; kA < A3_size; kA++) {
		        size_t pA3 = (pA2 * A3_size) + kA;
		        size_t py2 = (jA * y2_size) + kA;
		        y_vals[py2] = y_vals[py2] + (A_vals[pA3] * ti);
		      }
		    }
		  }
	  } else {
		  for (size_t jA = 0; jA < A1_size; jA++) {
		    for (size_t iA = 0; iA < A2_size; iA++) {
		      size_t pA2 = (jA * A2_size) + iA;
		      size_t py2 = (jA * y2_size) + iA;
		      double tk = 0;
		      for (size_t kA = 0; kA < A3_size; kA++) {
		        size_t pA3 = (pA2 * A3_size) + kA;
		        tk += A_vals[pA3] * x_vals[kA];
		      }
		      y_vals[py2] = tk;
		    }
		  }
	  }

	} else if (dim == 4) {
		if (mode == 0) {

		  for (size_t iA = 0; iA < A1_size; iA++) {
		    double ti = x_vals[iA];
		    for (size_t jA = 0; jA < A2_size; jA++) {
		      size_t pA2 = (iA * A2_size) + jA;
		      double tj = ti;
		      for (size_t kA = 0; kA < A3_size; kA++) {
		        size_t pA3 = (pA2 * A3_size) + kA;
		        size_t py2 = (jA * y2_size) + kA;
		        double tk = tj;
		        for (size_t lA = 0; lA < A4_size; lA++) {
		          size_t pA4 = (pA3 * A4_size) + lA;
		          size_t py3 = (py2 * y3_size) + lA;
		          y_vals[py3] = y_vals[py3] + (A_vals[pA4] * tk);
		        }
		      }
		    }
		  }			

		} else if (mode == 1) {

  for (size_t iA = 0; iA < A1_size; iA++) {
    for (size_t jA = 0; jA < A2_size; jA++) {
      size_t pA2 = (iA * A2_size) + jA;
      double tj = x_vals[jA];
      for (size_t kA = 0; kA < A3_size; kA++) {
        size_t pA3 = (pA2 * A3_size) + kA;
        size_t py2 = (iA * y2_size) + kA;
        double tk = tj;
        for (size_t lA = 0; lA < A4_size; lA++) {
          size_t pA4 = (pA3 * A4_size) + lA;
          size_t py3 = (py2 * y3_size) + lA;
          y_vals[py3] = y_vals[py3] + (A_vals[pA4] * tk);
        }
      }
    }
  }

		} else if (mode == 2) {

  for (size_t iA = 0; iA < A1_size; iA++) {
    for (size_t jA = 0; jA < A2_size; jA++) {
      size_t pA2 = (iA * A2_size) + jA;
      size_t py2 = (iA * y2_size) + jA;
      for (size_t kA = 0; kA < A3_size; kA++) {
        size_t pA3 = (pA2 * A3_size) + kA;
        double tk = x_vals[kA];
        for (size_t lA = 0; lA < A4_size; lA++) {
          size_t pA4 = (pA3 * A4_size) + lA;
          size_t py3 = (py2 * y3_size) + lA;
          y_vals[py3] = y_vals[py3] + (A_vals[pA4] * tk);
        }
      }
    }
  }

		} else if (mode == 3) {

  for (size_t iA = 0; iA < A1_size; iA++) {
    for (size_t jA = 0; jA < A2_size; jA++) {
      size_t pA2 = (iA * A2_size) + jA;
      size_t py2 = (iA * y2_size) + jA;
      for (size_t kA = 0; kA < A3_size; kA++) {
        size_t pA3 = (pA2 * A3_size) + kA;
        size_t py3 = (py2 * y3_size) + kA;
        double tl = 0;
        for (size_t lA = 0; lA < A4_size; lA++) {
          size_t pA4 = (pA3 * A4_size) + lA;
          tl += A_vals[pA4] * x_vals[lA];
        }
        y_vals[py3] = tl;
      }
    }
  }

		}

	} else if (dim == 5) {
		if (mode == 0) {
		  for (size_t iA = 0; iA < A1_size; iA++) {
		    double ti = x_vals[iA];
		    for (size_t jA = 0; jA < A2_size; jA++) {
		      size_t pA2 = (iA * A2_size) + jA;
		      double tj = ti;
		      for (size_t kA = 0; kA < A3_size; kA++) {
		        size_t pA3 = (pA2 * A3_size) + kA;
		        size_t py2 = (jA * y2_size) + kA;
		        double tk = tj;
		        for (size_t lA = 0; lA < A4_size; lA++) {
		          size_t pA4 = (pA3 * A4_size) + lA;
		          size_t py3 = (py2 * y3_size) + lA;
		          double tl = tk;
		          for (size_t mA = 0; mA < A5_size; mA++) {
		            size_t pA5 = (pA4 * A5_size) + mA;
		            size_t py4 = (py3 * y4_size) + mA;
		            y_vals[py4] = y_vals[py4] + (A_vals[pA5] * tl);
		          }
		        }
		      }
		    }
		  }
		} else if (mode == 1) {

  for (size_t iA = 0; iA < A1_size; iA++) {
    for (size_t jA = 0; jA < A2_size; jA++) {
      size_t pA2 = (iA * A2_size) + jA;
      double tj = x_vals[jA];
      for (size_t kA = 0; kA < A3_size; kA++) {
        size_t pA3 = (pA2 * A3_size) + kA;
        size_t py2 = (iA * y2_size) + kA;
        double tk = tj;
        for (size_t lA = 0; lA < A4_size; lA++) {
          size_t pA4 = (pA3 * A4_size) + lA;
          size_t py3 = (py2 * y3_size) + lA;
          double tl = tk;
          for (size_t mA = 0; mA < A5_size; mA++) {
            size_t pA5 = (pA4 * A5_size) + mA;
            size_t py4 = (py3 * y4_size) + mA;
            y_vals[py4] = y_vals[py4] + (A_vals[pA5] * tl);
          }
        }
      }
    }
  }

		} else if (mode == 2) {

  for (size_t iA = 0; iA < A1_size; iA++) {
    for (size_t jA = 0; jA < A2_size; jA++) {
      size_t pA2 = (iA * A2_size) + jA;
      size_t py2 = (iA * y2_size) + jA;
      for (size_t kA = 0; kA < A3_size; kA++) {
        size_t pA3 = (pA2 * A3_size) + kA;
        double tk = x_vals[kA];
        for (size_t lA = 0; lA < A4_size; lA++) {
          size_t pA4 = (pA3 * A4_size) + lA;
          size_t py3 = (py2 * y3_size) + lA;
          double tl = tk;
          for (size_t mA = 0; mA < A5_size; mA++) {
            size_t pA5 = (pA4 * A5_size) + mA;
            size_t py4 = (py3 * y4_size) + mA;
            y_vals[py4] = y_vals[py4] + (A_vals[pA5] * tl);
          }
        }
      }
    }
  }

		} else if (mode == 3) {

  for (size_t iA = 0; iA < A1_size; iA++) {
    for (size_t jA = 0; jA < A2_size; jA++) {
      size_t pA2 = (iA * A2_size) + jA;
      size_t py2 = (iA * y2_size) + jA;
      for (size_t kA = 0; kA < A3_size; kA++) {
        size_t pA3 = (pA2 * A3_size) + kA;
        size_t py3 = (py2 * y3_size) + kA;
        for (size_t lA = 0; lA < A4_size; lA++) {
          size_t pA4 = (pA3 * A4_size) + lA;
          double tl = x_vals[lA];
          for (size_t mA = 0; mA < A5_size; mA++) {
            size_t pA5 = (pA4 * A5_size) + mA;
            size_t py4 = (py3 * y4_size) + mA;
            y_vals[py4] = y_vals[py4] + (A_vals[pA5] * tl);
          }
        }
      }
    }
  }


		} else if (mode == 4) {

  for (size_t iA = 0; iA < A1_size; iA++) {
    for (size_t jA = 0; jA < A2_size; jA++) {
      size_t pA2 = (iA * A2_size) + jA;
      size_t py2 = (iA * y2_size) + jA;
      for (size_t kA = 0; kA < A3_size; kA++) {
        size_t pA3 = (pA2 * A3_size) + kA;
        size_t py3 = (py2 * y3_size) + kA;
        for (size_t lA = 0; lA < A4_size; lA++) {
          size_t pA4 = (pA3 * A4_size) + lA;
          size_t py4 = (py3 * y4_size) + lA;
          double tm = 0;
          for (size_t mA = 0; mA < A5_size; mA++) {
            size_t pA5 = (pA4 * A5_size) + mA;
            tm += A_vals[pA5] * x_vals[mA];
          }
          y_vals[py4] = tm;
        }
      }
    }
  }

		}

	} else if (dim == 6) {
		if (mode == 0) {
		  for (size_t iA = 0; iA < A1_size; iA++) {
		    double ti = x_vals[iA];
		    for (size_t jA = 0; jA < A2_size; jA++) {
		      size_t pA2 = (iA * A2_size) + jA;
		      double tj = ti;
		      for (size_t kA = 0; kA < A3_size; kA++) {
		        size_t pA3 = (pA2 * A3_size) + kA;
		        size_t py2 = (jA * y2_size) + kA;
		        double tk = tj;
		        for (size_t lA = 0; lA < A4_size; lA++) {
		          size_t pA4 = (pA3 * A4_size) + lA;
		          size_t py3 = (py2 * y3_size) + lA;
		          double tl = tk;
		          for (size_t mA = 0; mA < A5_size; mA++) {
		            size_t pA5 = (pA4 * A5_size) + mA;
		            size_t py4 = (py3 * y4_size) + mA;
		            double tm = tl;
		            for (size_t oA = 0; oA < A6_size; oA++) {
		              size_t pA6 = (pA5 * A6_size) + oA;
		              size_t py5 = (py4 * y5_size) + oA;
		              y_vals[py5] = y_vals[py5] + (A_vals[pA6] * tm);
		            }
		          }
		        }
		      }
		    }
		  }	
		} else if (mode == 1) {

  for (size_t iA = 0; iA < A1_size; iA++) {
    for (size_t jA = 0; jA < A2_size; jA++) {
      size_t pA2 = (iA * A2_size) + jA;
      double tj = x_vals[jA];
      for (size_t kA = 0; kA < A3_size; kA++) {
        size_t pA3 = (pA2 * A3_size) + kA;
        size_t py2 = (iA * y2_size) + kA;
        double tk = tj;
        for (size_t lA = 0; lA < A4_size; lA++) {
          size_t pA4 = (pA3 * A4_size) + lA;
          size_t py3 = (py2 * y3_size) + lA;
          double tl = tk;
          for (size_t mA = 0; mA < A5_size; mA++) {
            size_t pA5 = (pA4 * A5_size) + mA;
            size_t py4 = (py3 * y4_size) + mA;
            double tm = tl;
            for (size_t oA = 0; oA < A6_size; oA++) {
              size_t pA6 = (pA5 * A6_size) + oA;
              size_t py5 = (py4 * y5_size) + oA;
              y_vals[py5] = y_vals[py5] + (A_vals[pA6] * tm);
            }
          }
        }
      }
    }
  }

		} else if (mode == 2) {

  for (size_t iA = 0; iA < A1_size; iA++) {
    for (size_t jA = 0; jA < A2_size; jA++) {
      size_t pA2 = (iA * A2_size) + jA;
      size_t py2 = (iA * y2_size) + jA;
      for (size_t kA = 0; kA < A3_size; kA++) {
        size_t pA3 = (pA2 * A3_size) + kA;
        double tk = x_vals[kA];
        for (size_t lA = 0; lA < A4_size; lA++) {
          size_t pA4 = (pA3 * A4_size) + lA;
          size_t py3 = (py2 * y3_size) + lA;
          double tl = tk;
          for (size_t mA = 0; mA < A5_size; mA++) {
            size_t pA5 = (pA4 * A5_size) + mA;
            size_t py4 = (py3 * y4_size) + mA;
            double tm = tl;
            for (size_t oA = 0; oA < A6_size; oA++) {
              size_t pA6 = (pA5 * A6_size) + oA;
              size_t py5 = (py4 * y5_size) + oA;
              y_vals[py5] = y_vals[py5] + (A_vals[pA6] * tm);
            }
          }
        }
      }
    }
  }


		} else if (mode == 3) {

  for (size_t iA = 0; iA < A1_size; iA++) {
    for (size_t jA = 0; jA < A2_size; jA++) {
      size_t pA2 = (iA * A2_size) + jA;
      size_t py2 = (iA * y2_size) + jA;
      for (size_t kA = 0; kA < A3_size; kA++) {
        size_t pA3 = (pA2 * A3_size) + kA;
        size_t py3 = (py2 * y3_size) + kA;
        for (size_t lA = 0; lA < A4_size; lA++) {
          size_t pA4 = (pA3 * A4_size) + lA;
          double tl = x_vals[lA];
          for (size_t mA = 0; mA < A5_size; mA++) {
            size_t pA5 = (pA4 * A5_size) + mA;
            size_t py4 = (py3 * y4_size) + mA;
            double tm = tl;
            for (size_t oA = 0; oA < A6_size; oA++) {
              size_t pA6 = (pA5 * A6_size) + oA;
              size_t py5 = (py4 * y5_size) + oA;
              y_vals[py5] = y_vals[py5] + (A_vals[pA6] * tm);
            }
          }
        }
      }
    }
  }

		} else if (mode == 4) {

 for (size_t iA = 0; iA < A1_size; iA++) {
    for (size_t jA = 0; jA < A2_size; jA++) {
      size_t pA2 = (iA * A2_size) + jA;
      size_t py2 = (iA * y2_size) + jA;
      for (size_t kA = 0; kA < A3_size; kA++) {
        size_t pA3 = (pA2 * A3_size) + kA;
        size_t py3 = (py2 * y3_size) + kA;
        for (size_t lA = 0; lA < A4_size; lA++) {
          size_t pA4 = (pA3 * A4_size) + lA;
          size_t py4 = (py3 * y4_size) + lA;
          for (size_t mA = 0; mA < A5_size; mA++) {
            size_t pA5 = (pA4 * A5_size) + mA;
            double tm = x_vals[mA];
            for (size_t oA = 0; oA < A6_size; oA++) {
              size_t pA6 = (pA5 * A6_size) + oA;
              size_t py5 = (py4 * y5_size) + oA;
              y_vals[py5] = y_vals[py5] + (A_vals[pA6] * tm);
            }
          }
        }
      }
    }
  }

		} else if (mode == 5) {

  for (size_t iA = 0; iA < A1_size; iA++) {
    for (size_t jA = 0; jA < A2_size; jA++) {
      size_t pA2 = (iA * A2_size) + jA;
      size_t py2 = (iA * y2_size) + jA;
      for (size_t kA = 0; kA < A3_size; kA++) {
        size_t pA3 = (pA2 * A3_size) + kA;
        size_t py3 = (py2 * y3_size) + kA;
        for (size_t lA = 0; lA < A4_size; lA++) {
          size_t pA4 = (pA3 * A4_size) + lA;
          size_t py4 = (py3 * y4_size) + lA;
          for (size_t mA = 0; mA < A5_size; mA++) {
            size_t pA5 = (pA4 * A5_size) + mA;
            size_t py5 = (py4 * y5_size) + mA;
            double to = 0;
            for (size_t oA = 0; oA < A6_size; oA++) {
              size_t pA6 = (pA5 * A6_size) + oA;
              to += A_vals[pA6] * x_vals[oA];
            }
            y_vals[py5] = to;
          }
        }
      }
    }
  }


		}
	} else if (dim == 7) {
		if (mode == 0) {
		  for (size_t iA = 0; iA < A1_size; iA++) {
		    double ti = x_vals[iA];
		    for (size_t jA = 0; jA < A2_size; jA++) {
		      size_t pA2 = (iA * A2_size) + jA;
		      double tj = ti;
		      for (size_t kA = 0; kA < A3_size; kA++) {
		        size_t pA3 = (pA2 * A3_size) + kA;
		        size_t py2 = (jA * y2_size) + kA;
		        double tk = tj;
		        for (size_t lA = 0; lA < A4_size; lA++) {
		          size_t pA4 = (pA3 * A4_size) + lA;
		          size_t py3 = (py2 * y3_size) + lA;
		          double tl = tk;
		          for (size_t mA = 0; mA < A5_size; mA++) {
		            size_t pA5 = (pA4 * A5_size) + mA;
		            size_t py4 = (py3 * y4_size) + mA;
		            double tm = tl;
		            for (size_t oA = 0; oA < A6_size; oA++) {
		              size_t pA6 = (pA5 * A6_size) + oA;
		              size_t py5 = (py4 * y5_size) + oA;
		              double to = tm;
		              for (size_t pA = 0; pA < A7_size; pA++) {
		                size_t pA7 = (pA6 * A7_size) + pA;
		                size_t py6 = (py5 * y6_size) + pA;
		                y_vals[py6] = y_vals[py6] + (A_vals[pA7] * to);
		              }
		            }
		          }
		        }
		      }
		    }
		  }
		} else if (mode == 1) {

  for (size_t iA = 0; iA < A1_size; iA++) {
    for (size_t jA = 0; jA < A2_size; jA++) {
      size_t pA2 = (iA * A2_size) + jA;
      double tj = x_vals[jA];
      for (size_t kA = 0; kA < A3_size; kA++) {
        size_t pA3 = (pA2 * A3_size) + kA;
        size_t py2 = (iA * y2_size) + kA;
        double tk = tj;
        for (size_t lA = 0; lA < A4_size; lA++) {
          size_t pA4 = (pA3 * A4_size) + lA;
          size_t py3 = (py2 * y3_size) + lA;
          double tl = tk;
          for (size_t mA = 0; mA < A5_size; mA++) {
            size_t pA5 = (pA4 * A5_size) + mA;
            size_t py4 = (py3 * y4_size) + mA;
            double tm = tl;
            for (size_t oA = 0; oA < A6_size; oA++) {
              size_t pA6 = (pA5 * A6_size) + oA;
              size_t py5 = (py4 * y5_size) + oA;
              double to = tm;
              for (size_t pA = 0; pA < A7_size; pA++) {
                size_t pA7 = (pA6 * A7_size) + pA;
                size_t py6 = (py5 * y6_size) + pA;
                y_vals[py6] = y_vals[py6] + (A_vals[pA7] * to);
              }
            }
          }
        }
      }
    }
  }

		} else if (mode == 2) {

  for (size_t iA = 0; iA < A1_size; iA++) {
    for (size_t jA = 0; jA < A2_size; jA++) {
      size_t pA2 = (iA * A2_size) + jA;
      size_t py2 = (iA * y2_size) + jA;
      for (size_t kA = 0; kA < A3_size; kA++) {
        size_t pA3 = (pA2 * A3_size) + kA;
        double tk = x_vals[kA];
        for (size_t lA = 0; lA < A4_size; lA++) {
          size_t pA4 = (pA3 * A4_size) + lA;
          size_t py3 = (py2 * y3_size) + lA;
          double tl = tk;
          for (size_t mA = 0; mA < A5_size; mA++) {
            size_t pA5 = (pA4 * A5_size) + mA;
            size_t py4 = (py3 * y4_size) + mA;
            double tm = tl;
            for (size_t oA = 0; oA < A6_size; oA++) {
              size_t pA6 = (pA5 * A6_size) + oA;
              size_t py5 = (py4 * y5_size) + oA;
              double to = tm;
              for (size_t pA = 0; pA < A7_size; pA++) {
                size_t pA7 = (pA6 * A7_size) + pA;
                size_t py6 = (py5 * y6_size) + pA;
                y_vals[py6] = y_vals[py6] + (A_vals[pA7] * to);
              }
            }
          }
        }
      }
    }
  }


		} else if (mode == 3) {

  for (size_t iA = 0; iA < A1_size; iA++) {
    for (size_t jA = 0; jA < A2_size; jA++) {
      size_t pA2 = (iA * A2_size) + jA;
      size_t py2 = (iA * y2_size) + jA;
      for (size_t kA = 0; kA < A3_size; kA++) {
        size_t pA3 = (pA2 * A3_size) + kA;
        size_t py3 = (py2 * y3_size) + kA;
        for (size_t lA = 0; lA < A4_size; lA++) {
          size_t pA4 = (pA3 * A4_size) + lA;
          double tl = x_vals[lA];
          for (size_t mA = 0; mA < A5_size; mA++) {
            size_t pA5 = (pA4 * A5_size) + mA;
            size_t py4 = (py3 * y4_size) + mA;
            double tm = tl;
            for (size_t oA = 0; oA < A6_size; oA++) {
              size_t pA6 = (pA5 * A6_size) + oA;
              size_t py5 = (py4 * y5_size) + oA;
              double to = tm;
              for (size_t pA = 0; pA < A7_size; pA++) {
                size_t pA7 = (pA6 * A7_size) + pA;
                size_t py6 = (py5 * y6_size) + pA;
                y_vals[py6] = y_vals[py6] + (A_vals[pA7] * to);
              }
            }
          }
        }
      }
    }
  }


		} else if (mode == 4) {

 for (size_t iA = 0; iA < A1_size; iA++) {
    for (size_t jA = 0; jA < A2_size; jA++) {
      size_t pA2 = (iA * A2_size) + jA;
      size_t py2 = (iA * y2_size) + jA;
      for (size_t kA = 0; kA < A3_size; kA++) {
        size_t pA3 = (pA2 * A3_size) + kA;
        size_t py3 = (py2 * y3_size) + kA;
        for (size_t lA = 0; lA < A4_size; lA++) {
          size_t pA4 = (pA3 * A4_size) + lA;
          size_t py4 = (py3 * y4_size) + lA;
          for (size_t mA = 0; mA < A5_size; mA++) {
            size_t pA5 = (pA4 * A5_size) + mA;
            double tm = x_vals[mA];
            for (size_t oA = 0; oA < A6_size; oA++) {
              size_t pA6 = (pA5 * A6_size) + oA;
              size_t py5 = (py4 * y5_size) + oA;
              double to = tm;
              for (size_t pA = 0; pA < A7_size; pA++) {
                size_t pA7 = (pA6 * A7_size) + pA;
                size_t py6 = (py5 * y6_size) + pA;
                y_vals[py6] = y_vals[py6] + (A_vals[pA7] * to);
              }
            }
          }
        }
      }
    }
  }

		} else if (mode == 5) {

  for (size_t iA = 0; iA < A1_size; iA++) {
    for (size_t jA = 0; jA < A2_size; jA++) {
      size_t pA2 = (iA * A2_size) + jA;
      size_t py2 = (iA * y2_size) + jA;
      for (size_t kA = 0; kA < A3_size; kA++) {
        size_t pA3 = (pA2 * A3_size) + kA;
        size_t py3 = (py2 * y3_size) + kA;
        for (size_t lA = 0; lA < A4_size; lA++) {
          size_t pA4 = (pA3 * A4_size) + lA;
          size_t py4 = (py3 * y4_size) + lA;
          for (size_t mA = 0; mA < A5_size; mA++) {
            size_t pA5 = (pA4 * A5_size) + mA;
            size_t py5 = (py4 * y5_size) + mA;
            for (size_t oA = 0; oA < A6_size; oA++) {
              size_t pA6 = (pA5 * A6_size) + oA;
              double to = x_vals[oA];
              for (size_t pA = 0; pA < A7_size; pA++) {
                size_t pA7 = (pA6 * A7_size) + pA;
                size_t py6 = (py5 * y6_size) + pA;
                y_vals[py6] = y_vals[py6] + (A_vals[pA7] * to);
              }
            }
          }
        }
      }
    }
  }


		} else if (mode == 6) {

  for (size_t iA = 0; iA < A1_size; iA++) {
    for (size_t jA = 0; jA < A2_size; jA++) {
      size_t pA2 = (iA * A2_size) + jA;
      size_t py2 = (iA * y2_size) + jA;
      for (size_t kA = 0; kA < A3_size; kA++) {
        size_t pA3 = (pA2 * A3_size) + kA;
        size_t py3 = (py2 * y3_size) + kA;
        for (size_t lA = 0; lA < A4_size; lA++) {
          size_t pA4 = (pA3 * A4_size) + lA;
          size_t py4 = (py3 * y4_size) + lA;
          for (size_t mA = 0; mA < A5_size; mA++) {
            size_t pA5 = (pA4 * A5_size) + mA;
            size_t py5 = (py4 * y5_size) + mA;
            for (size_t oA = 0; oA < A6_size; oA++) {
              size_t pA6 = (pA5 * A6_size) + oA;
              size_t py6 = (py5 * y6_size) + oA;
              double tp = 0;
              for (size_t pA = 0; pA < A7_size; pA++) {
                size_t pA7 = (pA6 * A7_size) + pA;
                tp += A_vals[pA7] * x_vals[pA];
              }
              y_vals[py6] = tp;
            }
          }
        }
      }
    }
  }
		}
	} else if (dim == 8) {
		if (mode == 0) {
		 for (size_t iA = 0; iA < A1_size; iA++) {
		    double ti = x_vals[iA];
		    for (size_t jA = 0; jA < A2_size; jA++) {
		      size_t pA2 = (iA * A2_size) + jA;
		      double tj = ti;
		      for (size_t kA = 0; kA < A3_size; kA++) {
		        size_t pA3 = (pA2 * A3_size) + kA;
		        size_t py2 = (jA * y2_size) + kA;
		        double tk = tj;
		        for (size_t lA = 0; lA < A4_size; lA++) {
		          size_t pA4 = (pA3 * A4_size) + lA;
		          size_t py3 = (py2 * y3_size) + lA;
		          double tl = tk;
		          for (size_t mA = 0; mA < A5_size; mA++) {
		            size_t pA5 = (pA4 * A5_size) + mA;
		            size_t py4 = (py3 * y4_size) + mA;
		            double tm = tl;
		            for (size_t oA = 0; oA < A6_size; oA++) {
		              size_t pA6 = (pA5 * A6_size) + oA;
		              size_t py5 = (py4 * y5_size) + oA;
		              double to = tm;
		              for (size_t pA = 0; pA < A7_size; pA++) {
		                size_t pA7 = (pA6 * A7_size) + pA;
		                size_t py6 = (py5 * y6_size) + pA;
		                double tp = to;
		                for (size_t rA = 0; rA < A8_size; rA++) {
		                  size_t pA8 = (pA7 * A8_size) + rA;
		                  size_t py7 = (py6 * y7_size) + rA;
		                  y_vals[py7] = y_vals[py7] + (A_vals[pA8] * tp);
		                }
		              }
		            }
		          }
		        }
		      }
		    }
		}
		} else if (mode == 1) {

  for (size_t iA = 0; iA < A1_size; iA++) {
    for (size_t jA = 0; jA < A2_size; jA++) {
      size_t pA2 = (iA * A2_size) + jA;
      double tj = x_vals[jA];
      for (size_t kA = 0; kA < A3_size; kA++) {
        size_t pA3 = (pA2 * A3_size) + kA;
        size_t py2 = (iA * y2_size) + kA;
        double tk = tj;
        for (size_t lA = 0; lA < A4_size; lA++) {
          size_t pA4 = (pA3 * A4_size) + lA;
          size_t py3 = (py2 * y3_size) + lA;
          double tl = tk;
          for (size_t mA = 0; mA < A5_size; mA++) {
            size_t pA5 = (pA4 * A5_size) + mA;
            size_t py4 = (py3 * y4_size) + mA;
            double tm = tl;
            for (size_t oA = 0; oA < A6_size; oA++) {
              size_t pA6 = (pA5 * A6_size) + oA;
              size_t py5 = (py4 * y5_size) + oA;
              double to = tm;
              for (size_t pA = 0; pA < A7_size; pA++) {
                size_t pA7 = (pA6 * A7_size) + pA;
                size_t py6 = (py5 * y6_size) + pA;
                double tp = to;
                for (size_t rA = 0; rA < A8_size; rA++) {
                  size_t pA8 = (pA7 * A8_size) + rA;
                  size_t py7 = (py6 * y7_size) + rA;
                  y_vals[py7] = y_vals[py7] + (A_vals[pA8] * tp);
                }
              }
            }
          }
        }
      }
    }
  }

		} else if (mode == 2) {

  for (size_t iA = 0; iA < A1_size; iA++) {
    for (size_t jA = 0; jA < A2_size; jA++) {
      size_t pA2 = (iA * A2_size) + jA;
      size_t py2 = (iA * y2_size) + jA;
      for (size_t kA = 0; kA < A3_size; kA++) {
        size_t pA3 = (pA2 * A3_size) + kA;
        double tk = x_vals[kA];
        for (size_t lA = 0; lA < A4_size; lA++) {
          size_t pA4 = (pA3 * A4_size) + lA;
          size_t py3 = (py2 * y3_size) + lA;
          double tl = tk;
          for (size_t mA = 0; mA < A5_size; mA++) {
            size_t pA5 = (pA4 * A5_size) + mA;
            size_t py4 = (py3 * y4_size) + mA;
            double tm = tl;
            for (size_t oA = 0; oA < A6_size; oA++) {
              size_t pA6 = (pA5 * A6_size) + oA;
              size_t py5 = (py4 * y5_size) + oA;
              double to = tm;
              for (size_t pA = 0; pA < A7_size; pA++) {
                size_t pA7 = (pA6 * A7_size) + pA;
                size_t py6 = (py5 * y6_size) + pA;
                double tp = to;
                for (size_t rA = 0; rA < A8_size; rA++) {
                  size_t pA8 = (pA7 * A8_size) + rA;
                  size_t py7 = (py6 * y7_size) + rA;
                  y_vals[py7] = y_vals[py7] + (A_vals[pA8] * tp);
                }
              }
            }
          }
        }
      }
    }
  }

		} else if (mode == 3) {

  for (size_t iA = 0; iA < A1_size; iA++) {
    for (size_t jA = 0; jA < A2_size; jA++) {
      size_t pA2 = (iA * A2_size) + jA;
      size_t py2 = (iA * y2_size) + jA;
      for (size_t kA = 0; kA < A3_size; kA++) {
        size_t pA3 = (pA2 * A3_size) + kA;
        size_t py3 = (py2 * y3_size) + kA;
        for (size_t lA = 0; lA < A4_size; lA++) {
          size_t pA4 = (pA3 * A4_size) + lA;
          double tl = x_vals[lA];
          for (size_t mA = 0; mA < A5_size; mA++) {
            size_t pA5 = (pA4 * A5_size) + mA;
            size_t py4 = (py3 * y4_size) + mA;
            double tm = tl;
            for (size_t oA = 0; oA < A6_size; oA++) {
              size_t pA6 = (pA5 * A6_size) + oA;
              size_t py5 = (py4 * y5_size) + oA;
              double to = tm;
              for (size_t pA = 0; pA < A7_size; pA++) {
                size_t pA7 = (pA6 * A7_size) + pA;
                size_t py6 = (py5 * y6_size) + pA;
                double tp = to;
                for (size_t rA = 0; rA < A8_size; rA++) {
                  size_t pA8 = (pA7 * A8_size) + rA;
                  size_t py7 = (py6 * y7_size) + rA;
                  y_vals[py7] = y_vals[py7] + (A_vals[pA8] * tp);
                }
              }
            }
          }
        }
      }
    }
  }

		} else if (mode == 4) {

  for (size_t iA = 0; iA < A1_size; iA++) {
    for (size_t jA = 0; jA < A2_size; jA++) {
      size_t pA2 = (iA * A2_size) + jA;
      size_t py2 = (iA * y2_size) + jA;
      for (size_t kA = 0; kA < A3_size; kA++) {
        size_t pA3 = (pA2 * A3_size) + kA;
        size_t py3 = (py2 * y3_size) + kA;
        for (size_t lA = 0; lA < A4_size; lA++) {
          size_t pA4 = (pA3 * A4_size) + lA;
          size_t py4 = (py3 * y4_size) + lA;
          for (size_t mA = 0; mA < A5_size; mA++) {
            size_t pA5 = (pA4 * A5_size) + mA;
            double tm = x_vals[mA];
            for (size_t oA = 0; oA < A6_size; oA++) {
              size_t pA6 = (pA5 * A6_size) + oA;
              size_t py5 = (py4 * y5_size) + oA;
              double to = tm;
              for (size_t pA = 0; pA < A7_size; pA++) {
                size_t pA7 = (pA6 * A7_size) + pA;
                size_t py6 = (py5 * y6_size) + pA;
                double tp = to;
                for (size_t rA = 0; rA < A8_size; rA++) {
                  size_t pA8 = (pA7 * A8_size) + rA;
                  size_t py7 = (py6 * y7_size) + rA;
                  y_vals[py7] = y_vals[py7] + (A_vals[pA8] * tp);
                }
              }
            }
          }
        }
      }
    }
  }

		} else if (mode == 5) {

  for (size_t iA = 0; iA < A1_size; iA++) {
    for (size_t jA = 0; jA < A2_size; jA++) {
      size_t pA2 = (iA * A2_size) + jA;
      size_t py2 = (iA * y2_size) + jA;
      for (size_t kA = 0; kA < A3_size; kA++) {
        size_t pA3 = (pA2 * A3_size) + kA;
        size_t py3 = (py2 * y3_size) + kA;
        for (size_t lA = 0; lA < A4_size; lA++) {
          size_t pA4 = (pA3 * A4_size) + lA;
          size_t py4 = (py3 * y4_size) + lA;
          for (size_t mA = 0; mA < A5_size; mA++) {
            size_t pA5 = (pA4 * A5_size) + mA;
            size_t py5 = (py4 * y5_size) + mA;
            for (size_t oA = 0; oA < A6_size; oA++) {
              size_t pA6 = (pA5 * A6_size) + oA;
              double to = x_vals[oA];
              for (size_t pA = 0; pA < A7_size; pA++) {
                size_t pA7 = (pA6 * A7_size) + pA;
                size_t py6 = (py5 * y6_size) + pA;
                double tp = to;
                for (size_t rA = 0; rA < A8_size; rA++) {
                  size_t pA8 = (pA7 * A8_size) + rA;
                  size_t py7 = (py6 * y7_size) + rA;
                  y_vals[py7] = y_vals[py7] + (A_vals[pA8] * tp);
                }
              }
            }
          }
        }
      }
    }
  }


		} else if (mode == 6) {

  for (size_t iA = 0; iA < A1_size; iA++) {
    for (size_t jA = 0; jA < A2_size; jA++) {
      size_t pA2 = (iA * A2_size) + jA;
      size_t py2 = (iA * y2_size) + jA;
      for (size_t kA = 0; kA < A3_size; kA++) {
        size_t pA3 = (pA2 * A3_size) + kA;
        size_t py3 = (py2 * y3_size) + kA;
        for (size_t lA = 0; lA < A4_size; lA++) {
          size_t pA4 = (pA3 * A4_size) + lA;
          size_t py4 = (py3 * y4_size) + lA;
          for (size_t mA = 0; mA < A5_size; mA++) {
            size_t pA5 = (pA4 * A5_size) + mA;
            size_t py5 = (py4 * y5_size) + mA;
            for (size_t oA = 0; oA < A6_size; oA++) {
              size_t pA6 = (pA5 * A6_size) + oA;
              size_t py6 = (py5 * y6_size) + oA;
              for (size_t pA = 0; pA < A7_size; pA++) {
                size_t pA7 = (pA6 * A7_size) + pA;
                double tp = x_vals[pA];
                for (size_t rA = 0; rA < A8_size; rA++) {
                  size_t pA8 = (pA7 * A8_size) + rA;
                  size_t py7 = (py6 * y7_size) + rA;
                  y_vals[py7] = y_vals[py7] + (A_vals[pA8] * tp);
                }
              }
            }
          }
        }
      }
    }
  }

		} else if (mode == 7) {

  for (size_t iA = 0; iA < A1_size; iA++) {
    for (size_t jA = 0; jA < A2_size; jA++) {
      size_t pA2 = (iA * A2_size) + jA;
      size_t py2 = (iA * y2_size) + jA;
      for (size_t kA = 0; kA < A3_size; kA++) {
        size_t pA3 = (pA2 * A3_size) + kA;
        size_t py3 = (py2 * y3_size) + kA;
        for (size_t lA = 0; lA < A4_size; lA++) {
          size_t pA4 = (pA3 * A4_size) + lA;
          size_t py4 = (py3 * y4_size) + lA;
          for (size_t mA = 0; mA < A5_size; mA++) {
            size_t pA5 = (pA4 * A5_size) + mA;
            size_t py5 = (py4 * y5_size) + mA;
            for (size_t oA = 0; oA < A6_size; oA++) {
              size_t pA6 = (pA5 * A6_size) + oA;
              size_t py6 = (py5 * y6_size) + oA;
              for (size_t pA = 0; pA < A7_size; pA++) {
                size_t pA7 = (pA6 * A7_size) + pA;
                size_t py7 = (py6 * y7_size) + pA;
                double tr = 0;
                for (size_t rA = 0; rA < A8_size; rA++) {
                  size_t pA8 = (pA7 * A8_size) + rA;
                  tr += A_vals[pA8] * x_vals[rA];
                }
                y_vals[py7] = tr;
              }
            }
          }
        }
      }
    }
  }


		}

	} else if (dim == 9) {
		if (mode == 0) {
		  for (size_t iA = 0; iA < A1_size; iA++) {
		    double ti = x_vals[iA];
		    for (size_t jA = 0; jA < A2_size; jA++) {
		      size_t pA2 = (iA * A2_size) + jA;
		      double tj = ti;
		      for (size_t kA = 0; kA < A3_size; kA++) {
		        size_t pA3 = (pA2 * A3_size) + kA;
		        size_t py2 = (jA * y2_size) + kA;
		        double tk = tj;
		        for (size_t lA = 0; lA < A4_size; lA++) {
		          size_t pA4 = (pA3 * A4_size) + lA;
		          size_t py3 = (py2 * y3_size) + lA;
		          double tl = tk;
		          for (size_t mA = 0; mA < A5_size; mA++) {
		            size_t pA5 = (pA4 * A5_size) + mA;
		            size_t py4 = (py3 * y4_size) + mA;
		            double tm = tl;
		            for (size_t oA = 0; oA < A6_size; oA++) {
		              size_t pA6 = (pA5 * A6_size) + oA;
		              size_t py5 = (py4 * y5_size) + oA;
		              double to = tm;
		              for (size_t pA = 0; pA < A7_size; pA++) {
		                size_t pA7 = (pA6 * A7_size) + pA;
		                size_t py6 = (py5 * y6_size) + pA;
		                double tp = to;
		                for (size_t rA = 0; rA < A8_size; rA++) {
		                  size_t pA8 = (pA7 * A8_size) + rA;
		                  size_t py7 = (py6 * y7_size) + rA;
		                  double tr = tp;
		                  for (size_t sA = 0; sA < A9_size; sA++) {
		                    size_t pA9 = (pA8 * A9_size) + sA;
		                    size_t py8 = (py7 * y8_size) + sA;
		                    y_vals[py8] = y_vals[py8] + (A_vals[pA9] * tr);
		                  }
		                }
		              }
		            }
		          }
		        }
		      }
		    }
		  }
		} else if (mode == 1) {

  for (size_t iA = 0; iA < A1_size; iA++) {
    for (size_t jA = 0; jA < A2_size; jA++) {
      size_t pA2 = (iA * A2_size) + jA;
      double tj = x_vals[jA];
      for (size_t kA = 0; kA < A3_size; kA++) {
        size_t pA3 = (pA2 * A3_size) + kA;
        size_t py2 = (iA * y2_size) + kA;
        double tk = tj;
        for (size_t lA = 0; lA < A4_size; lA++) {
          size_t pA4 = (pA3 * A4_size) + lA;
          size_t py3 = (py2 * y3_size) + lA;
          double tl = tk;
          for (size_t mA = 0; mA < A5_size; mA++) {
            size_t pA5 = (pA4 * A5_size) + mA;
            size_t py4 = (py3 * y4_size) + mA;
            double tm = tl;
            for (size_t oA = 0; oA < A6_size; oA++) {
              size_t pA6 = (pA5 * A6_size) + oA;
              size_t py5 = (py4 * y5_size) + oA;
              double to = tm;
              for (size_t pA = 0; pA < A7_size; pA++) {
                size_t pA7 = (pA6 * A7_size) + pA;
                size_t py6 = (py5 * y6_size) + pA;
                double tp = to;
                for (size_t rA = 0; rA < A8_size; rA++) {
                  size_t pA8 = (pA7 * A8_size) + rA;
                  size_t py7 = (py6 * y7_size) + rA;
                  double tr = tp;
                  for (size_t sA = 0; sA < A9_size; sA++) {
                    size_t pA9 = (pA8 * A9_size) + sA;
                    size_t py8 = (py7 * y8_size) + sA;
                    y_vals[py8] = y_vals[py8] + (A_vals[pA9] * tr);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

		} else if (mode == 2) {

  for (size_t iA = 0; iA < A1_size; iA++) {
    for (size_t jA = 0; jA < A2_size; jA++) {
      size_t pA2 = (iA * A2_size) + jA;
      size_t py2 = (iA * y2_size) + jA;
      for (size_t kA = 0; kA < A3_size; kA++) {
        size_t pA3 = (pA2 * A3_size) + kA;
        double tk = x_vals[kA];
        for (size_t lA = 0; lA < A4_size; lA++) {
          size_t pA4 = (pA3 * A4_size) + lA;
          size_t py3 = (py2 * y3_size) + lA;
          double tl = tk;
          for (size_t mA = 0; mA < A5_size; mA++) {
            size_t pA5 = (pA4 * A5_size) + mA;
            size_t py4 = (py3 * y4_size) + mA;
            double tm = tl;
            for (size_t oA = 0; oA < A6_size; oA++) {
              size_t pA6 = (pA5 * A6_size) + oA;
              size_t py5 = (py4 * y5_size) + oA;
              double to = tm;
              for (size_t pA = 0; pA < A7_size; pA++) {
                size_t pA7 = (pA6 * A7_size) + pA;
                size_t py6 = (py5 * y6_size) + pA;
                double tp = to;
                for (size_t rA = 0; rA < A8_size; rA++) {
                  size_t pA8 = (pA7 * A8_size) + rA;
                  size_t py7 = (py6 * y7_size) + rA;
                  double tr = tp;
                  for (size_t sA = 0; sA < A9_size; sA++) {
                    size_t pA9 = (pA8 * A9_size) + sA;
                    size_t py8 = (py7 * y8_size) + sA;
                    y_vals[py8] = y_vals[py8] + (A_vals[pA9] * tr);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

		} else if (mode == 3) {

  for (size_t iA = 0; iA < A1_size; iA++) {
    for (size_t jA = 0; jA < A2_size; jA++) {
      size_t pA2 = (iA * A2_size) + jA;
      size_t py2 = (iA * y2_size) + jA;
      for (size_t kA = 0; kA < A3_size; kA++) {
        size_t pA3 = (pA2 * A3_size) + kA;
        size_t py3 = (py2 * y3_size) + kA;
        for (size_t lA = 0; lA < A4_size; lA++) {
          size_t pA4 = (pA3 * A4_size) + lA;
          double tl = x_vals[lA];
          for (size_t mA = 0; mA < A5_size; mA++) {
            size_t pA5 = (pA4 * A5_size) + mA;
            size_t py4 = (py3 * y4_size) + mA;
            double tm = tl;
            for (size_t oA = 0; oA < A6_size; oA++) {
              size_t pA6 = (pA5 * A6_size) + oA;
              size_t py5 = (py4 * y5_size) + oA;
              double to = tm;
              for (size_t pA = 0; pA < A7_size; pA++) {
                size_t pA7 = (pA6 * A7_size) + pA;
                size_t py6 = (py5 * y6_size) + pA;
                double tp = to;
                for (size_t rA = 0; rA < A8_size; rA++) {
                  size_t pA8 = (pA7 * A8_size) + rA;
                  size_t py7 = (py6 * y7_size) + rA;
                  double tr = tp;
                  for (size_t sA = 0; sA < A9_size; sA++) {
                    size_t pA9 = (pA8 * A9_size) + sA;
                    size_t py8 = (py7 * y8_size) + sA;
                    y_vals[py8] = y_vals[py8] + (A_vals[pA9] * tr);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

		} else if (mode == 4) {

  for (size_t iA = 0; iA < A1_size; iA++) {
    for (size_t jA = 0; jA < A2_size; jA++) {
      size_t pA2 = (iA * A2_size) + jA;
      size_t py2 = (iA * y2_size) + jA;
      for (size_t kA = 0; kA < A3_size; kA++) {
        size_t pA3 = (pA2 * A3_size) + kA;
        size_t py3 = (py2 * y3_size) + kA;
        for (size_t lA = 0; lA < A4_size; lA++) {
          size_t pA4 = (pA3 * A4_size) + lA;
          size_t py4 = (py3 * y4_size) + lA;
          for (size_t mA = 0; mA < A5_size; mA++) {
            size_t pA5 = (pA4 * A5_size) + mA;
            double tm = x_vals[mA];
            for (size_t oA = 0; oA < A6_size; oA++) {
              size_t pA6 = (pA5 * A6_size) + oA;
              size_t py5 = (py4 * y5_size) + oA;
              double to = tm;
              for (size_t pA = 0; pA < A7_size; pA++) {
                size_t pA7 = (pA6 * A7_size) + pA;
                size_t py6 = (py5 * y6_size) + pA;
                double tp = to;
                for (size_t rA = 0; rA < A8_size; rA++) {
                  size_t pA8 = (pA7 * A8_size) + rA;
                  size_t py7 = (py6 * y7_size) + rA;
                  double tr = tp;
                  for (size_t sA = 0; sA < A9_size; sA++) {
                    size_t pA9 = (pA8 * A9_size) + sA;
                    size_t py8 = (py7 * y8_size) + sA;
                    y_vals[py8] = y_vals[py8] + (A_vals[pA9] * tr);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

		} else if (mode == 5) {

  for (size_t iA = 0; iA < A1_size; iA++) {
    for (size_t jA = 0; jA < A2_size; jA++) {
      size_t pA2 = (iA * A2_size) + jA;
      size_t py2 = (iA * y2_size) + jA;
      for (size_t kA = 0; kA < A3_size; kA++) {
        size_t pA3 = (pA2 * A3_size) + kA;
        size_t py3 = (py2 * y3_size) + kA;
        for (size_t lA = 0; lA < A4_size; lA++) {
          size_t pA4 = (pA3 * A4_size) + lA;
          size_t py4 = (py3 * y4_size) + lA;
          for (size_t mA = 0; mA < A5_size; mA++) {
            size_t pA5 = (pA4 * A5_size) + mA;
            size_t py5 = (py4 * y5_size) + mA;
            for (size_t oA = 0; oA < A6_size; oA++) {
              size_t pA6 = (pA5 * A6_size) + oA;
              double to = x_vals[oA];
              for (size_t pA = 0; pA < A7_size; pA++) {
                size_t pA7 = (pA6 * A7_size) + pA;
                size_t py6 = (py5 * y6_size) + pA;
                double tp = to;
                for (size_t rA = 0; rA < A8_size; rA++) {
                  size_t pA8 = (pA7 * A8_size) + rA;
                  size_t py7 = (py6 * y7_size) + rA;
                  double tr = tp;
                  for (size_t sA = 0; sA < A9_size; sA++) {
                    size_t pA9 = (pA8 * A9_size) + sA;
                    size_t py8 = (py7 * y8_size) + sA;
                    y_vals[py8] = y_vals[py8] + (A_vals[pA9] * tr);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

		} else if (mode == 6) {

  for (size_t iA = 0; iA < A1_size; iA++) {
    for (size_t jA = 0; jA < A2_size; jA++) {
      size_t pA2 = (iA * A2_size) + jA;
      size_t py2 = (iA * y2_size) + jA;
      for (size_t kA = 0; kA < A3_size; kA++) {
        size_t pA3 = (pA2 * A3_size) + kA;
        size_t py3 = (py2 * y3_size) + kA;
        for (size_t lA = 0; lA < A4_size; lA++) {
          size_t pA4 = (pA3 * A4_size) + lA;
          size_t py4 = (py3 * y4_size) + lA;
          for (size_t mA = 0; mA < A5_size; mA++) {
            size_t pA5 = (pA4 * A5_size) + mA;
            size_t py5 = (py4 * y5_size) + mA;
            for (size_t oA = 0; oA < A6_size; oA++) {
              size_t pA6 = (pA5 * A6_size) + oA;
              size_t py6 = (py5 * y6_size) + oA;
              for (size_t pA = 0; pA < A7_size; pA++) {
                size_t pA7 = (pA6 * A7_size) + pA;
                double tp = x_vals[pA];
                for (size_t rA = 0; rA < A8_size; rA++) {
                  size_t pA8 = (pA7 * A8_size) + rA;
                  size_t py7 = (py6 * y7_size) + rA;
                  double tr = tp;
                  for (size_t sA = 0; sA < A9_size; sA++) {
                    size_t pA9 = (pA8 * A9_size) + sA;
                    size_t py8 = (py7 * y8_size) + sA;
                    y_vals[py8] = y_vals[py8] + (A_vals[pA9] * tr);
                  }
                }
              }
            }
          }
        }
      }
    }
  }


		} else if (mode == 7) {

  for (size_t iA = 0; iA < A1_size; iA++) {
    for (size_t jA = 0; jA < A2_size; jA++) {
      size_t pA2 = (iA * A2_size) + jA;
      size_t py2 = (iA * y2_size) + jA;
      for (size_t kA = 0; kA < A3_size; kA++) {
        size_t pA3 = (pA2 * A3_size) + kA;
        size_t py3 = (py2 * y3_size) + kA;
        for (size_t lA = 0; lA < A4_size; lA++) {
          size_t pA4 = (pA3 * A4_size) + lA;
          size_t py4 = (py3 * y4_size) + lA;
          for (size_t mA = 0; mA < A5_size; mA++) {
            size_t pA5 = (pA4 * A5_size) + mA;
            size_t py5 = (py4 * y5_size) + mA;
            for (size_t oA = 0; oA < A6_size; oA++) {
              size_t pA6 = (pA5 * A6_size) + oA;
              size_t py6 = (py5 * y6_size) + oA;
              for (size_t pA = 0; pA < A7_size; pA++) {
                size_t pA7 = (pA6 * A7_size) + pA;
                size_t py7 = (py6 * y7_size) + pA;
                for (size_t rA = 0; rA < A8_size; rA++) {
                  size_t pA8 = (pA7 * A8_size) + rA;
                  double tr = x_vals[rA];
                  for (size_t sA = 0; sA < A9_size; sA++) {
                    size_t pA9 = (pA8 * A9_size) + sA;
                    size_t py8 = (py7 * y8_size) + sA;
                    y_vals[py8] = y_vals[py8] + (A_vals[pA9] * tr);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

		} else if (mode == 8) {

  for (size_t iA = 0; iA < A1_size; iA++) {
    for (size_t jA = 0; jA < A2_size; jA++) {
      size_t pA2 = (iA * A2_size) + jA;
      size_t py2 = (iA * y2_size) + jA;
      for (size_t kA = 0; kA < A3_size; kA++) {
        size_t pA3 = (pA2 * A3_size) + kA;
        size_t py3 = (py2 * y3_size) + kA;
        for (size_t lA = 0; lA < A4_size; lA++) {
          size_t pA4 = (pA3 * A4_size) + lA;
          size_t py4 = (py3 * y4_size) + lA;
          for (size_t mA = 0; mA < A5_size; mA++) {
            size_t pA5 = (pA4 * A5_size) + mA;
            size_t py5 = (py4 * y5_size) + mA;
            for (size_t oA = 0; oA < A6_size; oA++) {
              size_t pA6 = (pA5 * A6_size) + oA;
              size_t py6 = (py5 * y6_size) + oA;
              for (size_t pA = 0; pA < A7_size; pA++) {
                size_t pA7 = (pA6 * A7_size) + pA;
                size_t py7 = (py6 * y7_size) + pA;
                for (size_t rA = 0; rA < A8_size; rA++) {
                  size_t pA8 = (pA7 * A8_size) + rA;
                  size_t py8 = (py7 * y8_size) + rA;
                  double ts = 0;
                  for (size_t sA = 0; sA < A9_size; sA++) {
                    size_t pA9 = (pA8 * A9_size) + sA;
                    ts += A_vals[pA9] * x_vals[sA];
                  }
                  y_vals[py8] = ts;
                }
              }
            }
          }
        }
      }
    }
  }

		}

	} else if (dim == 10) {
		if (mode == 0) {
		  for (size_t iA = 0; iA < A1_size; iA++) {
		    double ti = x_vals[iA];
		    for (size_t jA = 0; jA < A2_size; jA++) {
		      size_t pA2 = (iA * A2_size) + jA;
		      double tj = ti;
		      for (size_t kA = 0; kA < A3_size; kA++) {
		        size_t pA3 = (pA2 * A3_size) + kA;
		        size_t py2 = (jA * y2_size) + kA;
		        double tk = tj;
		        for (size_t lA = 0; lA < A4_size; lA++) {
		          size_t pA4 = (pA3 * A4_size) + lA;
		          size_t py3 = (py2 * y3_size) + lA;
		          double tl = tk;
		          for (size_t mA = 0; mA < A5_size; mA++) {
		            size_t pA5 = (pA4 * A5_size) + mA;
		            size_t py4 = (py3 * y4_size) + mA;
		            double tm = tl;
		            for (size_t oA = 0; oA < A6_size; oA++) {
		              size_t pA6 = (pA5 * A6_size) + oA;
		              size_t py5 = (py4 * y5_size) + oA;
		              double to = tm;
		              for (size_t pA = 0; pA < A7_size; pA++) {
		                size_t pA7 = (pA6 * A7_size) + pA;
		                size_t py6 = (py5 * y6_size) + pA;
		                double tp = to;
		                for (size_t rA = 0; rA < A8_size; rA++) {
		                  size_t pA8 = (pA7 * A8_size) + rA;
		                  size_t py7 = (py6 * y7_size) + rA;
		                  double tr = tp;
		                  for (size_t sA = 0; sA < A9_size; sA++) {
		                    size_t pA9 = (pA8 * A9_size) + sA;
		                    size_t py8 = (py7 * y8_size) + sA;
		                    double ts = tr;
		                    for (size_t tA = 0; tA < A10_size; tA++) {
		                      size_t pA10 = (pA9 * A10_size) + tA;
		                      size_t py9 = (py8 * y9_size) + tA;
		                      y_vals[py9] = y_vals[py9] + (A_vals[pA10] * ts);
		                    }
		                  }
		                }
		              }
		            }
		          }
		        }
		      }
		    }
		  }
		} else if (mode == 1) {

  for (size_t iA = 0; iA < A1_size; iA++) {
    for (size_t jA = 0; jA < A2_size; jA++) {
      size_t pA2 = (iA * A2_size) + jA;
      double tj = x_vals[jA];
      for (size_t kA = 0; kA < A3_size; kA++) {
        size_t pA3 = (pA2 * A3_size) + kA;
        size_t py2 = (iA * y2_size) + kA;
        double tk = tj;
        for (size_t lA = 0; lA < A4_size; lA++) {
          size_t pA4 = (pA3 * A4_size) + lA;
          size_t py3 = (py2 * y3_size) + lA;
          double tl = tk;
          for (size_t mA = 0; mA < A5_size; mA++) {
            size_t pA5 = (pA4 * A5_size) + mA;
            size_t py4 = (py3 * y4_size) + mA;
            double tm = tl;
            for (size_t oA = 0; oA < A6_size; oA++) {
              size_t pA6 = (pA5 * A6_size) + oA;
              size_t py5 = (py4 * y5_size) + oA;
              double to = tm;
              for (size_t pA = 0; pA < A7_size; pA++) {
                size_t pA7 = (pA6 * A7_size) + pA;
                size_t py6 = (py5 * y6_size) + pA;
                double tp = to;
                for (size_t rA = 0; rA < A8_size; rA++) {
                  size_t pA8 = (pA7 * A8_size) + rA;
                  size_t py7 = (py6 * y7_size) + rA;
                  double tr = tp;
                  for (size_t sA = 0; sA < A9_size; sA++) {
                    size_t pA9 = (pA8 * A9_size) + sA;
                    size_t py8 = (py7 * y8_size) + sA;
                    double ts = tr;
                    for (size_t tA = 0; tA < A10_size; tA++) {
                      size_t pA10 = (pA9 * A10_size) + tA;
                      size_t py9 = (py8 * y9_size) + tA;
                      y_vals[py9] = y_vals[py9] + (A_vals[pA10] * ts);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

		} else if (mode == 2) {

  for (size_t iA = 0; iA < A1_size; iA++) {
    for (size_t jA = 0; jA < A2_size; jA++) {
      size_t pA2 = (iA * A2_size) + jA;
      size_t py2 = (iA * y2_size) + jA;
      for (size_t kA = 0; kA < A3_size; kA++) {
        size_t pA3 = (pA2 * A3_size) + kA;
        double tk = x_vals[kA];
        for (size_t lA = 0; lA < A4_size; lA++) {
          size_t pA4 = (pA3 * A4_size) + lA;
          size_t py3 = (py2 * y3_size) + lA;
          double tl = tk;
          for (size_t mA = 0; mA < A5_size; mA++) {
            size_t pA5 = (pA4 * A5_size) + mA;
            size_t py4 = (py3 * y4_size) + mA;
            double tm = tl;
            for (size_t oA = 0; oA < A6_size; oA++) {
              size_t pA6 = (pA5 * A6_size) + oA;
              size_t py5 = (py4 * y5_size) + oA;
              double to = tm;
              for (size_t pA = 0; pA < A7_size; pA++) {
                size_t pA7 = (pA6 * A7_size) + pA;
                size_t py6 = (py5 * y6_size) + pA;
                double tp = to;
                for (size_t rA = 0; rA < A8_size; rA++) {
                  size_t pA8 = (pA7 * A8_size) + rA;
                  size_t py7 = (py6 * y7_size) + rA;
                  double tr = tp;
                  for (size_t sA = 0; sA < A9_size; sA++) {
                    size_t pA9 = (pA8 * A9_size) + sA;
                    size_t py8 = (py7 * y8_size) + sA;
                    double ts = tr;
                    for (size_t tA = 0; tA < A10_size; tA++) {
                      size_t pA10 = (pA9 * A10_size) + tA;
                      size_t py9 = (py8 * y9_size) + tA;
                      y_vals[py9] = y_vals[py9] + (A_vals[pA10] * ts);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }


		} else if (mode == 3) {

  for (size_t iA = 0; iA < A1_size; iA++) {
    for (size_t jA = 0; jA < A2_size; jA++) {
      size_t pA2 = (iA * A2_size) + jA;
      size_t py2 = (iA * y2_size) + jA;
      for (size_t kA = 0; kA < A3_size; kA++) {
        size_t pA3 = (pA2 * A3_size) + kA;
        size_t py3 = (py2 * y3_size) + kA;
        for (size_t lA = 0; lA < A4_size; lA++) {
          size_t pA4 = (pA3 * A4_size) + lA;
          double tl = x_vals[lA];
          for (size_t mA = 0; mA < A5_size; mA++) {
            size_t pA5 = (pA4 * A5_size) + mA;
            size_t py4 = (py3 * y4_size) + mA;
            double tm = tl;
            for (size_t oA = 0; oA < A6_size; oA++) {
              size_t pA6 = (pA5 * A6_size) + oA;
              size_t py5 = (py4 * y5_size) + oA;
              double to = tm;
              for (size_t pA = 0; pA < A7_size; pA++) {
                size_t pA7 = (pA6 * A7_size) + pA;
                size_t py6 = (py5 * y6_size) + pA;
                double tp = to;
                for (size_t rA = 0; rA < A8_size; rA++) {
                  size_t pA8 = (pA7 * A8_size) + rA;
                  size_t py7 = (py6 * y7_size) + rA;
                  double tr = tp;
                  for (size_t sA = 0; sA < A9_size; sA++) {
                    size_t pA9 = (pA8 * A9_size) + sA;
                    size_t py8 = (py7 * y8_size) + sA;
                    double ts = tr;
                    for (size_t tA = 0; tA < A10_size; tA++) {
                      size_t pA10 = (pA9 * A10_size) + tA;
                      size_t py9 = (py8 * y9_size) + tA;
                      y_vals[py9] = y_vals[py9] + (A_vals[pA10] * ts);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

		} else if (mode == 4) {

  for (size_t iA = 0; iA < A1_size; iA++) {
    for (size_t jA = 0; jA < A2_size; jA++) {
      size_t pA2 = (iA * A2_size) + jA;
      size_t py2 = (iA * y2_size) + jA;
      for (size_t kA = 0; kA < A3_size; kA++) {
        size_t pA3 = (pA2 * A3_size) + kA;
        size_t py3 = (py2 * y3_size) + kA;
        for (size_t lA = 0; lA < A4_size; lA++) {
          size_t pA4 = (pA3 * A4_size) + lA;
          size_t py4 = (py3 * y4_size) + lA;
          for (size_t mA = 0; mA < A5_size; mA++) {
            size_t pA5 = (pA4 * A5_size) + mA;
            double tm = x_vals[mA];
            for (size_t oA = 0; oA < A6_size; oA++) {
              size_t pA6 = (pA5 * A6_size) + oA;
              size_t py5 = (py4 * y5_size) + oA;
              double to = tm;
              for (size_t pA = 0; pA < A7_size; pA++) {
                size_t pA7 = (pA6 * A7_size) + pA;
                size_t py6 = (py5 * y6_size) + pA;
                double tp = to;
                for (size_t rA = 0; rA < A8_size; rA++) {
                  size_t pA8 = (pA7 * A8_size) + rA;
                  size_t py7 = (py6 * y7_size) + rA;
                  double tr = tp;
                  for (size_t sA = 0; sA < A9_size; sA++) {
                    size_t pA9 = (pA8 * A9_size) + sA;
                    size_t py8 = (py7 * y8_size) + sA;
                    double ts = tr;
                    for (size_t tA = 0; tA < A10_size; tA++) {
                      size_t pA10 = (pA9 * A10_size) + tA;
                      size_t py9 = (py8 * y9_size) + tA;
                      y_vals[py9] = y_vals[py9] + (A_vals[pA10] * ts);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

		} else if (mode == 5) {

  for (size_t iA = 0; iA < A1_size; iA++) {
    for (size_t jA = 0; jA < A2_size; jA++) {
      size_t pA2 = (iA * A2_size) + jA;
      size_t py2 = (iA * y2_size) + jA;
      for (size_t kA = 0; kA < A3_size; kA++) {
        size_t pA3 = (pA2 * A3_size) + kA;
        size_t py3 = (py2 * y3_size) + kA;
        for (size_t lA = 0; lA < A4_size; lA++) {
          size_t pA4 = (pA3 * A4_size) + lA;
          size_t py4 = (py3 * y4_size) + lA;
          for (size_t mA = 0; mA < A5_size; mA++) {
            size_t pA5 = (pA4 * A5_size) + mA;
            size_t py5 = (py4 * y5_size) + mA;
            for (size_t oA = 0; oA < A6_size; oA++) {
              size_t pA6 = (pA5 * A6_size) + oA;
              double to = x_vals[oA];
              for (size_t pA = 0; pA < A7_size; pA++) {
                size_t pA7 = (pA6 * A7_size) + pA;
                size_t py6 = (py5 * y6_size) + pA;
                double tp = to;
                for (size_t rA = 0; rA < A8_size; rA++) {
                  size_t pA8 = (pA7 * A8_size) + rA;
                  size_t py7 = (py6 * y7_size) + rA;
                  double tr = tp;
                  for (size_t sA = 0; sA < A9_size; sA++) {
                    size_t pA9 = (pA8 * A9_size) + sA;
                    size_t py8 = (py7 * y8_size) + sA;
                    double ts = tr;
                    for (size_t tA = 0; tA < A10_size; tA++) {
                      size_t pA10 = (pA9 * A10_size) + tA;
                      size_t py9 = (py8 * y9_size) + tA;
                      y_vals[py9] = y_vals[py9] + (A_vals[pA10] * ts);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

		} else if (mode == 6) {

  for (size_t iA = 0; iA < A1_size; iA++) {
    for (size_t jA = 0; jA < A2_size; jA++) {
      size_t pA2 = (iA * A2_size) + jA;
      size_t py2 = (iA * y2_size) + jA;
      for (size_t kA = 0; kA < A3_size; kA++) {
        size_t pA3 = (pA2 * A3_size) + kA;
        size_t py3 = (py2 * y3_size) + kA;
        for (size_t lA = 0; lA < A4_size; lA++) {
          size_t pA4 = (pA3 * A4_size) + lA;
          size_t py4 = (py3 * y4_size) + lA;
          for (size_t mA = 0; mA < A5_size; mA++) {
            size_t pA5 = (pA4 * A5_size) + mA;
            size_t py5 = (py4 * y5_size) + mA;
            for (size_t oA = 0; oA < A6_size; oA++) {
              size_t pA6 = (pA5 * A6_size) + oA;
              size_t py6 = (py5 * y6_size) + oA;
              for (size_t pA = 0; pA < A7_size; pA++) {
                size_t pA7 = (pA6 * A7_size) + pA;
                double tp = x_vals[pA];
                for (size_t rA = 0; rA < A8_size; rA++) {
                  size_t pA8 = (pA7 * A8_size) + rA;
                  size_t py7 = (py6 * y7_size) + rA;
                  double tr = tp;
                  for (size_t sA = 0; sA < A9_size; sA++) {
                    size_t pA9 = (pA8 * A9_size) + sA;
                    size_t py8 = (py7 * y8_size) + sA;
                    double ts = tr;
                    for (size_t tA = 0; tA < A10_size; tA++) {
                      size_t pA10 = (pA9 * A10_size) + tA;
                      size_t py9 = (py8 * y9_size) + tA;
                      y_vals[py9] = y_vals[py9] + (A_vals[pA10] * ts);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

		} else if (mode == 7) {

  for (size_t iA = 0; iA < A1_size; iA++) {
    for (size_t jA = 0; jA < A2_size; jA++) {
      size_t pA2 = (iA * A2_size) + jA;
      size_t py2 = (iA * y2_size) + jA;
      for (size_t kA = 0; kA < A3_size; kA++) {
        size_t pA3 = (pA2 * A3_size) + kA;
        size_t py3 = (py2 * y3_size) + kA;
        for (size_t lA = 0; lA < A4_size; lA++) {
          size_t pA4 = (pA3 * A4_size) + lA;
          size_t py4 = (py3 * y4_size) + lA;
          for (size_t mA = 0; mA < A5_size; mA++) {
            size_t pA5 = (pA4 * A5_size) + mA;
            size_t py5 = (py4 * y5_size) + mA;
            for (size_t oA = 0; oA < A6_size; oA++) {
              size_t pA6 = (pA5 * A6_size) + oA;
              size_t py6 = (py5 * y6_size) + oA;
              for (size_t pA = 0; pA < A7_size; pA++) {
                size_t pA7 = (pA6 * A7_size) + pA;
                size_t py7 = (py6 * y7_size) + pA;
                for (size_t rA = 0; rA < A8_size; rA++) {
                  size_t pA8 = (pA7 * A8_size) + rA;
                  double tr = x_vals[rA];
                  for (size_t sA = 0; sA < A9_size; sA++) {
                    size_t pA9 = (pA8 * A9_size) + sA;
                    size_t py8 = (py7 * y8_size) + sA;
                    double ts = tr;
                    for (size_t tA = 0; tA < A10_size; tA++) {
                      size_t pA10 = (pA9 * A10_size) + tA;
                      size_t py9 = (py8 * y9_size) + tA;
                      y_vals[py9] = y_vals[py9] + (A_vals[pA10] * ts);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

		} else if (mode == 8) {

  for (size_t iA = 0; iA < A1_size; iA++) {
    for (size_t jA = 0; jA < A2_size; jA++) {
      size_t pA2 = (iA * A2_size) + jA;
      size_t py2 = (iA * y2_size) + jA;
      for (size_t kA = 0; kA < A3_size; kA++) {
        size_t pA3 = (pA2 * A3_size) + kA;
        size_t py3 = (py2 * y3_size) + kA;
        for (size_t lA = 0; lA < A4_size; lA++) {
          size_t pA4 = (pA3 * A4_size) + lA;
          size_t py4 = (py3 * y4_size) + lA;
          for (size_t mA = 0; mA < A5_size; mA++) {
            size_t pA5 = (pA4 * A5_size) + mA;
            size_t py5 = (py4 * y5_size) + mA;
            for (size_t oA = 0; oA < A6_size; oA++) {
              size_t pA6 = (pA5 * A6_size) + oA;
              size_t py6 = (py5 * y6_size) + oA;
              for (size_t pA = 0; pA < A7_size; pA++) {
                size_t pA7 = (pA6 * A7_size) + pA;
                size_t py7 = (py6 * y7_size) + pA;
                for (size_t rA = 0; rA < A8_size; rA++) {
                  size_t pA8 = (pA7 * A8_size) + rA;
                  size_t py8 = (py7 * y8_size) + rA;
                  for (size_t sA = 0; sA < A9_size; sA++) {
                    size_t pA9 = (pA8 * A9_size) + sA;
                    double ts = x_vals[sA];
                    for (size_t tA = 0; tA < A10_size; tA++) {
                      size_t pA10 = (pA9 * A10_size) + tA;
                      size_t py9 = (py8 * y9_size) + tA;
                      y_vals[py9] = y_vals[py9] + (A_vals[pA10] * ts);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }


		} else if (mode == 9) {

  for (size_t iA = 0; iA < A1_size; iA++) {
    for (size_t jA = 0; jA < A2_size; jA++) {
      size_t pA2 = (iA * A2_size) + jA;
      size_t py2 = (iA * y2_size) + jA;
      for (size_t kA = 0; kA < A3_size; kA++) {
        size_t pA3 = (pA2 * A3_size) + kA;
        size_t py3 = (py2 * y3_size) + kA;
        for (size_t lA = 0; lA < A4_size; lA++) {
          size_t pA4 = (pA3 * A4_size) + lA;
          size_t py4 = (py3 * y4_size) + lA;
          for (size_t mA = 0; mA < A5_size; mA++) {
            size_t pA5 = (pA4 * A5_size) + mA;
            size_t py5 = (py4 * y5_size) + mA;
            for (size_t oA = 0; oA < A6_size; oA++) {
              size_t pA6 = (pA5 * A6_size) + oA;
              size_t py6 = (py5 * y6_size) + oA;
              for (size_t pA = 0; pA < A7_size; pA++) {
                size_t pA7 = (pA6 * A7_size) + pA;
                size_t py7 = (py6 * y7_size) + pA;
                for (size_t rA = 0; rA < A8_size; rA++) {
                  size_t pA8 = (pA7 * A8_size) + rA;
                  size_t py8 = (py7 * y8_size) + rA;
                  for (size_t sA = 0; sA < A9_size; sA++) {
                    size_t pA9 = (pA8 * A9_size) + sA;
                    size_t py9 = (py8 * y9_size) + sA;
                    double tt = 0;
                    for (size_t tA = 0; tA < A10_size; tA++) {
                      size_t pA10 = (pA9 * A10_size) + tA;
                      tt += A_vals[pA10] * x_vals[tA];
                    }
                    y_vals[py9] = tt;
                  }
                }
              }
            }
          }
        }
      }
    }
  }

		}
	}


}
*/

// Assume that n is an exact power of 2
void Rec_Mult(double *C, double *A, double *B, int n, int rowsize) {

  if (n == 1) {
    // printf("Base case\n");
    C[0] += A[0] * B[0];
    // printf("This is the addition of A[0]=%f with B[0]=%f so together C[0]=%f\n", A[0], B[0], C[0]);

  } else {

    int d11 = 0;
    int d12 = n/2;
    int d21 = (n/2) *rowsize;
    int d22 = (n/2) * (rowsize+1);
    Rec_Mult(C+d11, A+d11, B+d11, n/2,rowsize);
    // printf("Store in C[%d] result of multiplying A[%d] with B[%d] \nResult at %d (n/2) iteration:\n", d11, d11, d11, n/2);
    // print_to_console(C+d11, n/2);
    Rec_Mult(C+d11, A+d12, B+d21, n/2,rowsize);
    // printf("Store in C[%d] result of multiplying A[%d] with B[%d] \nResult at %d (n/2) iteration:\n", d11, d12, d21, n/2);
    // print_to_console(C+d11, n/2);
    Rec_Mult(C+d12, A+d11, B+d12, n/2,rowsize);
    // printf("Store in C[%d] result of multiplying A[%d] with B[%d] \nResult at %d (n/2) iteration:\n", d12, d11, d12, n/2);
    // print_to_console(C+d12, n/2);
    Rec_Mult(C+d12, A+d12, B+d22, n/2,rowsize);
    // printf("Store in C[%d] result of multiplying A[%d] with B[%d] \nResult at %d (n/2) iteration:\n", d12, d12, d22, n/2);
    // print_to_console(C+d12, n/2);
    Rec_Mult(C+d21, A+d21, B+d11, n/2,rowsize);
    // printf("Store in C[%d] result of multiplying A[%d] with B[%d] \nResult at %d (n/2) iteration:\n", d21, d21, d11, n/2);
    // print_to_console(C+d21, n/2);
    Rec_Mult(C+d21, A+d22, B+d21, n/2,rowsize);
    // printf("Store in C[%d] result of multiplying A[%d] with B[%d] \nResult at %d (n/2) iteration:\n", d21, d22, d21, n/2);
    // print_to_console(C+d21, n/2);
    Rec_Mult(C+d22, A+d21, B+d12, n/2,rowsize);
    // printf("Store in C[%d] result of multiplying A[%d] with B[%d] \nResult at %d (n/2) iteration:\n", d22, d21, d12, n/2);
    // print_to_console(C+d22, n/2);
    Rec_Mult(C+d22, A+d22, B+d22, n/2,rowsize);
    // printf("Store in C[%d] result of multiplying A[%d] with B[%d] \nResult at %d (n/2) iteration:\n", d22, d22, d22, n/2);
    // print_to_console(C+d22, n/2);
  } 
}

void Mult(double *C, double *A, double *B, int n) {
  for (int i=0;i < n; i++)
    for (int j=0; j < n; j++)
      for (int k=0; k < n; k++) {
        C[i*n+j] += A[i*n+k] * B[k*n+j];
        // printf("Store in C[%d] result of multiplying A[%d] with B[%d]\n", i*n+j, i*n+k, k*n+j);
      }

}

void Tiled_Mult(double *C, double *A, double *B, int n) {

  // printf("Tiled Mult\n");

  int s = 2;

  for (int i1=0; i1 < n; i1+=s) {
    // printf("i1=%d, ", i1);

    for (int j1=0; j1 < n; j1+=s) {
      // printf("j1=%d, ", j1);

      for (int k1=0; k1 < n; k1+=s) {
        // printf("k1=%d\n", k1);

        for (int i=i1; i < i1+s && i<n; i++) {
          // printf("i=%d, ", i);

          for (int j=j1; j<j1+s && j<n; j++) {
            // printf("j=%d\n", j);

            for (int k=k1; k<k1+s && k<n; k++) {
              // printf("k=%d\n", k);

              C[i*n+j] += A[i*n+k] * B[k*n+j];
              // printf("TILED: Store in C[%d] result of multiplying A[%d] with B[%d]\n", i*n+j, i*n+k, k*n+j);

            }
          }
        }
      }
    }
  }

}

void
tvm_test_dgemm(const struct tensor_storage * restrict tenki, const struct lin_storage * restrict vectorski, struct lin_storage * result_tensorski, const size_t modeski) {
    (void) tenki;
    (void) vectorski;
    (void) modeski;
    (void) result_tensorski;

    double alpha = 1;
    double beta = 1;
  // Let's introduce 2x2 matrices (2 of them)

  int k = 4;
  int l = 4;
  int m = 4;
  int total_size = 16;

  double tensor[16] = {2,2,5.0,7.0, 2,3,5.0,7.0, 2,3,5.0,7.0, 2,3,5.0,7.0};

  double result[16] = {0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0};

  int l_dimension = 2; // 4
  int lda = 3; // 3
  int n = 4; // 2

  // libxsmm_dmmfunction kernel1, kernel2, kernel3;

  double matrix_a_row_major[6] = {6, 3, 5, 4, 2, 8};
  double matrix_a_col_major[6] = {6, 5, 2, 3, 4, 8};

  double matrix_b_col_major[8] = {7, 2, 8, 4, 5, 3, 4, 3};
  double matrix_b_row_major[8] = {7, 5, 2, 3, 8, 4, 4, 3};

  // cblas_dgemm(
  // CblasRowMajor, CblasNoTrans, CblasNoTrans, // RowMajor refers to BOTH matrices; None are transposed
  // 3, 4, 2,
  // alpha,
  // matrix_a_row_major, 2,
  // matrix_b_col_major, 4,
  // beta,
  // result, 4);

  // print_to_console(result, total_size);
  // printf("1==================\n");
  // memset(result, total_size, sizeof(double)*total_size);

  // cblas_dgemm(
  // CblasRowMajor, CblasNoTrans, CblasTrans, // RowMajor refers to BOTH matrices; None are transposed
  // 3, 4, 2,
  // alpha,
  // matrix_a_row_major, 2,
  // matrix_b_row_major, 2,
  // beta,
  // result, 4);

  // print_to_console(result, total_size);
  // printf("2==================\n");
  // memset(result, total_size, sizeof(double)*total_size);

  // cblas_dgemm(
  // CblasRowMajor, CblasTrans, CblasNoTrans,
  // 3, 4, 2,
  // alpha,
  // matrix_a_col_major, 3,
  // matrix_b_col_major, 4,
  // beta,
  // result, 4);

  // print_to_console(result, total_size);
  printf("3==================\n");
  memset(result, total_size, sizeof(double)*total_size);

  cblas_dgemm(
  CblasRowMajor, CblasTrans, CblasTrans,
  3, 4, 2,
  alpha,
  matrix_a_col_major, 3,
  matrix_b_row_major, 2,
  beta,
  result, 4);
  
  print_to_console(result, total_size);

  printf("LIBXSMM BOILERPLATE CODE=========================");
  
  // const double beta = 1;
  const char transa = 'N', transb = 'N';
  const libxsmm_blasint batchsize = 1000;
  const libxsmm_blasint m2 = 13;
  const libxsmm_blasint n2 = 5;
  const libxsmm_blasint k2 = 7;
  double* const a = malloc(sizeof(double) * batchsize * m2 * k2);
  double* const b = malloc(sizeof(double) * batchsize * k2 * n2);
  double* const c = malloc(sizeof(double) * m2 * n2);
  libxsmm_blasint ki, i, j;
  const int flags_trans = LIBXSMM_GEMM_FLAGS(transa, transb);
  const int flags_ab = (LIBXSMM_NEQ(0, beta) ? 0 : LIBXSMM_GEMM_FLAG_BETA_0);



  /* determine matrix shape and precision */
  const libxsmm_bitfield prefetch = libxsmm_get_gemm_prefetch(LIBXSMM_PREFETCH_AUTO);
  //const libxsmm_bitfield prefetch = LIBXSMM_GEMM_PREFETCH_NONE;
  const libxsmm_gemm_shape gemm_shape = libxsmm_create_gemm_shape(m2, n2, k2, m2 /*lda*/, k2 /*ldb*/, m2 /*ldc*/, 
        LIBXSMM_DATATYPE(DTYPE), LIBXSMM_DATATYPE(DTYPE), LIBXSMM_DATATYPE(DTYPE), LIBXSMM_DATATYPE(DTYPE) );
  /* generate and dispatch a matrix multiplication kernel */
  const libxsmm_gemmfunction kernel2 = libxsmm_dispatch_gemm(gemm_shape, LIBXSMM_GEMM_FLAG_NONE, prefetch);







  libxsmm_gemm_param gemm_param; /* collect call-arguments into single structure */
  
  assert(NULL != kernel2 && NULL != a && NULL != b && NULL != c);
  // LIBXSMM_UNUSED(argc);
  // LIBXSMM_UNUSED(argv);
  for (i = 0; i < batchsize; ++i) { /* initialize input */
    for (ki = 0; ki < k2; ++ki) {
      for (j = 0; j < m2; ++j) a[i * j * ki] = ((double)1) / ((i + j + ki) % 25);
      for (j = 0; j < n2; ++j) b[i * j * ki] = ((double)7) / ((i + j + ki) % 75);
    }
  }
  memset(c, 0, sizeof(double) * m2 * n2);
  /* kernel multiplies and accumulates matrices: C += Ai * Bi */
  gemm_param.c.primary = c;
  for (i = 0; i < batchsize; ++i) {
    gemm_param.a.primary = (double*)(a + i * m2 * k2);
    gemm_param.b.primary = (double*)(b + i * k2 * n2);
    kernel2(&gemm_param);
  }
  free(a), free(b), free(c);

  printf("4==================\n");

  memset(result, total_size, sizeof(double)*total_size);

	libxsmm_descriptor_blob l_xgemmBlob;
	const libxsmm_gemm_descriptor* l_desc = 0;
	int prefetch = LIBXSMM_PREFETCH_AUTO;

  l_desc = libxsmm_gemm_descriptor_init( &l_xgemmBlob,
                      LIBXSMM_DATATYPE_F32,
                      LIBXSMM_DATATYPE_F32,
                      LIBXSMM_DATATYPE_F32,
                      LIBXSMM_DATATYPE_F32,
                      3, 4, 2,
                      0, 0, 0,
                      0, prefetch);
  // generate and store function for this kernels
  libxsmm_smmfunction kernel = libxsmm_xmmdispatch( l_desc ).dmm;

  // kernel = libxsmm_xmmdispatch(nn, result_size, kk);//, NULL, NULL, NULL, NULL, NULL, NULL, &prefetch);
  // kernel1 = libxsmm_dmmdispatch(4, 3, 2, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
  // kernel1 = libxsmm_xmmdispatch(4, 3, 2);//, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
  kernel(matrix_b_col_major, matrix_a_row_major, result);//, NULL, NULL, NULL);

  print_to_console(result, total_size);

  // printf("5==================\n");
  // memset(result, total_size, sizeof(double)*total_size);

  // const libxsmm_blasint trans1 = 4;
  // const libxsmm_blasint trans2 = 2;
  // const libxsmm_blasint trans3 = 4;
  // memset(result, total_size, sizeof(double)*total_size);
  // // kernel2 = libxsmm_dmmdispatch(4, 3, 2, &trans1, &trans2, &trans3, NULL, NULL, NULL, NULL);
  // kernel2 = libxsmm_xmmdispatch(4, 3, 2);//, &trans1, &trans2, &trans3, NULL, NULL, NULL, NULL);
  // kernel1(matrix_b_col_major, matrix_a_row_major, result);//, NULL, NULL, NULL);

  // print_to_console(result, total_size);
  // // printf("6==================\n");
  // // memset(result, total_size, sizeof(double)*total_size);













  // cblas_dgemm(
  // CblasColMajor, CblasNoTrans, CblasNoTrans, // RowMajor refers to BOTH matrices; None are transposed
  // 3, 4, 2,
  // alpha,
  // matrix_a_col_major, 3,
  // matrix_b_row_major, 2,
  // beta,
  // result, 3);

  // print_to_console(result, total_size);
  // printf("7==================\n");
  // memset(result, total_size, sizeof(double)*total_size);


  // cblas_dgemm(
  // CblasRowMajor, CblasNoTrans, CblasNoTrans, // RowMajor refers to BOTH matrices; None are transposed
  // 4, 3, 2,
  // alpha,
  // matrix_b_row_major, 2,
  // matrix_a_col_major, 3,
  // beta,
  // result, 3);
  
  // print_to_console(result, total_size);
  // printf("FIXED? 7==================\n");
  // memset(result, total_size, sizeof(double)*total_size);

  // cblas_dgemm(
  // CblasColMajor, CblasNoTrans, CblasTrans, // RowMajor refers to BOTH matrices; None are transposed
  // 3, 4, 2,
  // alpha,
  // matrix_a_col_major, 3,
  // matrix_b_col_major, 4,
  // beta,
  // result, 3);

  // print_to_console(result, total_size);
  // printf("8==================\n");
  // memset(result, total_size, sizeof(double)*total_size);

  // cblas_dgemm(
  // CblasColMajor, CblasTrans, CblasNoTrans, // RowMajor refers to BOTH matrices; None are transposed
  // 3, 4, 2,
  // alpha,
  // matrix_a_row_major, 2,
  // matrix_b_row_major, 2,
  // beta,
  // result, 3);

  // print_to_console(result, total_size);
  // printf("9==================\n");
  // memset(result, total_size, sizeof(double)*total_size);

  // cblas_dgemm(
  // CblasColMajor, CblasTrans, CblasTrans, // RowMajor refers to BOTH matrices; None are transposed
  // 3, 4, 2,
  // alpha,
  // matrix_a_row_major, 2,
  // matrix_b_col_major, 4,
  // beta,
  // result, 3);

  // print_to_console(result, total_size);
  // printf("10==================\n");
  // memset(result, total_size, sizeof(double)*total_size);

  // kernel1 = libxsmm_xmmdispatch(3, 4, 2);//, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
  // kernel1(matrix_a_col_major, matrix_b_row_major, result);//, NULL, NULL, NULL);

  // print_to_console(result, total_size);
  // printf("11==================\n");
  // memset(result, total_size, sizeof(double)*total_size);
  // printf("\n\n");

  // // Notes: must use address-of for params (m,n,k) and Trans is enough (can omit!)
  // // Question: does it call MKL internally?
  // const char notrans = 'N';
  // const char trans = 'T';
  // const libxsmm_blasint transH = 3;
  // libxsmm_dgemm(&notrans, &trans,
  //     &transH, &trans3, &trans2, // 3, 4, 2 (same as LIBX above!)
  //     NULL, matrix_a_col_major, NULL, 
  //     matrix_b_col_major, NULL, 
  //     NULL, result, NULL);

  // print_to_console(result, total_size);
  // printf("12==================\n");
  // memset(result, total_size, sizeof(double)*total_size);




















  // double matrix[16] = {2 ,13.0,19,23, 11.0,13.0,19,23, 11.0,13.0,19,23, 11.0,13.0,19,23};


  // // int n = 2; // result_size
  // // int mode_size = 2; // vector_size

  // cblas_dgemm(
  // CblasRowMajor, // const CBLAS_LAYOUT
  // // diff 1
  // CblasNoTrans, CblasNoTrans,
  // // diff 2
  // k,l,m, // 2, n, mode_size, // const MKL_size_t (s)
  // alpha, // const double
  // tensor, m, // const double*, const MKL_size_t
  // matrix, l, // const double*, const MKL_size_t
  // beta, // const float
  // result, l); // const double*, const MKL_size_t

  // print_to_console(result, total_size);
  // memset(result, total_size, sizeof(double)*total_size);
  // print_to_console(result, total_size);

  // cblas_dgemm(
  // CblasRowMajor, // const CBLAS_LAYOUT
  // // diff 1
  // CblasNoTrans, CblasTrans,
  // // diff 2
  // k,l,m, // 2, n, mode_size, // const MKL_size_t (s)
  // alpha, // const double
  // tensor, m, // const double*, const MKL_size_t
  // matrix, l, // const double*, const MKL_size_t
  // beta, // const float
  // result, l); // const double*, const MKL_size_t

  // print_to_console(result, total_size);
  // memset(result, total_size, sizeof(double)*total_size);
  // print_to_console(result, total_size);

  // cblas_dgemm(
  // CblasColMajor, // const CBLAS_LAYOUT
  // // diff 1
  // CblasNoTrans, CblasNoTrans,
  // // diff 2
  // l, k, m, 
  // // n, 2, mode_size, // const MKL_size_t (s)
  // alpha, // const double
  // tensor, l, // const double*, const MKL_size_t
  // matrix, m, // const double*, const MKL_size_t
  // beta, // const float
  // result, l); // const double*, const MKL_size_t

  // print_to_console(result, total_size);
  // memset(result, total_size, sizeof(double)*total_size);
  // print_to_console(result, total_size);

  // cblas_dgemm(
  // CblasColMajor, // const CBLAS_LAYOUT
  // // diff 1
  // CblasNoTrans, CblasTrans,
  // // diff 2
  // l, k, m, 
  // // n, 2, mode_size, // const MKL_size_t (s)
  // alpha, // const double
  // tensor, l, // const double*, const MKL_size_t
  // matrix, m, // const double*, const MKL_size_t
  // beta, // const float
  // result, l); // const double*, const MKL_size_t

  // print_to_console(result, total_size);
  // memset(result, total_size, sizeof(double)*total_size);
  // print_to_console(result, total_size);

  // Rec_Mult(result, tensor, matrix, k, k);
  // print_to_console(result, total_size);
  // memset(result, total_size, sizeof(double)*total_size);
  // print_to_console(result, total_size);

  // Tiled_Mult(result, tensor, matrix, k);
  // print_to_console(result, total_size);
  // memset(result, total_size, sizeof(double)*total_size);
  // print_to_console(result, total_size);

  // Mult(result, tensor, matrix, k);
  // print_to_console(result, total_size);
  // memset(result, total_size, sizeof(double)*total_size);
  // print_to_console(result, total_size);

  // double tensor2[6] = {0,1,2,4,5,1};
  // double matrix2[12] = {1,1,1,1, 2,2,2,2, 3,3,3,3};
  // double matrix3[8] = {1,2,3,4,1,2,5,2};
  // double result2[12] = {0,0,0,0, 0,0,0,0, 0,0,0,0};

  // libxsmm_dmmfunction kernel;

  // k = 2;
  // l = 4;
  // m = 3;
  // total_size = 8;

  // printf("===========================\n\n");

  // // MATHEMATICALLY IT IS
  // // 2x3 and 3x4 matrices

  // cblas_dgemm(
  // CblasRowMajor, // const CBLAS_LAYOUT
  // // diff 1
  // CblasNoTrans, CblasNoTrans,
  // // diff 2
  // k, l, m, // 2, n, mode_size, // const MKL_size_t (s)
  // alpha, // const double
  // tensor2, m, // const double*, const MKL_size_t
  // matrix2, l, // const double*, const MKL_size_t
  // beta, // const float
  // result2, l); // const double*, const MKL_size_t

  // print_to_console(result2, total_size);
  // memset(result2, total_size, sizeof(double)*total_size);
  // print_to_console(result2, total_size);
  // printf("\nNOW LIBX VERSION\n");

  // kernel = libxsmm_dmmdispatch(l, k, m, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
  // kernel(matrix2, tensor2, result2, NULL, NULL, NULL);
  
  // print_to_console(result2, total_size);
  // memset(result2, total_size, sizeof(double)*total_size);
  // print_to_console(result2, total_size);
  // printf("\nNOW ANOTEHR MKL\n");

  // cblas_dgemm(
  // CblasRowMajor, // const CBLAS_LAYOUT
  // // diff 1
  // CblasNoTrans, CblasTrans,
  // // diff 2
  // k, l, m, // 2, n, mode_size, // const MKL_size_t (s)
  // alpha, // const double
  // tensor2, m, // const double*, const MKL_size_t
  // matrix2, m, // const double*, const MKL_size_t
  // beta, // const float
  // result2, l); // const double*, const MKL_size_t

  // print_to_console(result2, total_size);
  // memset(result2, total_size, sizeof(double)*total_size);
  // print_to_console(result2, total_size);
  // printf("\nNOW COL MAJOR\n");

  // // Mathematically, 4x2, 2x3 (because we want to remove from left side)
  // k = 4;
  // l = 3;
  // m = 2;
  // total_size = 12;

  // cblas_dgemm(
  // CblasRowMajor, // const CBLAS_LAYOUT
  // // diff 1
  // CblasNoTrans, CblasNoTrans,
  // // diff 2
  // k, l, m,
  // alpha, // const double
  // matrix3, m, // const double*, const MKL_size_t
  // tensor2, l, // const double*, const MKL_size_t
  // beta, // const float
  // result2, l); // const double*, const MKL_size_t

  // print_to_console(result2, total_size);
  // memset(result2, total_size, sizeof(double)*total_size);
  // print_to_console(result2, total_size);
  // printf("\n");

  // cblas_dgemm(
  // CblasRowMajor, // const CBLAS_LAYOUT
  // // diff 1
  // CblasNoTrans, CblasNoTrans,
  // // diff 2
  // k, l, m,
  // alpha, // const double
  // matrix3, m, // const double*, const MKL_size_t
  // tensor2, m, // const double*, const MKL_size_t
  // beta, // const float
  // result2, l); // const double*, const MKL_size_t

  // print_to_console(result2, total_size);
  // memset(result2, total_size, sizeof(double)*total_size);
  // print_to_console(result2, total_size);
  // printf("\nNOW LIBX\n");

  // // vector always ON THE RIGHT (!) o, so m-> rightmost
  // // 
  // kernel = libxsmm_dmmdispatch(l, k, m, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
  // kernel(tensor2, matrix3, result2, NULL, NULL, NULL);

  // print_to_console(result2, total_size);
  // memset(result2, total_size, sizeof(double)*total_size);
  // print_to_console(result2, total_size);

}

// void
// tvm_tensor_major_mine(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

// 	// size_t next = 0;
		
// 	size_t last_dim_index = tensor->dim-1;
// 	size_t * mul = malloc(tensor->dim * sizeof(size_t));
// 	mul[last_dim_index] = 1;

// 	for (size_t i=last_dim_index; i!=0; --i) {
// 		mul[i-1] = mul[i] * tensor->layout[tensor->layout_perm[i]];
// 		if ((i-1)==mode) break;
// 	}

// 	size_t vector_size = vector->size;
// 	size_t right_size = mul[mode];
// 	size_t result_size = result_tensor->size;
// 	size_t left_size = 0;
// 	left_size = tensor->lin.size / (vector_size * right_size);

// 	size_t t = 0;
// 	if (mode != last_dim_index) {
// 		size_t out_offset = 0;
// 		for (size_t i=0; i<left_size; ++i) {
// 			for (size_t v=0; v<vector_size; ++v) {
// 				for (size_t j=0; j<right_size; ++j) {
// 					result_tensor->data[j+out_offset] += tensor->lin.data[t++] * vector->data[v];
// 				}
// 			}
// 			out_offset += right_size;
// 		}
// 	} else {
// 		for (size_t j=0; j<result_size; ++j) {
// 			for (size_t v=0; v<vector_size; ++v) {
// 				result_tensor->data[j] += tensor->lin.data[t++] * vector->data[v];
// 			}
// 		}
// 	}

// 	free(mul);
// }

// int compare_2( const void* a, const void* b)
// {
//      DTYPE int_a = * ( (DTYPE*) a );
//      DTYPE int_b = * ( (DTYPE*) b );

//      if ( int_a == int_b ) return 0;
//      else if ( int_a < int_b ) return -1; 
//      else return 1;
// }

// ////////////// MODE-INDEPENDENT COMPUTATION: BASE CASES
// // 1. BLAS

// void
// tvm_output_major_BLAS_row_BLAS(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result, const size_t mode) {
//   (void) mode;
//   double alpha = 1;
//   double beta = 1;
//   const MKL_INT incx = 1;
//   const MKL_INT incy = 1;
//   const MKL_INT n = result->size;
//   const MKL_INT mode_size = vector->size;
//   const MKL_INT lda = mode_size;
//   // printf("n=%d, mode_size=%d, lda=%d\n", n, mode_size, lda);
//   // const double * const restrict tensor_ptr = tensor->lin.data;
//   // const double * const restrict vector_ptr = vector->data;
//   // double * const restrict result_ptr = result->data;

//   cblas_dgemv(
//     CblasRowMajor, // const CBLAS_LAYOUT
//     CblasNoTrans, // const CBLAS_TRANSPOSE
//     n, mode_size, // const MKL_size_t (s)
//     alpha, // const double
//     tensor->lin.data, lda, // const double*, const MKL_size_t
//     vector->data, incx, // const double*, const MKL_size_t
//     beta, // const float
//     result->data, incy); // const double*, const MKL_size_t

// }

// void
// tvm_output_major_BLAS_row_GEMM(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result, const size_t mode) {
//       (void) mode;

//         double alpha = 1;
//         double beta = 1;

//         const MKL_INT n = result->size;
//         const MKL_INT mode_size = vector->size;

//         // const MKL_INT lda = n;
//         // const MKL_INT incx = mode_size;
//         // const MKL_INT incx = n;
//         // const MKL_INT incy = n;

//         // BIZZARRE! Below does the vector major version, still
//         // Probably because I switched too many things going back to the same formulation
//         // in the end...
//   //   	cblas_dgemm(
// 		// CblasRowMajor, // const CBLAS_LAYOUT
// 		// // diff 1
// 		// CblasNoTrans, CblasNoTrans,
// 		// // diff 2
// 		// 1, n, mode_size, // const MKL_size_t (s)
// 		// alpha, // const double
// 		// vector->data, mode_size,
// 		// tensor->lin.data, n,
// 		// beta,
// 		// result->data, n);

//     	cblas_dgemm(
// 		CblasRowMajor, // const CBLAS_LAYOUT
// 		// diff 1
// 		CblasNoTrans, CblasNoTrans,
// 		// diff 2
// 		1, n, mode_size, // const MKL_size_t (s)
// 		alpha, // const double
// 		tensor->lin.data, mode_size, // const double*, const MKL_size_t
// 		vector->data, n, // const double*, const MKL_size_t
// 		beta, // const float
// 		result->data, n); // const double*, const MKL_size_t
// }

// void
// tvm_vector_major_BLAS_col_GEMM(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result, const size_t mode) {
//        (void) mode;
//        double alpha = 1;
//         double beta = 1;

//         const MKL_INT n = result->size;
//         const MKL_INT mode_size = vector->size;

//         const MKL_INT lda = n;
//         const MKL_INT incx = mode_size;
//         const MKL_INT incy = n;

//     	cblas_dgemm(
// 		CblasColMajor, // const CBLAS_LAYOUT
// 		// diff 1
// 		CblasNoTrans, CblasNoTrans,
// 		// diff 2
// 		n, 1, mode_size, // const MKL_size_t (s)
// 		alpha, // const double
// 		tensor->lin.data, lda, // const double*, const MKL_size_t
// 		vector->data, incx, // const double*, const MKL_size_t
// 		beta, // const float
// 		result->data, incy); // const double*, const MKL_size_t
// }

// void
// tvm_vector_major_BLAS_col_GEMM_libx(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result, const size_t mode) {
// 	      (void) mode;

//         const int m = result->size;
//         const int n = 1;
//         const int k = vector->size;
//         /* JIT Kernel */
// 	  	libxsmm_dmmfunction kernel = libxsmm_xmmdispatch(m, n, k);//, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
// 	  	if (kernel) {
// 		  	kernel(tensor->lin.data, vector->data, result->data);//, NULL, NULL, NULL);
// 		}

// }

// void
// tvm_output_major_BLAS_row_libx(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result, const size_t mode) {
//       (void) mode;

//         const int m = result->size;
//         const int n = 1;
//         const int k = vector->size;
//         /* JIT Kernel */
// 	  	libxsmm_dmmfunction kernel = libxsmm_xmmdispatch(n, m, k);//, NULL, NULL, NULL, NULL, NULL, NULL, NULL); // libxsmm_dmmdispatch(m, n, k, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
// 	  	if (kernel) {
// 		  	// kernel(tensor->lin.data, vector->data, result->data, NULL, NULL, NULL);
// 		  	kernel(vector->data, tensor->lin.data, result->data);//, NULL, NULL, NULL);
// 		}

// }

// void
// tvm_vector_major_BLAS_col_BLAS(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result, const size_t mode) {
// 	      (void) mode;
// const double alpha = 1;
// 	const double beta = 1;
// 	const MKL_INT incx = 1;
// 	const MKL_INT incy = 1;
// 	const MKL_INT n = result->size;
// 	const MKL_INT mode_size = vector->size;
// 	const MKL_INT lda = n;
// 	cblas_dgemv(
// 		CblasColMajor, // const CBLAS_LAYOUT
// 		CblasNoTrans, // const CBLAS_TRANSPOSE
// 		n, mode_size, // const MKL_size_t (s)
// 		alpha, // const double
// 		tensor->lin.data, lda, // const double*, const MKL_size_t
// 		vector->data, incx, // const double*, const MKL_size_t
// 		beta, // const float
// 		result->data, incy); // const double*, const MKL_size_t
// }

// // 2. BLIS

// void
// tvm_BLIS_row(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result, const size_t mode) {
//     (void) tensor;
//     (void) vector;
//     (void) result;
//     (void) mode;
//  //        double alpha = 1;
//  //        double beta = 1;
//  //        inc_t incx = 1;
//  //        inc_t incy = 1;
// 	// inc_t n = result->size;
// 	// inc_t mode_size = vector->size; // tensor->layout[tensor->layout_perm[dim-1]]; // first dim
// 	// inc_t lda = mode_size;
// #if 0
// 	err_t bli_init(void);
// 	bli_dgemv(
// 		BLIS_NO_TRANSPOSE,
// 		BLIS_NO_CONJUGATE,
// 		n, mode_size, // dim_t (x2)
// 		&alpha, // double* 
// 		tensor->lin.data, // double*
// 		lda, 1, // inc_t rsa, inc_t csa
// 		vector->data, incx, // double*, inc_t incx
// 		&beta, // double* aussi
// 		result->data, incy, // double*, inc_t incy
// 		NULL); // you can ignore this parameter and simply pass in NULL
// 	err_t bli_finalize(void);
// #endif
// }

// void
// tvm_BLIS_col(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result, const size_t mode) {
//     (void) tensor;
//     (void) vector;
//     (void) result;
//     (void) mode;
//  //        double alpha = 1;
//  //        double beta = 1;
//  //        inc_t incx = 1;
//  //        inc_t incy = 1;
// 	// inc_t n = result->size;
//  //        inc_t mode_size = vector->size;
//  //        inc_t lda = n;
// #if 0
// 	err_t bli_init(void);
// 	bli_dgemv(
// 		BLIS_NO_TRANSPOSE,
// 		BLIS_NO_CONJUGATE,
// 		n, mode_size, // dim_t (x2)
// 		&alpha, // double* 
// 		tensor->lin.data, // double*
// 		1, lda, // inc_t rsa, inc_t csa
// 		vector->data, incx, // double*, inc_t incx
// 		&beta, // double* aussi
// 		result->data, incy, // double*, inc_t incy
// 		NULL); // you can ignore this parameter and simply pass in NULL
// 	err_t bli_finalize(void);
// #endif
// }

// // 3. TRANSPOSE-calls

// void
// tvm_vector_major_BLAS_col_BLAS_trans(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result, const size_t mode) {
//   (void) mode;
// 	const double alpha = 1;
// 	const double beta = 1;
// 	const MKL_INT incx = 1;
// 	const MKL_INT incy = 1;
// 	const MKL_INT n = result->size;
// 	const MKL_INT mode_size = vector->size;
// 	const MKL_INT lda = n;
// 	cblas_dgemv(
// 		CblasRowMajor, // const CBLAS_LAYOUT
// 		CblasTrans, // const CBLAS_TRANSPOSE
// 		mode_size, n,
// 		alpha, // const double
// 		tensor->lin.data, lda, // const double*, const MKL_size_t
// 		vector->data, incx, // const double*, const MKL_size_t
// 		beta, // const float
// 		result->data, incy); // const double*, const MKL_size_t
// }

// void
// tvm_output_major_BLAS_row_BLAS_trans(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result, const size_t mode) {
//    (void) mode;
//         double alpha = 1;
//         double beta = 1;
//         const MKL_INT incx = 1;
//         const MKL_INT incy = 1;
//         const MKL_INT n = result->size;
//         const MKL_INT mode_size = vector->size;
//         const MKL_INT lda = mode_size;
//     	cblas_dgemv(
// 		CblasColMajor, // const CBLAS_LAYOUT
// 		CblasTrans, // const CBLAS_TRANSPOSE
// 		mode_size, n,
// 		alpha, // const double
// 		tensor->lin.data, lda, // const double*, const MKL_size_t
// 		vector->data, incx, // const double*, const MKL_size_t
// 		beta, // const float
// 		result->data, incy); // const double*, const MKL_size_t
// }

// ////////////// MODE-INDEPENDENT COMPUTATION:
// // LOOP -> col_major call for biggest matrix where mode is the outer dimension
// // 2) subcases
// // Lopp -> col_BLAS_trans
// // Loop -> col_BLAS

// // void
// // tvm_vector_major_BLIS_col_mode(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result, const size_t mode) {

// //   printf("BLISSing\n");
// // 	const MKL_INT incx = 1; 
// //   const MKL_INT incy = 1; 
// //   double alpha = 1;
// //   double beta = 1;
// //   size_t dim = tensor->dim;
// //   size_t right_size = 1;
  
// //   // for (size_t d=dim-1; d<dim; --d) {
// //   //   if (d < mode) {
// //   //     right_size *= tensor->layout[tensor->layout_perm[d]];
// //   //   }
// //   // }

// //   int temp_primes[27] = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103}; 
// //   int zero_primes[19] = {2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}; 

// //   for (size_t d=dim-1; d<dim; --d) {
// //    if (d > mode) {
// //      right_size *= tensor->layout[tensor->layout_perm[d]];
// //    }
// //   }

// //   // size_t left_size = result->size / right_size;
// //   // const dim_t mode_size = vector->size;
// //   // const dim_t n = left_size;
// //   // size_t left_t_size = tensor->lin.size / right_size;
// //   // size_t next = 0; 
// //   // size_t next_result = 0; 

// //   inc_t n = result->size;
// //   inc_t mode_size = vector->size; // tensor->layout[tensor->layout_perm[dim-1]]; // first dim
// //   // Try to introduce the two options...? For a 3D tensor (based on the stride)

// //   // row-major formulation
// //   if (mode == dim-1) {

// //     inc_t lda = mode_size;
// //     bli_dgemv(
// //       BLIS_NO_TRANSPOSE,
// //       BLIS_NO_CONJUGATE,
// //       n, mode_size, // dim_t (x2)
// //       &alpha, // double* 
// //       tensor->lin.data, // double*
// //       lda, 1, // inc_t rsa, inc_t csa
// //       vector->data, incx, // double*, inc_t incx
// //       &beta, // double* aussi
// //       result->data, incy, // double*, inc_t incy
// //       NULL); // you can ignore this parameter and simply pass in NULL

// //   } else if (mode == 0) {

// //     inc_t lda = n;
// //     bli_dgemv(
// //       BLIS_NO_TRANSPOSE,
// //       BLIS_NO_CONJUGATE,
// //       n, mode_size, // dim_t (x2)
// //       &alpha, // double* 
// //       tensor->lin.data, // double*
// //       1, lda, // inc_t rsa, inc_t csa
// //       vector->data, incx, // double*, inc_t incx
// //       &beta, // double* aussi
// //       result->data, incy, // double*, inc_t incy
// //       NULL); // you can ignore this parameter and simply pass in NULL
    
// //   } else {
  
// //     for (size_t i=0; i<tensor->lin.size; ++i) {
// //       tensor->lin.data[i] = temp_primes[i];
// //     }
// //     for (size_t i=0; i<vector->size; ++i) {
// //       vector->data[i] = zero_primes[i];
// //     }

// //     // inc_t lda = 3;
// //     bli_dgemv(
// //       BLIS_NO_TRANSPOSE,
// //       BLIS_NO_CONJUGATE,
// //       n, mode_size, // dim_t (x2)
// //       &alpha, // double* 
// //       tensor->lin.data, // double*
// //       9, 1, // inc_t rsa, inc_t csa
// //       vector->data, incx, // double*, inc_t incx
// //       &beta, // double* aussi
// //       result->data, incy, // double*, inc_t incy
// //       NULL); // you can ignore this parameter and simply pass in NULL
    
// //     // for (size_t i=0; i<right_size; ++i) {
// //     //   bli_dgemv(
// //     //     BLIS_NO_TRANSPOSE,
// //     //     BLIS_NO_CONJUGATE,
// //     //     n, mode_size, // dim_t (x2)
// //     //     &alpha, // double* 
// //     //     (tensor->lin.data + next), // double*
// //     //     1, left_size,
// //     //     vector->data, 1, // double*, inc_t incx
// //     //     &beta, // double* aussi
// //     //     (result->data + next_result), 1, // double*, inc_t incy
// //     //     NULL); // you can ignore this parameter and simply pass in NULL
// //     //   next += left_t_size;
// //     //   next_result += left_size;
// //     // }

// //   }


// // }

// void
// tvm_vector_major_BLAS_col_mode(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result, const size_t mode) {

//         const double *const tensor_ptr = tensor->lin.data;
//         double *const result_ptr = result->data;

//         const MKL_INT incx = 1; 
//         const MKL_INT incy = 1; 
//         double alpha = 1;
//         double beta = 1;
//         size_t dim = tensor->dim;

// 		    size_t right_size = 1;
// 		    for (size_t d=dim-1; d<dim; --d) {
// 			   if (d > mode) {
//           right_size *= tensor->layout[d];
// 			   }
// 		    }
//         const MKL_INT mode_size = vector->size;
//         const MKL_INT n = right_size;
//         size_t mat_size = (size_t) mode_size * (size_t) n; 
//         size_t left_size = tensor->lin.size / mat_size;
//         const MKL_INT n2 = result->size;
//         if (mode != dim-1) {
//             for (size_t i=0; i<left_size; ++i) {
//                 const double * next = tensor_ptr + i*mat_size;
//                 double * next_result = result_ptr + i*n;
//                 cblas_dgemv(
//                   CblasColMajor, // const CBLAS_LAYOUT
//                   CblasNoTrans, // const CBLAS_TRANSPOSE
//                   n, mode_size, // const MKL_size_t (s)
//                   alpha, // const double
//                   next, n, // const double*, const MKL_size_t
//                   vector->data, incx, // const double*, const MKL_size_t
//                   beta, // const float
//                   next_result, incy); // const double*, const MKL_size_t
//             }
//   	    } else {
//               cblas_dgemv(
//                 CblasRowMajor, // const CBLAS_LAYOUT
//                 CblasNoTrans, // const CBLAS_TRANSPOSE
//                 n2, mode_size,
//                 alpha, // const double
//                 tensor_ptr, mode_size, // const double*, const MKL_size_t
//                 vector->data, incx, // const double*, const MKL_size_t
//                 beta, // const float
//                 result_ptr, incy); // const double*, const MKL_size_t
//         }
// }

// void
// tvm_vector_major_BLAS_col_mode2(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result, const size_t mode) {

//         size_t PUT_MODE;
//         if (mode != tensor->dim-1) {
//           PUT_MODE = mode+1;
//         } else {
//           PUT_MODE = mode;
//         }

//         const MKL_INT incx = 1; 
//         const MKL_INT incy = 1; 
//         double alpha = 1;
//         double beta = 1;
//         size_t dim = tensor->dim;
//         size_t right_size = 1;
//         for (size_t d=dim-1; d<dim; --d) {
//          if (d > PUT_MODE) {
//            right_size *= tensor->layout[tensor->layout_perm[d]];
//          }
//         }
//         const MKL_INT mode_size = tensor->layout[PUT_MODE];
//         const MKL_INT n = right_size;
//         size_t mat_size = (size_t) mode_size * (size_t) n; 
//         size_t left_size = tensor->lin.size / mat_size;
//         size_t next = 0;
//         size_t next_result = 0; 

//         // printf("SPECIAL details: right_size=%zu, left_size=%zu, mode_size=%zu\n",
//         //   right_size, left_size, mode_size);

//         if (mode != dim-1) {

//           for (size_t i=0; i<left_size/mode_size; ++i) {
//             for (int j=0; j<mode_size; ++j) {
//                 cblas_dgemv(
//                         CblasColMajor, // const CBLAS_LAYOUT
//                         CblasNoTrans, // const CBLAS_TRANSPOSE
//                         mat_size, 1, // const MKL_size_t (s)
//                         alpha, // const double
//                         (tensor->lin.data + next), mat_size, // const double*, const MKL_size_t
//                         vector->data+j, incx, // const double*, const MKL_size_t
//                         beta, // const float
//                         (result->data + next_result), incy); // const double*, const MKL_size_t
//                 next += mat_size; // mode_size = 1 so not mat_size...
//                 // print_to_console(result->data+next_result, result->size);
//                 // printf("\n");
//             }
//             next_result += n*mode_size;
//           }

//       } else {
//         // Just a single ROW MAJOR loop call
//         cblas_dgemv(
//         CblasRowMajor, // const CBLAS_LAYOUT
//         CblasNoTrans, // const CBLAS_TRANSPOSE
//         result->size, mode_size, // const MKL_size_t (s)
//         alpha, // const double
//         tensor->lin.data, mode_size, // const double*, const MKL_size_t
//         vector->data, incx, // const double*, const MKL_size_t
//         beta, // const float
//         result->data, incy); // const double*, const MKL_size_t
//       } 
//     }

// void
// tvm_vector_major_BLAS_col_mode_libx(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result, const size_t mode) {

// 		// const int prefetch = LIBXSMM_PREFETCH_AUTO;
// 		libxsmm_dmmfunction kernel;
//     size_t dim = tensor->dim;
// 		size_t right_size = 1;
// 		for (size_t d=dim-1; d<dim; --d) {
// 			if (d > mode) {
// 				right_size *= tensor->layout[tensor->layout_perm[d]];
// 			}
// 		}
//     const int nn = 1;
//     const int kk = vector->size;
//     size_t mat_size = (size_t) kk * (size_t) right_size; 
//     size_t left_size = tensor->lin.size / mat_size;
//     const int result_size = result->size;
//     const double *const tensor_ptr = tensor->lin.data;
//     const double *const vector_ptr = vector->data;
// 	  double *const result_ptr = result->data;
//     if (mode != dim-1) {
//   		kernel = libxsmm_xmmdispatch(right_size, nn, kk);//, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
//   	} else {
//   		kernel = libxsmm_xmmdispatch(nn, result_size, kk);//, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
//   	}
//     if (!kernel) {
//       printf("CRASH!!! - LIBX kernel not available!\n");
//     }
//     if (mode != dim-1) {
//         // #pragma omp parallel for
//         for (size_t i=0; i<left_size; ++i) {
//             const double * next = tensor_ptr + i*mat_size;
//             double * next_result = result_ptr + i*right_size;
//             kernel(next, vector_ptr, next_result);//, NULL, NULL, NULL);
//         }
//     } else {
//     	kernel(vector_ptr, tensor_ptr, result_ptr);//, NULL, NULL, NULL);
//     }        
// }

// ////////////// MODE-INDEPENDENT COMPUTATION:
// // Two phase:
// // 1) Loop over the biggest matrices and perform an in-place UNFOLD to have row_major
// // 2) row_BLAS

// void
// tvm_output_major_BLAS_row(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result, const size_t mode) {
// 	const MKL_INT incy = 1;
// 	const MKL_INT incx = 1;
// 	const double beta = 1;
// 	const double alpha = 1;

// 	const size_t dim = tensor->dim;
// 	size_t right_size = 1;
// 	for (size_t i=dim-1; i>mode; --i) {
// 		right_size *= tensor->layout[tensor->layout_perm[i]];
// 	}
// 	const size_t mode_size = vector->size;

// 	const MKL_INT n = result->size;
// 	const MKL_INT lda = mode_size;

// 	if (mode != dim-1) {
// 		if (mode == 0) {
// 			mkl_dimatcopy(
// 				'R', 'T', mode_size, n,
// 				alpha, tensor->lin.data,
// 				n, mode_size);
// 		} else {
// 			size_t mat_size = right_size * mode_size;
// 			size_t left_size = tensor->lin.size / mat_size;
// 			for (size_t i=0; i<left_size; ++i) {
// 				mkl_dimatcopy(
// 					'R', 'T', mode_size, right_size,
// 					alpha, tensor->lin.data + (mat_size*i),
// 					right_size, mode_size);
// 			}
// 		}
// 	}

// 	cblas_dgemv(
// 		CblasRowMajor, // const CBLAS_LAYOUT
// 		CblasNoTrans, // const CBLAS_TRANSPOSE
// 		n, lda, // const MKL_size_t (s)
// 		alpha, // const double
// 		tensor->lin.data, lda, // const double*, const MKL_size_t
// 		vector->data, incx, // const double*, const MKL_size_t
// 		beta, // const float
// 		result->data, incy); // const double*, const MKL_size_t
// }

// void
// tvm_output_major_BLAS_row_onecall(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result, const size_t mode) {

// 	const MKL_INT incy = 1;
// 	const MKL_INT incx = 1;
// 	const double beta = 1;
// 	const double alpha = 1;

// 	const size_t dim = tensor->dim;
// 	size_t right_size = 1;
// 	for (size_t i=dim-1; i>mode; --i) {
// 		right_size *= tensor->layout[tensor->layout_perm[i]];
// 	}

// 	const MKL_INT n = result->size;
// 	const MKL_INT mode_size = vector->size;
// 	const MKL_INT lda = mode_size;

// 	// Matrix: tensor/right * right
// 	size_t left_size = tensor->lin.size / right_size;
// 	if (mode != dim-1) {
// 		mkl_dimatcopy(
// 			'R', 'T', left_size, right_size,
// 			alpha, tensor->lin.data,
// 			right_size, left_size);
// 	}

// 	cblas_dgemv(
// 		CblasRowMajor, // const CBLAS_LAYOUT
// 		CblasNoTrans, // const CBLAS_TRANSPOSE
// 		n, mode_size, // const MKL_size_t (s)
// 		alpha, // const double
// 		tensor->lin.data, lda, // const double*, const MKL_size_t
// 		vector->data, incx, // const double*, const MKL_size_t
// 		beta, // const float
// 		result->data, incy); // const double*, const MKL_size_t
// }

// void
// tvm_output_major_BLAS_row_unfold(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result, const size_t mode) {
// 	const MKL_INT incy = 1;
// 	const MKL_INT incx = 1;
// 	const double beta = 1;
// 	const double alpha = 1;

// 	const size_t dim = tensor->dim;
// 	size_t right_size = 1;
// 	for (size_t i=dim-1; i>mode; --i) {
// 		right_size *= tensor->layout[tensor->layout_perm[i]];
// 	}
// 	const size_t mode_size = vector->size;

// 	const MKL_INT n = result->size;
// 	const MKL_INT lda = mode_size;

// 	// ONLY IN NON-BENCH VERSION
// 	DTYPE* unfold = calloc(tensor->lin.size, sizeof(DTYPE));
// 	//for (size_t i=0; i<tensor->lin.size; ++i) {
// 		//unfold[i] = tensor->lin.data[i];
// 	//}
// 	//if (mode != dim-1) {
// 		if (mode == 0) {
// 			mkl_domatcopy(
// 				'R', 'T', mode_size, n,
// 				alpha, tensor->lin.data,
// 				n, unfold, mode_size);
// 		} else {
// 			size_t mat_size = right_size * mode_size;
// 			size_t left_size = tensor->lin.size / mat_size;
// 			for (size_t i=0; i<left_size; ++i) {
// 				mkl_domatcopy(
// 					'R', 'T', mode_size, right_size,
// 					alpha, tensor->lin.data + (mat_size*i),
// 					right_size, unfold + (mat_size*i), mode_size);
// 			}
// 		}
// 	//}

// 	cblas_dgemv(
// 		CblasRowMajor, // const CBLAS_LAYOUT
// 		CblasNoTrans, // const CBLAS_TRANSPOSE
// 		n, lda, // const MKL_size_t (s)
// 		alpha, // const double
// 		unfold, lda, // const double*, const MKL_size_t
// 		vector->data, incx, // const double*, const MKL_size_t
// 		beta, // const float
// 		result->data, incy); // const double*, const MKL_size_t

// 	free(unfold);
// }

// void
// tvm_output_major_BLAS_row_onecall_unfold(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result, const size_t mode) {

// 	const MKL_INT incy = 1;
// 	const MKL_INT incx = 1;
// 	const double beta = 1;
// 	const double alpha = 1;

// 	const size_t dim = tensor->dim;
// 	size_t right_size = 1;
// 	for (size_t i=dim-1; i>mode; --i) {
// 		right_size *= tensor->layout[tensor->layout_perm[i]];
// 	}

// 	const MKL_INT n = result->size;
// 	const MKL_INT mode_size = vector->size;
// 	const MKL_INT lda = mode_size;

// 	// ONLY IN NON-BENCH VERSION
// 	DTYPE* unfold = calloc(tensor->lin.size, sizeof(DTYPE));
// 	//for (size_t i=0; i<tensor->lin.size; ++i) {
// 		//unfold[i] = tensor->lin.data[i];
// 	//}

// 	// Matrix: tensor/right * right
// 	size_t left_size = tensor->lin.size / right_size;
// 	//if (mode != dim-1) {
// 	mkl_domatcopy(
// 		'R', 'T', left_size, right_size,
// 		alpha, tensor->lin.data,
// 		right_size, unfold, left_size);
// 	//}

// 	cblas_dgemv(
// 		CblasRowMajor, // const CBLAS_LAYOUT
// 		CblasNoTrans, // const CBLAS_TRANSPOSE
// 		n, mode_size, // const MKL_size_t (s)
// 		alpha, // const double
// 		unfold, lda, // const double*, const MKL_size_t
// 		vector->data, incx, // const double*, const MKL_size_t
// 		beta, // const float
// 		result->data, incy); // const double*, const MKL_size_t

// 	free(unfold);
// }

// ////////////// OTHER FUNCTIONS

// void
// tvm_output_major_BLAS_row_UNFOLD(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result, const size_t mode) {

//     const size_t dim = tensor->dim;

//     size_t right_size = 1;
//     size_t stride = 0;
//     for (size_t i=dim-1; i>mode; --i) {
// 	right_size *= tensor->layout[tensor->layout_perm[i]];
//     }
//     if (mode != 0) {
//         stride = right_size * (vector->size - 1);
//     }

//     size_t mode_size = vector->size; // tensor->layout[mode];

//     DTYPE * unfold = malloc(tensor->lin.size * sizeof(DTYPE));

//     size_t result_size = tensor->lin.size / mode_size;
//     size_t tensor_diff = 0;
//     size_t tensor_index = 0;
//     size_t next = 0;

//     for (size_t i=0; i<result_size; ++i) {
//         if ((i!=0) & (i % right_size == 0)) {
//             tensor_diff += stride;
//         }
//         tensor_index = i + tensor_diff;
//         for (size_t j=0; j<mode_size; ++j) {
//             unfold[next++] = tensor->lin.data[tensor_index];
//             tensor_index += right_size;
//         }
//     }

//     result->data[0] = unfold[0];
//     free(unfold);

// }

// void
// tvm_vector_major_BLAS_col_UNFOLD(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result, const size_t mode) {

//     const size_t dim = tensor->dim;

//     size_t right_size = 1;
//     size_t stride = 0;
//     for (size_t i=dim-1; i>mode; --i) {
//         right_size *= tensor->layout[i];
//     }
//     if (mode != 0) {
//         stride = right_size * (tensor->layout[mode] - 1);
//     }

//     size_t mode_size = vector->size; // tensor->layout[mode];
//     DTYPE * unfold = malloc(tensor->lin.size * sizeof(DTYPE));
//     size_t result_size = tensor->lin.size / mode_size;
//     size_t tensor_diff = 0;
//     size_t tensor_index = 0;
//     size_t next = 0;
// 	//if (mode != dim-1) {
// 	for (size_t j=0; j<mode_size; ++j) {
// 		tensor_index = tensor_diff;
// 		for (size_t i=0; i<result_size; ++i) {
// 			if ((i!=0) & (i % right_size == 0)) {
// 				tensor_index += stride;
// 			}
// 			unfold[next++] = tensor->lin.data[tensor_index + i];
// 		}
// 		tensor_diff += right_size;
// 	}
// 	//}
//     result->data[0] = unfold[0];
//     free(unfold);

// }

// void
// tvm_vector_major_BLAS_col(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result, const size_t mode) {

// 	const size_t dim = tensor->dim;

// 	size_t right_size = 1;
// 	// size_t stride = 0;
// 	// size_t stride2 = 0;
// 	for (size_t i=dim-1; i>mode; --i) {
// 		right_size *= tensor->layout[tensor->layout_perm[i]];
// 	}
// 	size_t left_size = 1;
// 	for (size_t i=0; i<mode; ++i) {
// 		left_size *= tensor->layout[tensor->layout_perm[i]];
// 	}

// 	// if (mode != 0) {
// 	// 	stride = right_size * (tensor->layout[tensor->layout_perm[mode]] - 1);	
// 	// } else {
// 	// 	stride = 0;
// 	// }

// 	const MKL_INT n = result->size;
// 	const MKL_INT mode_size= vector->size;
// 	const MKL_INT lda = n;

// 	size_t result_size = tensor->lin.size / mode_size;
// 	// size_t tensor_diff = 0;
// 	// size_t tensor_index = 0;
// 	// size_t next = 0;

// 	DTYPE * unfold = malloc(tensor->lin.size * sizeof(DTYPE));
// 	DTYPE * const unfold_base = unfold;
// 	DTYPE * tensor_ptr = tensor->lin.data;

// 	for (size_t i=0; i<left_size; ++i) {
// 		for (int j=0; j<mode_size; ++j) {
// 			CopyWithSSEPrefetchNT(unfold + j*result_size, tensor_ptr, right_size);
// 			tensor_ptr += right_size;
// 		}
// 		unfold += right_size;
// 	}
// 	unfold = unfold_base;

// 	// for (size_t j=0; j<(size_t) mode_size; ++j) {
// 	// 	tensor_index = tensor_diff;
// 	// 	for (size_t i=0; i<result_size; ++i) {
// 	// 		if ((i!=0) & (i % right_size == 0)) {
// 	// 			tensor_index += stride;
// 	// 		}
// 	// 		unfold[next++] = tensor->lin.data[tensor_index + i];
// 	// 	}
// 	// 	tensor_diff += right_size;
// 	// }

// 	const double alpha = 1;
// 	const double beta = 1;
// 	const MKL_INT incx = 1;
// 	const MKL_INT incy = 1;

// 	cblas_dgemv(
// 		CblasColMajor, // const CBLAS_LAYOUT
// 		CblasNoTrans, // const CBLAS_TRANSPOSE
// 		n, mode_size, // const MKL_size_t (s)
// 		alpha, // const double
// 		unfold, lda, // const double*, const MKL_size_t
// 		vector->data, incx, // const double*, const MKL_size_t
// 		beta, // const float
// 		result->data, incy); // const double*, const MKL_size_t

// 	free(unfold);
// }

// ///////////////////////////////////////////////////////////////////////////////
// // OUTPUT-MAJOR

// // non-obligatory misses on the vector (unless it fits size_to cache)
// void
// tvm_output_major_input_aligned_vectorized(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result) {

// 	const size_t result_size = result->size;
// 	const size_t vector_size = vector->size;
// 	size_t next = 0;

// 	DTYPE * restrict vector_ptr = vector->data;
// 	DTYPE * restrict result_ptr = result->data;
// 	DTYPE * restrict tensor_ptr = tensor->lin.data;

// 	DTYPE vector_buffer [4];
// 	DTYPE tensor_buffer [4];

// 	size_t approx = (vector_size / 4) * 4;
	
// 	if (vector_size < 4) {

// 		for(size_t i=0; i<result_size; ++i) {
// 			for (size_t j=0; j<vector_size; ++j, ++next) {
// 					result_ptr[i] += tensor_ptr[next] * vector_ptr[j];
// 				}
// 			}

// 	} else {

// 		for(size_t i=0; i<result_size; ++i) {

// 			// do full multiples of vector_size
// 			for (size_t j=0; j<approx; j+=4) {

// 				// copy into buffers
// 				for (int b=0; b<4; ++b, ++next) {
// 					vector_buffer[b] = vector_ptr[j+b];
// 					tensor_buffer[b] = tensor_ptr[next];
// 				}

// 				// compute with buffers
// 				for (int b=0; b<4; ++b) {
// 					result_ptr[i] += tensor_buffer[b] * vector_buffer[b];
// 				}
// 			}

// 			// do the remainder
// 			for (size_t j=0; j<vector_size%4; ++j, ++next) {
// 				result_ptr[i] += tensor_ptr[next] * vector_ptr[approx+j];
// 			}

// 		}

// 	}

// }

// // non-obligatory misses on the vector (unless it fits size_to cache)
// void
// tvm_output_major_input_aligned(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result) {
// 	size_t result_size = result->size;
// 	size_t vector_size = vector->size;
// 	size_t next = 0;
// 	// __m128d res, vec, unf;
// 	for(size_t i=0; i<result_size; ++i) {
// 		for (size_t j=0; j<vector_size; ++j) {
// 			//printf("res[%ld] += ten[%ld] * vec[%ld]\n", i, next, j);
// 			result->data[i] += tensor->lin.data[next++] * vector->data[j];
// 		}
// 	}
// 	// for(size_t i=0; i<result_size; i+=2) {
// 	// 	res = _mm_loadu_pd(result->data+i);
// 	// 	for (size_t j=0; j<vector_size; j+=2) {
// 	// 		vec = _mm_loadu_pd(vector->data+j);
// 	//  		unf = _mm_loadu_pd((tensor->lin.data)+(next));
// 	// 		res = _mm_add_pd(res, _mm_mul_pd(vec, unf));
// 	// 		res+2 = _mm_add_pd(res, _mm_mul_pd(vec, unf));
// 	// 		next+=2;
// 	// 	}
// 	// 	_mm_storeu_pd(result->data+i, res);
// 	// 	_mm_storeu_pd(result->data+i, res);
// 	// }
// }

// // non-obligatory misses on vector
// void
// tvm_output_major(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {
// 	size_t last_dim_index = tensor->dim-1;
// 	size_t * mul = malloc(tensor->dim * sizeof(size_t));
// 	size_t * diff = calloc(tensor->dim, sizeof(size_t));

// 	// TO DO: we only need [mode] so stop at mode
// 	// INIT STEP
// 	mul[last_dim_index] = 1;
// 	// N-STEP
// 	for (size_t i=last_dim_index; i!=0; --i) {
// 		mul[i-1] = mul[i] * tensor->layout[tensor->layout_perm[i]];
// 		//printf("mul[%d] = %d\n", i-1, mul[i-1]);
// 		diff[i] = mul[i-1] - mul[i];
// 		//printf("diff[%d] = %d\n", i, diff[i]);
// 		if (i==mode) break;
// 	}
	
// 	size_t vector_size = vector->size;
// 	size_t right_size = mul[mode];
// 	size_t result_size = tensor->lin.size / vector_size;
// 	size_t stride = diff[mode];

// 	size_t tensor_diff = 0;
// 	size_t tensor_index = 0;

// 	//printf("result_size=%d, vector_size=%d, right_size=%d\n", result_size, vector_size, right_size);
// 	// OUTPUT: this loop ensures we access the result_tensor linearly
// 	for (size_t i=0; i<result_size; ++i) {
// 		if ((i!=0) & (i % right_size == 0)) {
// 			tensor_diff += stride;
// 		}

// 		tensor_index = i + tensor_diff;
// 		// VECTOR: this loop ensures we exhaust all elements of the vector
// 		for (size_t j=0; j<vector_size; ++j) {
// 			// MULTIPLICATION OPERATION (identical for both)
// 			//printf("out2 vec=%d, out=%d, tensor=%d\n", j, i, tensor_index);
// 			result_tensor->data[i] += tensor->lin.data[tensor_index] * vector->data[j];
// 			// printf("tensor_index=%zu\n", tensor_index);
// 			// vector loop must increment tensor_index by mul[mode]
// 			tensor_index += mul[mode];
// 		}
// 		//if (++i >= result_size) {
// 			//break;
// 		//}
// 		//tensor_index =  i;
// 		//if (i % mul[mode] == 0) { // INSIGHT: in fact this is equivalent to a single counter
// 			//tensor_diff += diff[mode];
// 			//tensor_index += diff[mode];
// 		//}
// 		//printf("tensor_diff=%d\n", tenso

// 	}
// 	free(diff);
// 	free(mul);
// }

// // non-obligatory misses on vector
// void
// tvm_output_major_2(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {
// 	size_t last_dim_index = tensor->dim-1;
// 	size_t * mul = malloc(tensor->dim * sizeof(size_t));
// 	size_t * diff = calloc(tensor->dim, sizeof(size_t));

// 	// TO DO: we only need [mode] so stop at mode
// 	// INIT STEP
// 	mul[last_dim_index] = 1;
// 	// N-STEP
// 	for (size_t i=last_dim_index; i!=0; --i) {
// 		mul[i-1] = mul[i] * tensor->layout[tensor->layout_perm[i]];
// 		//printf("mul[%d] = %d\n", i-1, mul[i-1]);
// 		diff[i] = mul[i-1] - mul[i];
// 		//printf("diff[%d] = %d\n", i, diff[i]);
// 		if (i==mode) break;
// 	}
	
// 	size_t vector_size = vector->size;
// 	size_t result_size = tensor->lin.size / vector_size;

// 	size_t tensor_diff = 0;
// 	size_t tensor_index = 0;

// 	// OUTPUT: this loop ensures we access the result_tensor linearly
// 	for (size_t i=0;;) {
// 		// VECTOR: this loop ensures we exhaust all elements of the vector
// 		for (size_t j=0; j<vector_size; ++j) {
// 			// MULTIPLICATION OPERATION (identical for both)
// 			//printf("out vec=%d, out=%d, tensor=%d\n", j, i, tensor_index);
// 			result_tensor->data[i] += tensor->lin.data[tensor_index] * vector->data[j];
// 			// vector loop must increment tensor_index by mul[mode]
// 			tensor_index += mul[mode];
// 		}
// 		if (++i >= result_size) {
// 			break;
// 		}
// 		//tensor_index =  i;
// 		if (i % mul[mode] == 0) { // INSIGHT: in fact this is equivalent to a single counter
// 			tensor_diff += diff[mode];
// 			//tensor_index += diff[mode];
// 		}
// 		//printf("tensor_diff=%d\n", tensor_diff);
// 		tensor_index = i + tensor_diff;
// 	}
// 	free(diff);
// 	free(mul);
// }

// ///////////////////////////////////////////////////////////////////////////////
// // VECTOR-MAJOR

// // non-obligatory misses on the result (unless it fits size_to cache)
// void
// tvm_vector_major_input_aligned(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result) {
// 	size_t result_size = result->size;
// 	size_t vector_size = vector->size;
// 	size_t next = 0;
// 	for(size_t j=0; j<vector_size; ++j) {
// 		for (size_t i=0; i<result_size; ++i) {
// 			// printf("res[%ld] += ten[%ld] * vec[%ld]\n", i, next, j);
// 			result->data[i] += tensor->lin.data[next++] * vector->data[j];
//       // printf("ten * vec = res IS BASICALLY %f * %f = %f\n", tensor->lin.data[next-1], vector->data[j], result->data[i]);
// 		}
// 	}	
// }

// // improved version: fix the loop to be a loop
// void
// tvm_vector_major(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {
// 	// printf("what's wreong with me?\n");

//           // printf("RESULT BEFORE:\n");
//           // print_to_console(result_tensor->data, 10);

//   size_t last_dim_index = tensor->dim-1;
// 	size_t * mul = malloc(tensor->dim * sizeof(size_t));
// 	size_t * diff = calloc(tensor->dim, sizeof(size_t));
	
// 	mul[last_dim_index] = 1;
// 	for (size_t i=last_dim_index; i!=0; --i) {
// 		mul[i-1] = mul[i] * tensor->layout[tensor->layout_perm[i]];
// 		diff[i] = mul[i-1] - mul[i];
// 		//printf("mul[%d] = %d\n", i-1, mul[i-1]);
// 		//printf("diff[%d] = %d\n", i, diff[i]);
// 		if (i==mode) break;
// 	}
	
// 	size_t vector_size = vector->size;
// 	size_t result_size = tensor->lin.size / vector_size;
// 	// right_size + left_size = result_size
// 	size_t right_size = mul[mode];
// 	size_t stride = diff[mode];

// 	size_t tensor_diff = 0;
// 	size_t tensor_index = 0;

// 	//printf("result_size=%d, vector_size=%d, right_size=%d\n", result_size, vector_size, right_size);
// 	// VECTOR: this loop ensures we access the vector linearly
// 	for (size_t j=0; j<vector_size; ++j) {
// 		// we must reset the tensor_index (as the inner loop will jump out)
// 		tensor_index = tensor_diff;
// 		// OUTPUT: this loop ensures we exhaust all elements of the output
// 		for (size_t i=0; i<result_size; ++i) {
// 			// MULTIPLICATION OPERATION (same code for all algos)
// 			//tensor_index += ((i)/right_size)*stride; -> cannot keep adding (adding was supposed to be limited)
// 			if ((i!=0) & (i % right_size == 0)) {
// 				tensor_index += stride;
// 			}
// 			//tensor_index += i + i/right_size*stride;
// 			// printf("vec2 vec=%d, out=%d, tensor=%d\n", j, i, tensor_index + i);
// 			// printf("result[%d] = tensor[%d] * vector[%d]\n", i, tensor_index+i, j);
// 			result_tensor->data[i] += tensor->lin.data[tensor_index + i] * vector->data[j];
// 			//++tensor_index;
// 			//if (i % mul[mode] == 0) {
// 				//tensor_index += diff[mode];
// 			//}
// 			//++tensor_index;
// 		}
// 		tensor_diff += right_size;
// 	}
// 	free(diff);
// 	free(mul);

//           //   printf("RESULT AFTER:\n");
//           // print_to_console(result_tensor->data, 10);

// }

// // non-obligatory misses on output
// void
// tvm_vector_major_2(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {
// 	size_t last_dim_index = tensor->dim-1;
// 	size_t * mul = malloc(tensor->dim * sizeof(size_t));
// 	size_t * diff = calloc(tensor->dim, sizeof(size_t));
// 	// diff[0] = 0; (calloc)
	
// 	mul[last_dim_index] = 1;
// 	for (size_t i=last_dim_index; i!=0; --i) {
// 		mul[i-1] = mul[i] * tensor->layout[tensor->layout_perm[i]];
// 		diff[i] = mul[i-1] - mul[i];
// 		//printf("mul[%d] = %d\n", i-1, mul[i-1]);
// 		//printf("diff[%d] = %d\n", i, diff[i]);
// 		if (i==mode) break;
// 	}
	
// 	size_t vector_size = vector->size;
// 	size_t result_size = tensor->lin.size / vector_size;

// 	size_t tensor_diff = 0;
// 	size_t tensor_index = 0;

// 	// VECTOR: this loop ensures we access the vector linearly
// 	for (size_t j=0; j<vector_size; ++j) {
// 		// we must reset the tensor_index (as the inner loop will jump out)
// 		tensor_index = tensor_diff;
// 		// OUTPUT: this loop ensures we exhaust all elements of the output
// 		for (size_t i=0;;) {
// 			// MULTIPLICATION OPERATION (same code for all algos)
// 			// printf("vec vec=%d, out=%d, tensor=%d\n", j, i, tensor_index);
// 			result_tensor->data[i] += tensor->lin.data[tensor_index] * vector->data[j];
// 			if (++i >= result_size) {
// 				break;
// 			}
// 			if (i % mul[mode] == 0) {
// 				tensor_index += diff[mode];
// 			}
// 			++tensor_index;
// 		}
// 		// vector loop must increment tensor_index by mul[mode]
// 		//tensor_index = j + mul[mode];
// 		tensor_diff += mul[mode];
// 	}

// 	free(diff);
// 	free(mul);
// }

// ///////////////////////////////////////////////////////////////////////////////
// // TENSOR-MAJOR

// // take this guy!
void
tvm_tensor_major(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {

	//result_tensor->data = malloc(sizeof(size_t size_t) * result_tensor->size);
	size_t last_dim_index = tensor->dim-1;
	size_t * mul = malloc(tensor->dim * sizeof(size_t));
	mul[last_dim_index] = 1;

	for (size_t i=last_dim_index; i!=0; --i) {
		mul[i-1] = mul[i] * tensor->layout[tensor->layout_perm[i]];
		//printf("mul[%d] = %d\n", i-1, mul[i-1]);
		if ((i-1)==mode) break;
	}

	size_t vector_size = vector->size;
	size_t right_size = mul[mode];
	size_t left_size = 0;
	left_size = tensor->lin.size / (vector_size * right_size);

	size_t t = 0;
	size_t out_offset = 0;
	for (size_t i=0; i<left_size; ++i) {
		for (size_t v=0; v<vector_size; ++v) {
			for (size_t j=0; j<right_size; ++j) {
				//printf("res[%d] = ten[%d] * vec[%d]\n", j+out_offset, t, v);
				result_tensor->data[j+out_offset] += tensor->lin.data[t++] * vector->data[v];
			}
		}
		out_offset += right_size;
	}

	//prsize_t_to_console(result_tensor->data, result_tensor->size);
	free(mul);
}

// simply use three loops and get rid of the counter methods
// // thus far the most efficient implementation
// void
// tvm_tensor_major_2(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {
// 	size_t last_dim_index = tensor->dim-1;
// 	size_t * mul = malloc(tensor->dim * sizeof(size_t));
// 	mul[last_dim_index] = 1;
// 	for (size_t i=last_dim_index; i!=0; --i) {
// 		mul[i-1] = mul[i] * tensor->layout[tensor->layout_perm[i]];
// 		//printf("mul[%d] = %d\n", i-1, mul[i-1]);
// 		if ((i-1)==mode) break;
// 	}
// 	size_t vector_size = vector->size;
// 	size_t right_size = mul[mode];
// 	//size_t left_size = tensor->lin.size / (vector_size * right_size);
// 	size_t tensor_size = tensor->lin.size;
// 	//printf("left=%d, mode=%d, right=%d\n", left_size, vector_size, right_size);
// 	size_t t = 0;
// 	size_t v = 0;
// 	size_t out_offset = 0;
// 	for (;;) {
// 		for (size_t j=0; j<right_size; ++j) {
// 			//printf("vec vec=%d, out=%d, tensor=%d\n", v, j+out_offset, t);
// 			result_tensor->data[j+out_offset] += tensor->lin.data[t++] * vector->data[v];
// 		}
// 		if (++v == vector_size) {
// 			if (t==tensor_size) break;
// 			v=0;
// 			out_offset += right_size;
// 		}
// 	}
// 	free(mul);

// }

// // simply use three loops and get rid of the counter methods
// // thus far the most efficient implementation
// void
// tvm_tensor_major_noloop(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {
// 	size_t last_dim_index = tensor->dim-1;
// 	size_t * mul = malloc(tensor->dim * sizeof(size_t));
// 	mul[last_dim_index] = 1;
// 	for (size_t i=last_dim_index; i!=0; --i) {
// 		mul[i-1] = mul[i] * tensor->layout[tensor->layout_perm[i]];
// 		//printf("mul[%d] = %d\n", i-1, mul[i-1]);
// 		if ((i-1)==mode) break;
// 	}
// 	size_t vector_size = vector->size;
// 	size_t right_size = mul[mode];
// 	//size_t left_size = tensor->lin.size / (vector_size * right_size);
// 	size_t tensor_size = tensor->lin.size;
// 	//printf("left=%d, mode=%d, right=%d\n", left_size, vector_size, right_size);
// 	size_t t = 0;
// 	size_t v = 0;
// 	size_t out_offset = 0;
// 	for (;;) {
// 		for (size_t j=0; j<right_size; ++j) {
// 			//printf("vec vec=%d, out=%d, tensor=%d\n", v, j+out_offset, t);
// 			result_tensor->data[j+out_offset] += tensor->lin.data[t++] * vector->data[v];
// 		}
// 		if (++v == vector_size) {
// 			if (t==tensor_size) break;
// 			v=0;
// 			out_offset += right_size;
// 		}
// 	}
// 	free(mul);

// }

// // counter-implementation
// void
// tvm_tensor_major_3(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {
// 	// create a special multiplication table
// 	size_t last_dim_index = tensor->dim-1;
// 	size_t * mul = malloc(tensor->dim * sizeof(size_t));
// 	size_t * counter = calloc(tensor->dim, sizeof(size_t));

// 	// recursively
// 	// init step
// 	mul[last_dim_index] = 1;
// 	// n-step
// 	for (size_t i=last_dim_index; i!=0; --i) {
// 		// if i (next element) is the mode
// 		if (i == mode) {
// 			// just ignore the usual multiplication process
// 			mul[i-1] = mul[i];
// 		} else {
// 			mul[i-1] = mul[i] * tensor->layout[tensor->layout_perm[i]];
// 		}
// 	}

// 	// prsize_t the resulting multiplication array
// 	//printf("Elements of (tensor_vector_product) mul[]:\n");
// 	//prsize_t_to_console(mul, last_dim_index+1);

// 	size_t result_index = 0; // result_index: set to 0 in the beginning
// 	// inrement counters like the dimensions change in the tensor
// 	for (size_t i=0; i<tensor->lin.size; ++i) {

// 		// MULTIPLICATION OPERATION
// 		result_tensor->data[result_index] += tensor->lin.data[i] * vector->data[counter[mode]];
// 		result_index = 0; // result_index: reset to 0 again

// 		// tick the smallest dimension (of weight 1)
// 		++counter[last_dim_index];
// 		// this loop carries the little tocks over
// 		for (size_t j=last_dim_index; j<=last_dim_index; --j) {
// 			// threshold reached on the lower dimension
// 			if (j>0 && counter[j] == tensor->layout[tensor->layout_perm[j]]) {
// 				// increment the higher dimension
// 				++counter[j-1];
// 				// reset the lower dimension
// 				counter[j] = 0;
// 			}
// 			if (j != mode) {
// 				result_index += counter[j] * mul[j];
// 			}
// 		}
// 	}
	
// 	free(counter);
// 	free(mul);
// }

// // MULTITHREADING VERSION
// // OUTPUT-MAJOR

// // non-obligatory misses on vector
// void
// toma(const struct tensor_storage * restrict tensor, const struct lin_storage * restrict vector, struct lin_storage * result_tensor, const size_t mode) {
// 	size_t last_dim_index = tensor->dim-1;
// 	size_t * mul = malloc(tensor->dim * sizeof(size_t));
// 	size_t * diff = calloc(tensor->dim, sizeof(size_t));

// 	// TO DO: we only need [mode] so stop at mode
// 	// INIT STEP
// 	mul[last_dim_index] = 1;
// 	// N-STEP
// 	for (size_t i=last_dim_index; i!=0; --i) {
// 		mul[i-1] = mul[i] * tensor->layout[tensor->layout_perm[i]];
// 		diff[i] = mul[i-1] - mul[i];
// 		if (i==mode) break;
// 	}
	
// 	size_t vector_size = vector->size;
// 	size_t result_size = tensor->lin.size / vector_size;
// 	// FACTOR OF 2
// 	vector_size /= 2;

// 	size_t tensor_diff = 0;
// 	size_t tensor_index = 0;

// 	// OUTPUT: this loop ensures we access the result_tensor linearly
// 	for (size_t i=0;;) {
// 		// VECTOR: this loop ensures we exhaust all elements of the vector
// 		for (size_t j=0; j<vector_size; ++j) {
// 			// MULTIPLICATION OPERATION (identical for both)
// 			result_tensor->data[i] += tensor->lin.data[tensor_index] * vector->data[j];
// 			//printf("tensor_index=%d\n", tensor_index);
// 			// vector loop must increment tensor_index by mul[mode]
// 			tensor_index += mul[mode];
// 		}
// 		// FACTOR OF 2
// 		if (++i >= (result_size/2)) {
// 			break;
// 		}
// 		// FACTOR OF 2 
// 		if (i % (mul[mode]/2) == 0) { // INSIGHT: in fact this is equivalent to a single counter
// 			tensor_diff += diff[mode];
// 		}
// 		tensor_index = i + tensor_diff;
// 	}

// 	free(diff);
// 	free(mul);
// }

