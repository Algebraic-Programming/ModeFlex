#include <stdint.h>
#include <stddef.h>
#include <smmintrin.h>
#include <immintrin.h>

// ASSUME HASWELL 9AVX2.0)
// Assume memory alignment

// void
// CopyWithAVX(DTYPE * restrict dst, DTYPE * restrict src, size_t size) {
//     const size_t stride = 2 * sizeof(__m256i);
//     const size_t mul = sizeof(__m256i) / sizeof(DTYPE);
//     const size_t dtype_stride = 2 * mul;
//   	while (((uintptr_t)src & 15) != 0 && size > 0) {
// 		--size;
// 		*dst++ = *src++;
// 	}
// 	size *= sizeof(DTYPE);
// 	if (((uintptr_t)dst & 15) != 0) {
// 	    while (size >= stride)
// 	    {
// 	    	// _mm_prefetch((char*)(src + 0), _MM_HINT_NTA);
// 	       //  __m128i a = _mm_stream_load_si128((__m128i*)(src + 0*mul));
// 	       //  __m128i b = _mm_stream_load_si128((__m128i*)(src + 1*mul));
// 	      	// __m128i c = _mm_stream_load_si128((__m128i*)(src + 2*mul));
// 	       //  __m128i d = _mm_stream_load_si128((__m128i*)(src + 3*mul));
// 	       //  _mm_storeu_pd((double*)(dst + 0*mul), (__m128d)a);
// 	       //  _mm_storeu_pd((double*)(dst + 1*mul), (__m128d)b);
// 	       //  _mm_storeu_pd((double*)(dst + 2*mul), (__m128d)c);
// 	       //  _mm_storeu_pd((double*)(dst + 3*mul), (__m128d)d);
//         __m256i a = _mm256_load_si256((__m256i*)(src));
//         __m256i b = _mm256_load_si256((__m256i*)(src + 1*mul));
//         _mm256_storeu_si256((__m256i*)(dst), a);
//         _mm256_storeu_si256((__m256i*)(dst + 1*mul), b);
// 	        size -= stride;
// 	        src += dtype_stride;
// 	        dst += dtype_stride;
// 	    }
// 	} else {
// 	    while (size >= stride)
// 	    {
// 	    	// _mm_prefetch((char*)(src + 0), _MM_HINT_NTA);
// 	       //  __m128i a = _mm_stream_load_si128((__m128i*)(src + 0*mul));
// 	       //  __m128i b = _mm_stream_load_si128((__m128i*)(src + 1*mul));
// 	      	// __m128i c = _mm_stream_load_si128((__m128i*)(src + 2*mul));
// 	       //  __m128i d = _mm_stream_load_si128((__m128i*)(src + 3*mul));
// 	       //  _mm_store_pd((double*)(dst + 0*mul), (__m128d)a);
// 	       //  _mm_store_pd((double*)(dst + 1*mul), (__m128d)b);
// 	       //  _mm_store_pd((double*)(dst + 2*mul), (__m128d)c);
// 	       //  _mm_store_pd((double*)(dst + 3*mul), (__m128d)d);
//         __m256i a = _mm256_load_si256((__m256i*)(src));
//         __m256i b = _mm256_load_si256((__m256i*)(src + 1*mul));
//         _mm256_store_si256((__m256i*)(dst), a);
//         _mm256_store_si256((__m256i*)(dst + 1*mul), b);
// 	        size -= stride;
// 	        src += dtype_stride;
// 	        dst += dtype_stride;
// 	    }
// 	}
// 	// Now, the number of items we have to read/write is LESS maybe not multiple of (4?)
// 	size /= sizeof(DTYPE);
//     while(size) {
//     	--size;
//     	*dst++ = *src++;
//     }
// }

#define MMX2_MEMCPY_MIN_LEN 0x40
#define MMX_MMREG_SIZE 8
// ftp://ftp.work.acer-euro.com/gpl/AS1800/xine-lib/src/xine-utils/memcpy.c

int
aligned_by(const void * const ptr) {
	int power_of_2 = 1;
	// We have to test until 32 -> see if it fails with 32
	for (int i = power_of_2; i <= 32; i*=2) {
		if ((uintptr_t)ptr % i != 0) {
			// Failed with i so aligned by i/2
			return i/2;
		}
	}
	return 32; // Should never occur
}

// void
// CopyWithAVXNT(DTYPE * dst, DTYPE * src, size_t size) {
//     const size_t stride = 16 * sizeof(__m256i);
//     const size_t dtype_stride = 16 * (sizeof(__m256i) / sizeof(DTYPE));

//   	while (((uintptr_t)src & 31) != 0 && size > 0) {
// 			--size;
// 			*dst++ = *src++;
// 	}

// 	size *= sizeof(DTYPE);
// 	if (((uintptr_t)dst & 31) != 0) {
// 	    while (size >= stride)
// 	    {
//         __m256i a = _mm256_stream_load_si256((__m256i*)src + 0);
//         __m256i b = _mm256_stream_load_si256((__m256i*)src + 1);
//         __m256i c = _mm256_stream_load_si256((__m256i*)src + 2);
//         __m256i d = _mm256_stream_load_si256((__m256i*)src + 3);
//         __m256i e = _mm256_stream_load_si256((__m256i*)src + 4);
//         __m256i f = _mm256_stream_load_si256((__m256i*)src + 5);
//         __m256i g = _mm256_stream_load_si256((__m256i*)src + 6);
//         __m256i h = _mm256_stream_load_si256((__m256i*)src + 7);
//         __m256i i = _mm256_stream_load_si256((__m256i*)src + 8);
//         __m256i j = _mm256_stream_load_si256((__m256i*)src + 9);
//         __m256i k = _mm256_stream_load_si256((__m256i*)src + 10);
//         __m256i l = _mm256_stream_load_si256((__m256i*)src + 11);
//         __m256i m = _mm256_stream_load_si256((__m256i*)src + 12);
//         __m256i n = _mm256_stream_load_si256((__m256i*)src + 13);
//         __m256i o = _mm256_stream_load_si256((__m256i*)src + 14);
//         __m256i p = _mm256_stream_load_si256((__m256i*)src + 15);
//         _mm256_storeu_si256((__m256i*)dst + 0, a);
//         _mm256_storeu_si256((__m256i*)dst + 1, b);
//         _mm256_storeu_si256((__m256i*)dst + 2, c);
//         _mm256_storeu_si256((__m256i*)dst + 3, d);
//         _mm256_storeu_si256((__m256i*)dst + 4, e);
//         _mm256_storeu_si256((__m256i*)dst + 5, f);
//         _mm256_storeu_si256((__m256i*)dst + 6, g);
//         _mm256_storeu_si256((__m256i*)dst + 7, h);
//         _mm256_storeu_si256((__m256i*)dst + 8, i);
//         _mm256_storeu_si256((__m256i*)dst + 9, j);
//         _mm256_storeu_si256((__m256i*)dst + 10, k);
//         _mm256_storeu_si256((__m256i*)dst + 11, l);
//         _mm256_storeu_si256((__m256i*)dst + 12, m);
//         _mm256_storeu_si256((__m256i*)dst + 13, n);
//         _mm256_storeu_si256((__m256i*)dst + 14, o);
//         _mm256_storeu_si256((__m256i*)dst + 15, p);
// 	        size -= stride;
// 	        src += dtype_stride;
// 	        dst += dtype_stride;
// 	    }
// 	} else {
// 	    while (size >= stride)
// 	    {
//         __m256i a = _mm256_stream_load_si256((__m256i*)src + 0);
//         __m256i b = _mm256_stream_load_si256((__m256i*)src + 1);
//         __m256i c = _mm256_stream_load_si256((__m256i*)src + 2);
//         __m256i d = _mm256_stream_load_si256((__m256i*)src + 3);
//         __m256i e = _mm256_stream_load_si256((__m256i*)src + 4);
//         __m256i f = _mm256_stream_load_si256((__m256i*)src + 5);
//         __m256i g = _mm256_stream_load_si256((__m256i*)src + 6);
//         __m256i h = _mm256_stream_load_si256((__m256i*)src + 7);
//         __m256i i = _mm256_stream_load_si256((__m256i*)src + 8);
//         __m256i j = _mm256_stream_load_si256((__m256i*)src + 9);
//         __m256i k = _mm256_stream_load_si256((__m256i*)src + 10);
//         __m256i l = _mm256_stream_load_si256((__m256i*)src + 11);
//         __m256i m = _mm256_stream_load_si256((__m256i*)src + 12);
//         __m256i n = _mm256_stream_load_si256((__m256i*)src + 13);
//         __m256i o = _mm256_stream_load_si256((__m256i*)src + 14);
//         __m256i p = _mm256_stream_load_si256((__m256i*)src + 15);
//         _mm256_store_si256((__m256i*)dst + 0, a);
//         _mm256_store_si256((__m256i*)dst + 1, b);
//         _mm256_store_si256((__m256i*)dst + 2, c);
//         _mm256_store_si256((__m256i*)dst + 3, d);
//         _mm256_store_si256((__m256i*)dst + 4, e);
//         _mm256_store_si256((__m256i*)dst + 5, f);
//         _mm256_store_si256((__m256i*)dst + 6, g);
//         _mm256_store_si256((__m256i*)dst + 7, h);
//         _mm256_store_si256((__m256i*)dst + 8, i);
//         _mm256_store_si256((__m256i*)dst + 9, j);
//         _mm256_store_si256((__m256i*)dst + 10, k);
//         _mm256_store_si256((__m256i*)dst + 11, l);
//         _mm256_store_si256((__m256i*)dst + 12, m);
//         _mm256_store_si256((__m256i*)dst + 13, n);
//         _mm256_store_si256((__m256i*)dst + 14, o);
//         _mm256_store_si256((__m256i*)dst + 15, p);
// 	        size -= stride;
// 	        src += dtype_stride;
// 	        dst += dtype_stride;
// 	    }
// 	}
// 	// Now, the number of items we have to read/write is LESS maybe not multiple of (4?)
// 	size /= sizeof(DTYPE);
//     while(size) {
//     	--size;
//     	*dst++ = *src++;
//     }
// }

// void 
// CopyWithAVX(DTYPE * dst, DTYPE * src, size_t size) {
//     const size_t stride = 4 * sizeof(__m256i);
//     const size_t dtype_stride = 4 * (sizeof(__m256i) / sizeof(DTYPE));
//   	while (((uintptr_t)src & 31) != 0 && size > 0) {
// 			--size;
// 			*dst++ = *src++;
// 	}
// 	size *= sizeof(DTYPE);
// 	if (((uintptr_t)dst & 31) != 0) {
// 	    while (size >= stride)
// 	    {
// 	        __m256i a = _mm256_stream_load_si256((__m256i*)src + 0);
// 	        __m256i b = _mm256_stream_load_si256((__m256i*)src + 1);
// 	        __m256i c = _mm256_stream_load_si256((__m256i*)src + 2);
//        		__m256i d = _mm256_stream_load_si256((__m256i*)src + 3);
// 	        _mm256_storeu_si256((__m256i*)dst + 0, a);
//        		_mm256_storeu_si256((__m256i*)dst + 1, b);
// 	        _mm256_storeu_si256((__m256i*)dst + 2, c);
// 	        _mm256_storeu_si256((__m256i*)dst + 3, d);
// 	        size -= stride;
// 	        src += dtype_stride;
// 	        dst += dtype_stride;
// 	    }
// 	} else {
// 	    while (size >= stride)
// 	    {
// 	        __m256i a = _mm256_stream_load_si256((__m256i*)src + 0);
// 	        __m256i b = _mm256_stream_load_si256((__m256i*)src + 1);
// 	 	    __m256i c = _mm256_stream_load_si256((__m256i*)src + 2);
//        		__m256i d = _mm256_stream_load_si256((__m256i*)src + 3);
// 	        _mm256_store_si256((__m256i*)dst + 0, a);
//        		_mm256_store_si256((__m256i*)dst + 1, b);
// 	        _mm256_store_si256((__m256i*)dst + 2, c);
// 	        _mm256_store_si256((__m256i*)dst + 3, d);
// 	        size -= stride;
// 	        src += dtype_stride;
// 	        dst += dtype_stride;
// 	    }
// 	}
// 	// Now, the number of items we have to read/write is LESS maybe not multiple of (4?)
// 	size /= sizeof(DTYPE);
//     while(size) {
//     	--size;
//     	*dst++ = *src++;
//     }
// }

// void 
// CopyWithSSEPrefetchNT(DTYPE * restrict dst, DTYPE * restrict src, size_t size) {
//     const size_t stride = 4 * sizeof(__m128i);
//     const size_t mul = sizeof(__m128i) / sizeof(DTYPE);
//     const size_t dtype_stride = 4 * mul;
//   	while (((uintptr_t)src & 15) != 0 && size > 0) {
// 		--size;
// 		*dst++ = *src++;
// 	}
// 	size *= sizeof(DTYPE);
// 	if (((uintptr_t)dst & 15) != 0) {
// 	    while (size >= stride)
// 	    {
// 	    	// _mm_prefetch((char*)(src + 0), _MM_HINT_NTA);
// 	        __m128i a = _mm_stream_load_si128((__m128i*)(src + 0*mul));
// 	        __m128i b = _mm_stream_load_si128((__m128i*)(src + 1*mul));
// 	      	__m128i c = _mm_stream_load_si128((__m128i*)(src + 2*mul));
// 	        __m128i d = _mm_stream_load_si128((__m128i*)(src + 3*mul));
// 	        _mm_storeu_pd((double*)(dst + 0*mul), (__m128d)a);
// 	        _mm_storeu_pd((double*)(dst + 1*mul), (__m128d)b);
// 	        _mm_storeu_pd((double*)(dst + 2*mul), (__m128d)c);
// 	        _mm_storeu_pd((double*)(dst + 3*mul), (__m128d)d);
// 	        size -= stride;
// 	        src += dtype_stride;
// 	        dst += dtype_stride;
// 	    }
// 	} else {
// 	    while (size >= stride)
// 	    {
// 	    	// _mm_prefetch((char*)(src + 0), _MM_HINT_NTA);
// 	        __m128i a = _mm_stream_load_si128((__m128i*)(src + 0*mul));
// 	        __m128i b = _mm_stream_load_si128((__m128i*)(src + 1*mul));
// 	      	__m128i c = _mm_stream_load_si128((__m128i*)(src + 2*mul));
// 	        __m128i d = _mm_stream_load_si128((__m128i*)(src + 3*mul));
// 	        _mm_store_pd((double*)(dst + 0*mul), (__m128d)a);
// 	        _mm_store_pd((double*)(dst + 1*mul), (__m128d)b);
// 	        _mm_store_pd((double*)(dst + 2*mul), (__m128d)c);
// 	        _mm_store_pd((double*)(dst + 3*mul), (__m128d)d);
// 	        size -= stride;
// 	        src += dtype_stride;
// 	        dst += dtype_stride;
// 	    }
// 	}
// 	// Now, the number of items we have to read/write is LESS maybe not multiple of (4?)
// 	size /= sizeof(DTYPE);
//     while(size) {
//     	--size;
//     	*dst++ = *src++;
//     }
// }

void
CopyWithSSEPrefetchNT(DTYPE * restrict dst, DTYPE * restrict src, size_t size) {

  //   if (size < 16) {
	 //  	while (size) {
		// 	--size;
		// 	*dst++ = *src++;
		// }
  //   } else {
     	const size_t stride = 2 * sizeof(__m128i);
    	const size_t mul = sizeof(__m128i) / sizeof(DTYPE);
    	const size_t dtype_stride = 2 * mul;
	  	while (((uintptr_t)src & 15) != 0 && size > 0) {
			--size;
			*dst++ = *src++;
		}
		size *= sizeof(DTYPE);
		if (((uintptr_t)dst & 15) != 0) {
		    while (size >= stride)
		    {
		    	// _mm_prefetch((char*)(src + 0), _MM_HINT_NTA);
		        __m128i a = _mm_stream_load_si128((__m128i*)(src + 0*mul));
		        __m128i b = _mm_stream_load_si128((__m128i*)(src + 1*mul));
		        _mm_storeu_pd((double*)(dst + 0*mul), (__m128d)a);
		        _mm_storeu_pd((double*)(dst + 1*mul), (__m128d)b);
		        size -= stride;
		        src += dtype_stride;
		        dst += dtype_stride;
		    }
		} else {
		    while (size >= stride)
		    {
		    	// _mm_prefetch((char*)(src + 0), _MM_HINT_NTA);
		        __m128i a = _mm_stream_load_si128((__m128i*)(src + 0*mul));
		        __m128i b = _mm_stream_load_si128((__m128i*)(src + 1*mul));
		        _mm_store_pd((double*)(dst + 0*mul), (__m128d)a);
		        _mm_store_pd((double*)(dst + 1*mul), (__m128d)b);
		        size -= stride;
		        src += dtype_stride;
		        dst += dtype_stride;
		    }
		}
		// Now, the number of items we have to read/write is LESS maybe not multiple of (4?)
		size /= sizeof(DTYPE);
	    while(size) {
	    	--size;
	    	*dst++ = *src++;
	    }
    // }

}

void
nontemp_memcpy_by_2(DTYPE * restrict dst, const DTYPE * restrict source, size_t size) {
	while (((uintptr_t)source & 15) != 0 && size > 0) {
			--size;
			*dst++ = *source++;
	}
	__m128d memory_tmp;
	if (((uintptr_t)dst & 15) != 0) {
		size_t final = (size/2)*2;
		for (size_t i=0; i<final; i+=2) {
			memory_tmp = (__m128d) _mm_stream_load_si128((__m128i*)(source));
			_mm_storeu_pd((dst), memory_tmp);
			dst += 2;
			source += 2;
			size -= 2;
		}
		if (size == 1) {
			*dst++ = *source++;
		}
	} else {
		size_t final = (size/2)*2;
		for (size_t i=0; i<final; i+=2) {
			memory_tmp = (__m128d) _mm_stream_load_si128((__m128i*)(source));
			_mm_store_pd((dst), memory_tmp);
			dst += 2;
			source += 2;
			size -= 2;
		}
		if (size == 1) {
			*dst++ = *source++;
		}
	}
}

void
nontemp_memcpy(DTYPE * restrict dst, const DTYPE * restrict source, size_t size) {

	while ((uintptr_t)source % 16 != 0 && size > 0) {
		if ((uintptr_t)source % 16 == 0 && size >= 2) {
			__m128d memory_tmp = (__m128d) _mm_stream_load_si128((__m128i*)(source));
			_mm_storeu_pd((dst), memory_tmp);
			size -= 2;
			source += 2;
			dst += 2;
		} else {
			--size;
			*dst++ = *source++;
		}
	}
	if (aligned_by((void*)dst) < 16) {
		size_t final = (size/2)*2;
		//printf("final=%d\n", final);
		for (size_t i=0; i<final; i+=2) {
			__m128d memory_tmp = (__m128d) _mm_stream_load_si128((__m128i*)(source));
			_mm_storeu_pd((dst), memory_tmp);
			dst += 2;
			source += 2;
			size -= 2;
		}
		if (size == 1) {
			*dst++ = *source++;
		}
	} else {
		size_t final = (size/2)*2;
		for (size_t i=0; i<final; i+=2) {
			__m128d memory_tmp = (__m128d) _mm_stream_load_si128((__m128i*)(source));
			_mm_store_pd((dst), memory_tmp);
			dst += 2;
			source += 2;
			size -= 2;
		}
		if (size == 1) {
			*dst++ = *source++;
		}
	}
}

#if 0
	// Quite important: relative alignment is actually RIGHT if we use a cast to void pointers (or multiply the difference by sizeof(DTYPE))
	// printf("Relative alignment: %d\n", aligned_by((void*)((void*)dst-(void*)source)));
	
	// Actually, stores are always aligned for dst (because we know it's 32-byte boundary aligned(!))
	// printf("Stage 1, source aligned by %d, dst aligned by %d\n", aligned_by((void*)source), aligned_by((void*)dst));

	// copy over until we are aligned
	while ((uintptr_t)source % 32 != 0 && size > 0) {

		// printf("SOURCE Pointer not aligned by 32\n");

		if ((uintptr_t)source % 16 == 0 && size >= 2) {
			//printf("Pointer aligned by 16, read 2\n");
			__m128d memory_tmp = (__m128d) _mm_stream_load_si128((__m128i*)(source));
			// HERE: necessarily unaligned (we don't know actually...)
			// printf("how many of these more?\n");
			_mm_storeu_pd((dst), memory_tmp);
			size -= 2;
			// Increment the pointers manually
			source += 2;
			dst += 2;
		} else {
			//printf("Pointer not aligned by 32 nor 16: read 1\n");
			--size;
			// Only here we increment the pointers correctly
			*dst++ = *source++;
		}
	}

	// printf("Stage 2, source aligned by %d, dst aligned by %d\n", aligned_by((void*)source), aligned_by((void*)dst));

	if (aligned_by((void*)dst) < 32) {

		// ASSUME ALIGNMENT
		size_t final = (size/4)*4;
		for (size_t i=0; i<final; i+=4) {
			//printf("Read 4 nontemporarly\n");
			__m256d memory_tmp = (__m256d) _mm256_stream_load_si256((__m256i*)(source));
			_mm256_storeu_pd(dst, memory_tmp);
			// Update the pointers manually
			source += 4;
			dst += 4;
		}

		// ASSUME WE HAVE EITHER 3, 2, 1 or 0 to read
		size -= final;
		if (size >= 2) {
			//printf("Read 2 nontemporarly\n");
			__m128d memory_tmp = (__m128d) _mm_stream_load_si128((__m128i*)(source));
			_mm_storeu_pd((dst), memory_tmp);
			dst += 2;
			source += 2;
			size -= 2;
		}

		// EITHER 1 or 0 left
		if (size == 1) {
			//printf("1 left: read 1\n");
			*dst++ = *source++;
		}

	} else {

		// ASSUME ALIGNMENT
		size_t final = (size/4)*4;
		for (size_t i=0; i<final; i+=4) {
			//printf("Read 4 nontemporarly\n");
			__m256d memory_tmp = (__m256d) _mm256_stream_load_si256((__m256i*)(source));
			_mm256_store_pd(dst, memory_tmp);
			// Update the pointers manually
			source += 4;
			dst += 4;
		}

		// ASSUME WE HAVE EITHER 3, 2, 1 or 0 to read
		size -= final;
		if (size >= 2) {
			//printf("Read 2 nontemporarly\n");
			__m128d memory_tmp = (__m128d) _mm_stream_load_si128((__m128i*)(source));
			_mm_store_pd((dst), memory_tmp);
			dst += 2;
			source += 2;
			size -= 2;
		}

		// EITHER 1 or 0 left
		if (size == 1) {
			//printf("1 left: read 1\n");
			*dst++ = *source++;
		}

	}
#endif

	//////////////////// LOADS (integers)
#if 0
	// avx 1, unaligned, temporal, data crosses a cache line boundary
	__m256i _mm256_lddqu_si256 (__m256i const * mem_addr)
	// avx 1, unaligned, temporal
	__m256i _mm256_loadu_si256 (__m256i const * mem_addr)

	// sse 4.1, 16-byte aligned, nontemporal
	__m128i _mm_stream_load_si128 (const __m128i* mem_addr)
	// avx 2.0, 32-byte aligned, nontemporal
	__m256i _mm256_stream_load_si256 (__m256i const* mem_addr)

	//////////////////// Stores (doubles)

	// avx 1, unaligned, temporal
	void _mm256_storeu_pd (double * mem_addr, __m256d a)

	// sse 2, 16-byte aligned, temporal
	void _mm_store_pd (double* mem_addr, __m128d a)
	// avx 1, 32-byte aligned, temporal
	void _mm256_store_pd (double * mem_addr, __m256d a)
	// avx 1, 32-byte aligned, nontemporal
	void _mm256_stream_pd (double * mem_addr, __m256d a)
#endif
