#include "aplusb.h"
#include <x86intrin.h>

void a_plus_b_intrinsic(float* a, float* b, float* c, int n) {
    __m256 reg1,reg2,reg3;
    for (int i = 0; i < n; i += 8)
    {
	    reg1 = _mm256_load_ps(a + i);
	    reg2 = _mm256_load_ps(b + i);
	    reg3 = _mm256_add_ps(reg1, reg2);
	    _mm256_store_ps(c + i, reg3);
    }
    return;
}
