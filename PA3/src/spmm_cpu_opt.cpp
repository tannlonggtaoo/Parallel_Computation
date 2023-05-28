#include "spmm_cpu_opt.h"
#include <omp.h>

void run_spmm_cpu_placeholder(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int feat_len)
{
    #pragma omp parallel for
    for (int i = 0; i < num_v; ++i)
    {
        for (int j = ptr[i]; j < ptr[i + 1]; ++j)
        {
            for (int k = 0; k < feat_len; ++k)
            {
                vout[i * feat_len + k] += vin[idx[j] * feat_len + k] * val[j];
            }
        }
    }
}

void SpMMCPUOpt::preprocess(float *vin, float *vout)
{
}

void SpMMCPUOpt::run(float *vin, float *vout)
{
    run_spmm_cpu_placeholder(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
}
