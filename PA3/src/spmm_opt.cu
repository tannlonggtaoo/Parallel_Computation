#include "spmm_opt.h"

constexpr int THRBLK_SIZE = 128; // thread block size
constexpr int ROWBLK_SIZE = 32;  // #rows for each thread block
constexpr int INFEATURE_MAX = 256

__global__ void spmm_kernel_placeholder(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int INFEATURE)
{
    // STEP 1 preparation
    // params for the whole thread block
    const int rblklo = blockIdx.x * ROWBLK_SIZE; // row block begin row
    const int rblkhi = min((blockIdx.x + 1) * blockDim.x, num_v); // row block end row (not included)
    const int valcnt = (ptr[rblkhi] - ptr[rblklo] + THRBLK_SIZE - 1) / THRBLK_SIZE; // len(ptr) = num_v + 1
    __shared__ int ansbuf[ROWBLK_SIZE][INFEATURE_MAX]; // should use constexpr so define INFEATURE_MAX (assume INFEATURE <= 256)
    // shared memory is NOT INITIALIZED
    int cnt4eachthr = ROWBLK_SIZE * INFEATURE / THRBLK_SIZE;
    for (int i = cnt4eachthr * threadIdx.x; i < cnt4eachthr * (threadIdx.x + 1); i++)
    {
        ((int*)ansbuf)[i] = 0;
    }
    __syncthreads();
    // (segmentation here is the same as STEP 3)

    // params for this thread
    const int vallo = ptr[rblklo] + threadIdx.x * valcnt;
    const int valhi = min(ptr[rblklo] + (threadIdx.x + 1) * valcnt, ptr[rblkhi]) // not included as well
    
    // find which row(s) this thread is handling
    int rlo = rblklo;
    while (ptr[rlo] <= vallo) rlo++;
    rlo--;
    int rhi = rlo;
    while (ptr[rhi] <= valhi) rhi++; // rhi not included

    // STEP 2 computation
    // all atomic
    for (int r = rlo; r < rhi; r++)
    {
        // low efficiency
        int ibegin = (r == rlo) ? vallo : ptr[r];
        int iend = (r == (rhi - 1)) ? valhi : ptr[r+1];
        for (int j = 0; j < INFEATURE; j++)
        {
            // j: col index of B
            float result = 0.0f;
            for (int i = ibegin; i < iend; i++)
            {
                result += vin[idx[i] * INFEATURE + j] * val[i];
            }
            // may try other scopes (now on device)
            atomicAdd(&(ansbuf[r][j]), result);
        }
    }
    __syncthreads();

    // STEP 3 move to global memory
    for (int i = cnt4eachthr * threadIdx.x; i < cnt4eachthr * (threadIdx.x + 1); i++)
    {
        vout[rblklo * INFEATURE + i] = ((int*)ansbuf)[i];
    }
}
void SpMMOpt::preprocess(float *vin, float *vout)
{
    grid.x = (num_v + ROWBLK_SIZE - 1) / ROWBLK_SIZE; // num_v = #rows in total
    block.x = THRBLK_SIZE;
    // equalization within each thread block (ROWBLK_SIZE rows)...

}

void SpMMOpt::run(float *vin, float *vout)
{
    // TODO: your code
    spmm_kernel_placeholder<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
}
