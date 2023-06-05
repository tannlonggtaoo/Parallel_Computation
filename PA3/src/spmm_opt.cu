// reference : GE-SpMM (arXiv:2007.03179)
#include "spmm_opt.h"

constexpr int BLOCK_SIZE = 32;
constexpr int WARP_SIZE = 32;

__global__ void spmm_kernel_placeholder(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int INFEATURE)
{	
	const int rid = blockIdx.x * BLOCK_SIZE + threadIdx.y;
	// each thread block handles BLOCK_SIZE rows
	const int cid = blockIdx.y * WARP_SIZE + threadIdx.x;
	// 0 to feat_in
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;	
	// 0 to WARP_SIZE*BLOCK_SIZE
	if ((rid >= num_v) || (cid >= INFEATURE)) return;

	extern __shared__ int sm[];
	int* sm_k = sm;   									// for caching idx
	float* sm_v = (float*)sm + BLOCK_SIZE * WARP_SIZE;  // for caching val
	int sm_base = threadIdx.y * WARP_SIZE;				// 0,32,64,...

	int begin = ptr[rid], end = ptr[rid + 1];
	float result = 0.0f;
	int k,p,kk,cur;		// iter vars

	// iteration over whole row
	for (p = begin; p < end; p+=WARP_SIZE)
	{
		cur = p + threadIdx.x;
		// loading A (caching)
		if (cur < end)
		{
			sm_k[tid] = idx[cur];
			sm_v[tid] = val[cur];
		}
		__syncwarp();	// wait till all float num loaded
		
		// computation
		for (kk = 0; (kk < WARP_SIZE) && (p + kk < end); kk++)
		{
			k = sm_k[sm_base + kk];		// corresponding idx
			result += sm_v[sm_base + kk] * vin[k * INFEATURE + cid];
		}
		__syncwarp();
	}
	vout[rid * INFEATURE + cid] = result;
}
void SpMMOpt::preprocess(float *vin, float *vout)
{
    // TODO: your code
    grid.x = (num_v + BLOCK_SIZE - 1) / BLOCK_SIZE;
	grid.y = (feat_in + WARP_SIZE - 1) / WARP_SIZE;
    block.x = WARP_SIZE;
	block.y = BLOCK_SIZE;
}

void SpMMOpt::run(float *vin, float *vout)
{
    // TODO: your code
    spmm_kernel_placeholder<<<grid, block, WARP_SIZE*BLOCK_SIZE*(sizeof(int) + sizeof(float))>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
}
