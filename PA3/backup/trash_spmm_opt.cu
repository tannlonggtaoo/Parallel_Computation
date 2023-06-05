#include "spmm_opt.h"

constexpr int THRBLK_SIZE = 8; // thread block size
constexpr int ROWBLK_SIZE = 4;  // #rows for each thread block
constexpr int INFEATURE_MAX = 1;
//constexpr int THRBLK_SIZE = 128; // thread block size
//constexpr int ROWBLK_SIZE = 48;  // #rows for each thread block
//constexpr int INFEATURE_MAX = 256;



__global__ void spmm_kernel_placeholder(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int INFEATURE)
{
	// vin is B !!!


	//debug	
	if (blockIdx.x==0 && threadIdx.x == 0) printf("num_v=%d,INFEATURE=%d\n",num_v,INFEATURE);
    
	// STEP 1 preparation
    // params for the whole thread block
    const int rblklo = blockIdx.x * ROWBLK_SIZE; // row block begin row
    const int rblkhi = min((blockIdx.x + 1) * ROWBLK_SIZE, num_v); // row block end row (not included)

    const int valcnt = (ptr[rblkhi] - ptr[rblklo] + THRBLK_SIZE - 1) / THRBLK_SIZE; // len(ptr) = num_v + 1
    __shared__ float ansbuf[ROWBLK_SIZE * INFEATURE_MAX]; // should use constexpr so define INFEATURE_MAX (assume INFEATURE <= 256)
    // shared memory is NOT INITIALIZED
    int cnt4eachthr = ROWBLK_SIZE * INFEATURE_MAX / THRBLK_SIZE;

    // debug = 8
    //if (blockIdx.x==17) printf("cnt4eachthr%d\n",cnt4eachthr);
    //return;

    for (int i = cnt4eachthr * threadIdx.x; i < cnt4eachthr * (threadIdx.x + 1); i++)
    {
        ansbuf[i] = 0.0f;

		// debug
		// if(blockIdx.x==0) printf("threadidx=%d,ansbuf[%d][%d]=0\n",threadIdx.x,i/INFEATURE_MAX,i%INFEATURE_MAX);
    }
    __syncthreads();
    // (segmentation here is the same as STEP 3)

    // params for this thread

    // debug
    //if(blockIdx.x==0) printf("blockIdx=%d,threadIdx=%d,blockDim=%d,rblklo=%d,rblkhi=%d,valcnt=%d\n",blockIdx.x,threadIdx.x,blockDim.x,rblklo,rblkhi,valcnt);
    // for debugging (at least OK here...)

    const int vallo = ptr[rblklo] + threadIdx.x * valcnt;
    const int valhi = min(ptr[rblklo] + (threadIdx.x + 1) * valcnt, ptr[rblkhi]); // not included as well

	// debug
	// if(blockIdx.x==0) printf("threadIdx=%d,vallo=%d,valhi=%d,totalnum4blk=%d\n",threadIdx.x,vallo,valhi,ptr[rblkhi]-ptr[rblklo]);
	
	// end debug

	if (vallo < valhi)
	{
    	// find which row(s) this thread is handling
    	int rlo = rblklo;
    	while ((ptr[rlo] <= vallo) && (rlo < num_v)) rlo++;
    	rlo--;
    	int rhi = rlo;
    	while ((ptr[rhi] <= valhi) && (rhi < num_v) ) rhi++; // rhi not included

		// debug
		if (blockIdx.x==0 && threadIdx.x == 0) printf("ptr:%d %d %d %d %d %d %d %d %d ...%d \n",ptr[0],ptr[1],ptr[2],ptr[3],ptr[4],ptr[5],ptr[6],ptr[7],ptr[8],ptr[169343]);
		if (blockIdx.x==42335) printf("threadIdx=%d,rlo=%d,rhi=%d,vallo=%d,valhi=%d,valcnt=%d\n",threadIdx.x,rlo,rhi,vallo,valhi,valcnt);
		// return;


    	// STEP 2 computation
    	// all atomic
    	for (int r = rlo; r < rhi; r++)
    	{
        	// low efficiency
        	int ibegin = (r == rlo) ? vallo : ptr[r];
        	int iend = (r == (rhi - 1)) ? valhi : ptr[r+1];

			//debug
			if(blockIdx.x == 42335) printf("--threadidx=%d,r=%d,ibegin=%d,iend=%d\n",threadIdx.x,r,ibegin,iend);
			//continue;
			//end debug

			if (ibegin==iend) continue;

        	for (int j = 0; j < INFEATURE; j++)
        	{
            	// j: col index of B
            	float result = 0.0f;
            	for (int i = ibegin; i < iend; i++)
            	{
                	result += vin[idx[i] * INFEATURE + j] * val[i];
					// if(blockIdx.x == 0) printf("threadidx=%d,vin[%d][%d],val[%d]\n",threadIdx.x,idx[i],j,i);
            	}
            	// may try other scopes (now on device)
            	atomicAdd(ansbuf + (r - rblklo) * INFEATURE_MAX + j, result);
				
				//debug
				// if((r-rblklo)*INFEATURE_MAX + j >= INFEATURE_MAX * ROWBLK_SIZE) printf("blockidx=%d,threadidx=%d,r=%d,rblklo=%d,j=%d,ansbuf[%d]\n",blockIdx.x,threadIdx.x,r,rblklo,j,(r-rblklo)*INFEATURE_MAX+j);
				if(blockIdx.x == 42335) printf("threadIdx=%d,A[%d][%d]->A[%d][%d],C[%d][%d]+=%f\n",threadIdx.x,r,ibegin%num_v,r,iend%num_v,r,j,result);
        	}
    	}
	}
	__syncthreads();
	// debug

    // STEP 3 move to global memory
	// here vout.shape = [num_v,INFEATURE], so cnt4eachthr should be reassigned
	cnt4eachthr = ROWBLK_SIZE * INFEATURE / THRBLK_SIZE;
    for (int i = cnt4eachthr * threadIdx.x; i < cnt4eachthr * (threadIdx.x + 1); i++)
    {
        vout[(rblklo + i / INFEATURE) * INFEATURE + i % INFEATURE] = ansbuf[(i/INFEATURE) * INFEATURE_MAX + i%INFEATURE];
    	// if (blockIdx.x == 42335) printf("threadidx=%d,vout[%d][%d]=ansbuf[%d][%d](i=%d)\n",threadIdx.x,rblklo+i/INFEATURE,i%INFEATURE,i/INFEATURE,i%INFEATURE,i);
	}
}
void SpMMOpt::preprocess(float *vin, float *vout)
{
    grid.x = (num_v + ROWBLK_SIZE - 1) / ROWBLK_SIZE; // num_v = #rows in total
    block.x = THRBLK_SIZE;
    // equalization within each thread block (ROWBLK_SIZE rows)...

	// debug
	printf("grid.x=%d,block.x=%d\n",grid.x,block.x);
}

void SpMMOpt::run(float *vin, float *vout)
{
    // TODO: your code
    spmm_kernel_placeholder<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
}
