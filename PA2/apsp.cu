// PLEASE MODIFY THIS FILE TO IMPLEMENT YOUR SOLUTION

#include "apsp.h"

constexpr int b = 32;
constexpr int dmax = 100000 + 1; // edge weight < 100000 is guaranteed

namespace
{

__global__ void step1(const int p, const int n, int* graph)
{
    // handles the p-th diagnal block

    // copy the p-th block to shared mem first!
    __shared__ int cache[b][b];
    // coordinate order:
    //         x
    //       <--->
    //    A
    //  y |  (blk)
    //    V

    // local coordinates, e.g. cache[y][x]
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    // global coordinates (on graph), e.g. graph[yg*n + xg]
    const int xg = p * blockDim.x + x;
    const int yg = p * blockDim.y + y;

    if (xg < n && yg < n)
    {
        cache[y][x] = graph[yg * n + xg];
    }
    else
    {
        cache[y][x] = dmax;
    }

    __syncthreads(); // wait till all data prepared at shared mem

    // all blks will act like diagnal blks (no exp time wasted...)
    int newchoice;
    #pragma unroll
    for (int k = 0; k < b; k++)
    {
        newchoice = cache[y][k] + cache[k][x];
        __syncthreads(); // cache will be modified in the following lines
        cache[y][x] = min(cache[y][x], newchoice);
        __syncthreads();
    }

    // send results back to global memory
    if (xg < n && yg < n)
    {
        graph[yg * n + xg] = cache[y][x];
    }

}

__global__ void step2(const int p, const int n, int* graph)
{
    // handles the path between the p-th diagnal block and other blocks
    // only a cross-shape block field is related to the p-th block
    if (blockIdx.x == p)
    {
        return;     // could be better?
    }

    const int x = threadIdx.x;
    const int y = threadIdx.y;

    // global coordnate of p-th diag block
    int xg = p * blockDim.x + x;
    int yg = p * blockDim.y + y;

    // have to load 2 blks to shared memory
    // 1st: the p-th diagnal block
    // 2nd: the corresponding 'other block' determined by blkIdx

    __shared__ int diagnal[b][b];
    __shared__ int cache[b][b];     // self block

    // 1st
    if (xg < n && yg < n)
    {
        diagnal[y][x] = graph[yg * n + xg];
    }
    else
    {
        diagnal[y][x] = dmax;
    }

    // 2nd
    if (blockIdx.y == 0)
    {   // redirect xg,yg to current block to save space
        xg = blockDim.x * blockIdx.x + x; // row
        
    }
    else
    {
        yg = blockDim.x * blockIdx.x + y; // col
    }
    if (xg < n && yg < n)
    {
        cache[y][x] = graph[yg * n + xg];
    }
    else
    {
        cache[y][x] = dmax;
    }

    __syncthreads();

    // then update cache
    int newchoice;
    if (blockIdx.y == 0) // row
    {
        #pragma unroll
        for (int k = 0; k < blockDim.x; k++)
        {
            newchoice = diagnal[y][k] + cache[k][x];
            __syncthreads();
            cache[y][x] = min(cache[y][x], newchoice);
            __syncthreads();
        }
    }
    else // col
    {
        #pragma unroll
        for (int k = 0; k < blockDim.x; k++)
        {
            newchoice = cache[y][k] + diagnal[k][x];
            __syncthreads();
            cache[y][x] = min(cache[y][x], newchoice);
            __syncthreads();
        }
    }

    // send results back to global memory
    // xg,yg refers to current block so directly use it
    if (xg < n && yg < n)
    {
        graph[yg * n + xg] = cache[y][x];
    }
}

__global__ void step3(const int p, const int n, int* graph)
{
    // handles the rest blocks
    // quit if current block is in the 'cross field'
    if (blockIdx.x == p || blockIdx.y == p)
    {
        return;
    }

    // local coordinates, e.g. cache[y][x]
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    // global coordinates (on graph), e.g. graph[yg*n + xg]
    const int xg = blockIdx.x * blockDim.x + x;
    const int yg = blockIdx.y * blockDim.y + y;
    const int xgp = p * blockDim.x + x;
    const int ygp = p * blockDim.y + y;

    __shared__ int cache[b][b];
    __shared__ int rowblk[b][b];    // in the same row
    __shared__ int colblk[b][b];    // in the same col

    // load 3 blocks to shared memory
    if (xg < n && yg < n)
    {
        cache[y][x] = graph[yg * n + xg];
    }
    else
    {
        cache[y][x] = dmax;
    }
    if (xgp < n && yg < n)
    {
        rowblk[y][x] = graph[yg * n + xgp];
    }
    else
    {
        rowblk[y][x] = dmax;
    }
    if (xg < n && ygp < n)
    {
        colblk[y][x] = graph[ygp * n + xg];
    }
    else
    {
        colblk[y][x] = dmax;
    }
    __syncthreads();

    // update!
    int newchoice;
    #pragma unroll
    for (int k = 0; k < b; k++)
    {
        newchoice = rowblk[y][k] + colblk[k][x];
        __syncthreads(); // cache will be modified in the following lines
        cache[y][x] = min(cache[y][x], newchoice);
        __syncthreads();
    }
    // send results back to global memory
    if (xg < n && yg < n)
    {
        graph[yg * n + xg] = cache[y][x];
    }
}


}   //namespace

void printcudamem(/* device */ int *graph, int n)
{
    int* tmp = (int*)malloc(n*n);
    cudaMemcpy(tmp, graph, n*n, cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%d ",tmp[i*n+j]);
        }
        printf("\n");
    }

    free(tmp);
}

void apsp(int n, /* device */ int *graph) {
    dim3 thr(b, b);     // size of each block
    dim3 nblk_s1(1,1);                               // diagnal
    dim3 nblk_s2((n - 1) / b + 1,2);                 // cross
    // .y=0-> row, .y=1-> col, .x-> the order in row or col
    dim3 nblk_s3((n - 1) / b + 1,(n - 1) / b + 1);   // all
    for (int p = 0; p < (n - 1) / b + 1; p++)
    {
        printcudamem(graph,n);
        step1<<<nblk_s1, thr>>>(p, n, graph);
        printcudamem(graph,n);
        step2<<<nblk_s1, thr>>>(p, n, graph);
        step3<<<nblk_s1, thr>>>(p, n, graph);
    }
}

