#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>

#include "worker.h"

void Worker::sort() {
  /** Your code ... */
  // you can use variables in class Worker: n, nprocs, rank, block_len, data

  // n: length of the unsorted array
  // nprocs: num of process
  // rank: ...
  // block_len: the partition method is fixed so this var is given
  // data: data = new float[block_len]
  // each proc has its own worker

  // STEP 0: sort in process
  std::sort(this->data, this->data + this->block_len);

  // STEP 1: pair-wise merging
  MPI_Status send_status, recv_status;
  MPI_Request send_request, recv_request;
  float* mergebuf = new float[this->block_len * 2 + this->nprocs];
  // the last block_len < other block_len + nprocs
  // std::merge is enough, do not use std::inplace_merge

  int direction;
  int peerrank;
  int peerlen;

  // TODO : try min-max first
  for (int i = 0; i < this->nprocs; i++)
  {
    direction = (i % 2 == 0 ? 1 : -1) * (this->rank % 2 == 0 ? 1 : -1);
    peerrank = this->rank + direction;

    //debug
    //printf("[proc %d] @ iter %d : dir=%d, peerrank=%d\n",this->rank,i,direction,peerrank);
    //endofdebug

    if ((peerrank < 0) || (peerrank >= this->nprocs))
    {
      continue; // can be optimized later...?
    }
    MPI_Irecv(
      mergebuf + this->block_len,
      this->block_len + this->nprocs,
      MPI_FLOAT,
      peerrank,
      0,
      MPI_COMM_WORLD,
      &recv_request);
    MPI_Isend(
      this->data,
      this->block_len,
      MPI_FLOAT,
      peerrank,
      0,
      MPI_COMM_WORLD,
      &send_request);
    MPI_Wait(&recv_request, &recv_status);
    MPI_Get_count(&recv_status, MPI_FLOAT, &peerlen);
    std::merge(
      this->data,
      this->data + this->block_len,
      mergebuf + this->block_len,
      mergebuf + this->block_len + peerlen,
      mergebuf);
    if (direction == 1)
    {
      memcpy(this->data, mergebuf, this->block_len * sizeof(float));
    }
    else // direction == -1
    {
      memcpy(this->data, mergebuf + peerlen, this->block_len * sizeof(float));
    }
    // last step
    MPI_Wait(&send_request, &send_status);
  }
  delete [] mergebuf;
}
