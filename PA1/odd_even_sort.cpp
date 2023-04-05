#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>

#include "worker.h"

void Worker::sort()
{
  /** Your code ... */
  // you can use variables in class Worker: n, nprocs, rank, block_len, data

  // STEP -1: if i am empty then return
  if (this->block_len == 0) return;

  // STEP 0: sort in process
  std::sort(this->data, this->data + this->block_len);

  // STEP 1: preparing buffers, etc
  MPI_Status send_status[2], recv_status[2];
  MPI_Request send_request[2], recv_request[2];
  int comm_block_len = ceiling(this->n,this->nprocs);
  float** mergebuf = new float* [2];
  mergebuf[0] = new float[comm_block_len + this->block_len];
  mergebuf[1] = new float[comm_block_len + this->block_len];

  int direction, nextdirection;  // = 1 or -1
  int peerrank, nextpeerrank;
  int peerlen;
  bool peervalid, nextpeervalid;

  for (int i = 0; i < this->nprocs; i++)
  {
    direction = (i % 2 == 0 ? 1 : -1) * (this->rank % 2 == 0 ? 1 : -1);
    peerrank = this->rank + direction;
    peervalid = (peerrank >= 0) && (comm_block_len * peerrank < (int)this->n);

    nextdirection = -direction;
    nextpeerrank = this->rank + nextdirection;
    nextpeervalid = (nextpeerrank >= 0) && (comm_block_len * nextpeerrank < (int)this->n);

    // STEP 2 : iter 0, have to wait for the first block
    if ((i == 0) && peervalid)
    {
      MPI_Irecv(
        mergebuf[i%2] + this->block_len,
        comm_block_len,
        MPI_FLOAT,
        peerrank,
        i,
        MPI_COMM_WORLD,
        &recv_request[i%2]);
      MPI_Isend(
        this->data,
        this->block_len,
        MPI_FLOAT,
        peerrank,
        i,
        MPI_COMM_WORLD,
        &send_request[i%2]);
    }

    // STEP 3 : get next block (Irecv)
    // exception : last iter do not need to get next block
    if ((i != (this->nprocs - 1)) && nextpeervalid)
    {
      MPI_Irecv(
        mergebuf[(i+1)%2] + this->block_len,
        comm_block_len,
        MPI_FLOAT,
        nextpeerrank,
        i + 1,
        MPI_COMM_WORLD,
        &recv_request[(i+1)%2]);
    }

    // STEP 4 : merge if needed
    if ((peerrank >= 0) && peervalid)
    {
      MPI_Wait(&recv_request[i%2], &recv_status[i%2]);
      MPI_Get_count(&recv_status[i%2], MPI_FLOAT, &peerlen);
      std::merge(
        this->data,
        this->data + this->block_len,
        mergebuf[i%2] + this->block_len,
        mergebuf[i%2] + this->block_len + peerlen,
        mergebuf[i%2]);
      if (direction == 1)
      {
        memcpy(this->data, mergebuf[i%2], this->block_len * sizeof(float));
      }
      else // direction == -1
      {
        memcpy(this->data, mergebuf[i%2] + peerlen, this->block_len * sizeof(float));
      }
    }

    // STEP 5 : send merged data to peer in next iter
    if ((i != (this->nprocs - 1)) && nextpeervalid)
    {
      MPI_Isend(
        this->data,
        this->block_len,
        MPI_FLOAT,
        nextpeerrank,
        i + 1,
        MPI_COMM_WORLD,
        &send_request[(i+1)%2]);
    }
    
    // STEP 6 : wait for the send request in last epoch
    if (peervalid)
    {
      MPI_Wait(&send_request[i%2], &send_status[i%2]);
    }
  }

  // STEP 7 : delete space
  delete [] mergebuf[0];
  delete [] mergebuf[1];
  delete [] mergebuf;
}
