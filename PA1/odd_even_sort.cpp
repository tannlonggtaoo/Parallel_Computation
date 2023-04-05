#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>

#include "worker.h"

/* bool dont_need_merge(int peerrank, float mymargin, int iter, int direction)
{
  // check margin to determine if merging is needed
  float peermargin;
  MPI_Status send_status, recv_status;
  MPI_Request send_request, recv_request;
  MPI_Irecv(
      &peermargin,
      1,
      MPI_FLOAT,
      peerrank,
      iter,
      MPI_COMM_WORLD,
      &recv_request);
  MPI_Isend(
      &mymargin,
      1,
      MPI_FLOAT,
      peerrank,
      iter,
      MPI_COMM_WORLD,
      &send_request);

  MPI_Wait(&send_request, &send_status);
  MPI_Wait(&recv_request, &recv_status);

  return (((direction == 1) && (mymargin <= peermargin)) 
      || ((direction == -1) && (peermargin <= mymargin)));
} */

/* void printbuf(float* buf, int len, int rank, int iter)
{
  printf("[proc %d] iter %d :", rank, iter);
  for (int i = 0; i < len; i++)
  {
    printf("%lf ",buf[i]);
  }
  printf("\n");
} */

void Worker::sort()
{
  /** Your code ... */
  // you can use variables in class Worker: n, nprocs, rank, block_len, data

  // n: length of the unsorted array
  // nprocs: num of process
  // rank: ...
  // block_len: the partition method is fixed so this var is given
  // data: data = new float[block_len]
  // each proc has its own worker

  // STEP -1: if i am empty then return
  if (this->block_len == 0) return;

  // STEP 0: sort in process
  std::sort(this->data, this->data + this->block_len);

  // STEP 1: pair-wise merging
  MPI_Status send_status[2], recv_status[2];
  MPI_Request send_request[2], recv_request[2];
  int comm_block_len = ceiling(this->n,this->nprocs);
  float** mergebuf = new float* [2];
  mergebuf[0] = new float[comm_block_len + this->block_len];
  mergebuf[1] = new float[comm_block_len + this->block_len];

  int direction;
  int peerrank;
  int peerlen;

  for (int i = 0; i < this->nprocs; i++)
  {

    direction = (i % 2 == 0 ? 1 : -1) * (this->rank % 2 == 0 ? 1 : -1);
    peerrank = this->rank + direction;
    if (i == 0)
    {
      // first iter, need to wait for the first data
      if ((peerrank >= 0) && (comm_block_len * peerrank < (int)this->n))
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

        printf("[proc %d] iter %d : <--n--> proc %d, tag %d @ idx %d\n",this->rank, i, peerrank, i, i%2);
      }
    } // first iter

    // get next data without waiting
    // last iter do not need to get next data
    direction = -(i % 2 == 0 ? 1 : -1) * (this->rank % 2 == 0 ? 1 : -1);
    peerrank = this->rank + direction;
    if (i != (this->nprocs - 1))
    {
      if ((peerrank >= 0) && (comm_block_len * peerrank < (int)this->n))
      {
        MPI_Irecv(
          mergebuf[(i+1)%2] + this->block_len,
          comm_block_len,
          MPI_FLOAT,
          peerrank,
          i + 1,
          MPI_COMM_WORLD,
          &recv_request[(i+1)%2]);
        

        printf("[proc %d] iter %d next : ready to recieve proc %d, tag %d @ idx %d\n",this->rank, i, peerrank, i + 1, (i+1)%2);

      }
    }


    // determine if we need merging at this iter
    direction = (i % 2 == 0 ? 1 : -1) * (this->rank % 2 == 0 ? 1 : -1);
    peerrank = this->rank + direction;
    if ((peerrank >= 0) && (comm_block_len * peerrank < (int)this->n))
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

      printf("[proc %d] iter %d : merged @ idx %d\n",this->rank, i, i%2);
    }

    // send merged data
    direction = -(i % 2 == 0 ? 1 : -1) * (this->rank % 2 == 0 ? 1 : -1);
    peerrank = this->rank + direction;

    if (i != (this->nprocs - 1))
    {
      if ((peerrank >= 0) && (comm_block_len * peerrank < (int)this->n))
      {
        
        MPI_Isend(
          this->data,
          this->block_len,
          MPI_FLOAT,
          peerrank,
          i + 1,
          MPI_COMM_WORLD,
          &send_request[(i+1)%2]);

          printf("[proc %d] iter %d next : --n--> proc %d, tag %d @ idx %d\n",this->rank, i, peerrank, i + 1, (i+1)%2);
      }
    }
    // last step
    direction = (i % 2 == 0 ? 1 : -1) * (this->rank % 2 == 0 ? 1 : -1);
    peerrank = this->rank + direction;
    if ((peerrank >= 0) && (comm_block_len * peerrank < (int)this->n))
    {
      MPI_Wait(&send_request[i%2], &send_status[i%2]);
    }
  }
  delete [] mergebuf[0];
  delete [] mergebuf[1];
  delete [] mergebuf;
}
