#include <chrono>
#include <iostream>
#include <mpi.h>
#include <time.h>
#include <cstring>
#include <cmath>
#include <algorithm>

#define EPS 1e-8

namespace ch = std::chrono;

void printbuf(float* buf, int n)
{
    for (int i = 0; i < n; i++)
    {
        printf("%f ", *(buf + i));
    }
    printf("\n");
}

void Ring_Allreduce(void* sendbuf, void* recvbuf, int n, MPI_Comm comm, int comm_sz, int my_rank)
{
    // n = bufsize, comm_sz = #process, assume op = MPI_SUM
    // copy sendbuf to recvbuf
    memcpy(recvbuf, sendbuf, n * sizeof(float));

    // block (sz can be 0)
    int block_sz = n / comm_sz;
    int last_block_sz = n - block_sz * (comm_sz - 1);

    // debug
    printf("[info] @ process %d/%d, bufsize %d, block_sz %d, last_block_sz %d\n", my_rank, comm_sz, n, block_sz, last_block_sz);
    // end of debug

    // STEP 1 : reduce-scatter
    for (int k = 0; k < (comm_sz - 1); k++)
    {
        MPI_Request sendrequest;
        int sendsize = ((comm_sz + my_rank - k) % comm_sz) == (comm_sz - 1) ? last_block_sz : block_sz;

        // debug
        printf("proc %d : k %d, block %d -> proc %d\n", my_rank, k, (comm_sz + my_rank - k) % comm_sz, (my_rank + 1) % comm_sz);
        // end of debug

        if (sendsize > 0)
        {
        MPI_Isend((float*)recvbuf + ((comm_sz + my_rank - k) % comm_sz) * block_sz,
                  sendsize,
                  MPI_FLOAT,
                  (my_rank + 1) % comm_sz,
                  0,
                  comm,
                  &sendrequest);
        }

        int recvsize = ((comm_sz + my_rank - k - 1) % comm_sz) == (comm_sz - 1) ? last_block_sz : block_sz;
        
        // debug
        printf("proc %d : k %d, block %d <- proc %d\n", my_rank, k, (comm_sz + my_rank - k - 1) % comm_sz, (comm_sz + my_rank - 1) % comm_sz);
        // end of debug

        if (recvsize > 0)
        {
        MPI_Request recvrequest;
        float* recvnum = new float[recvsize];
        MPI_Irecv(recvnum,
                  recvsize,
                  MPI_FLOAT,
                  (comm_sz + my_rank - 1) % comm_sz,
                  MPI_ANY_TAG,
                  comm,
                  &recvrequest);
        MPI_Status recvstatus;
        MPI_Wait(&recvrequest, &recvstatus);

        
        for (int i = 0; i < recvsize; i++)
            {
                (*((float*)recvbuf + ((comm_sz + my_rank - k - 1) % comm_sz) * block_sz + i)) += recvnum[i]; 
            }
        delete [] recvnum;
        }
    }

    // computation may not finish
    // so sync is necessary
    // hint : output ordering is not guaranteed in MPI programs.
    MPI_Barrier(comm);

    // STEP 2 : allgather
    for (int k = 0; k < (comm_sz - 1); k++)
    {
        MPI_Request sendrequest;
        int sendsize = ((comm_sz + my_rank + 1 - k) % comm_sz) == (comm_sz - 1) ? last_block_sz : block_sz;

    // debug
        printf("proc %d : k %d, block %d --cp--> proc %d\n", my_rank, k, (comm_sz + my_rank + 1 - k) % comm_sz, (my_rank + 1) % comm_sz);
    // end of debug

        if (sendsize > 0)
        {
            MPI_Isend((float*)recvbuf + ((comm_sz + my_rank + 1 - k) % comm_sz) * block_sz,
                      sendsize,
                      MPI_FLOAT,
                      (my_rank + 1) % comm_sz,
                      0,
                      comm,
                      &sendrequest);
        }

        int recvsize = ((comm_sz + my_rank - k) % comm_sz) == (comm_sz - 1) ? last_block_sz : block_sz;
        
        // debug
        printf("proc %d : k %d, block %d <--cp-- proc %d\n", my_rank, k, (comm_sz + my_rank - k) % comm_sz, (comm_sz + my_rank - 1) % comm_sz);
        // end of debug

        if (recvsize > 0)
        {
            MPI_Request recvrequest;
            float* recvnum = new float[recvsize];
            MPI_Irecv(recvnum,
                      recvsize,
                      MPI_FLOAT,
                      (comm_sz + my_rank - 1) % comm_sz,
                      MPI_ANY_TAG,
                      comm,
                      &recvrequest);
            MPI_Status recvstatus;
            MPI_Wait(&recvrequest, &recvstatus);

            for (int i = 0; i < recvsize; i++)
            {
                (*((float*)recvbuf + ((comm_sz + my_rank - k) % comm_sz) * block_sz + i)) = recvnum[i];
            }
            delete [] recvnum;
        }
    }
    MPI_Barrier(comm);
}


// reduce + bcast
void Naive_Allreduce(void* sendbuf, void* recvbuf, int n, MPI_Comm comm, int comm_sz, int my_rank)
{
    MPI_Reduce(sendbuf, recvbuf, n, MPI_FLOAT, MPI_SUM, 0, comm);
    MPI_Bcast(recvbuf, n, MPI_FLOAT, 0, comm);
}

int main(int argc, char *argv[])
{
    int ITER = atoi(argv[1]);
    int n = atoi(argv[2]);
    float* mpi_sendbuf = new float[n];
    float* mpi_recvbuf = new float[n];
    float* naive_sendbuf = new float[n];
    float* naive_recvbuf = new float[n];
    float* ring_sendbuf = new float[n];
    float* ring_recvbuf = new float[n];

    MPI_Init(nullptr, nullptr);
    int comm_sz;
    int my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    
    srand(time(NULL) + my_rank);
    for (int i = 0; i < n; ++i)
        mpi_sendbuf[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    memcpy(naive_sendbuf, mpi_sendbuf, n * sizeof(float));
    memcpy(ring_sendbuf, mpi_sendbuf, n * sizeof(float));

    // debug
    printf("proc %d init\n", my_rank);
    printbuf((float*)ring_sendbuf, n);
    // end of debug

    //warmup and check
    MPI_Allreduce(mpi_sendbuf, mpi_recvbuf, n, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    Naive_Allreduce(naive_sendbuf, naive_recvbuf, n, MPI_COMM_WORLD, comm_sz, my_rank);
    Ring_Allreduce(ring_sendbuf, ring_recvbuf, n, MPI_COMM_WORLD, comm_sz, my_rank);
    bool correct = true;
    for (int i = 0; i < n; ++i)
        if (abs(mpi_recvbuf[i] - ring_recvbuf[i]) > EPS)
        {
            correct = false;
            break;
        }

    // debug
    printf("correct %d, proc %d reduced\n", (int)correct, my_rank);
    printbuf((float*)mpi_recvbuf, n);
    printbuf((float*)ring_recvbuf, n);
    // end of debug

    if (correct)
    {
        auto beg = ch::high_resolution_clock::now();
        for (int iter = 0; iter < ITER; ++iter)
            MPI_Allreduce(mpi_sendbuf, mpi_recvbuf, n, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        auto end = ch::high_resolution_clock::now();
        double mpi_dur = ch::duration_cast<ch::duration<double>>(end - beg).count() * 1000; //ms

        beg = ch::high_resolution_clock::now();
        for (int iter = 0; iter < ITER; ++iter)
            Naive_Allreduce(naive_sendbuf, naive_recvbuf, n, MPI_COMM_WORLD, comm_sz, my_rank);
        end = ch::high_resolution_clock::now();
        double naive_dur = ch::duration_cast<ch::duration<double>>(end - beg).count() * 1000; //ms

        beg = ch::high_resolution_clock::now();
        for (int iter = 0; iter < ITER; ++iter)
            Ring_Allreduce(ring_sendbuf, ring_recvbuf, n, MPI_COMM_WORLD, comm_sz, my_rank);
        end = ch::high_resolution_clock::now();
        double ring_dur = ch::duration_cast<ch::duration<double>>(end - beg).count() * 1000; //ms
        
        if (my_rank == 0)
        {
            std::cout << "Correct." << std::endl;
            std::cout << "MPI_Allreduce:   " << mpi_dur << " ms." << std::endl;
            std::cout << "Naive_Allreduce: " << naive_dur << " ms." << std::endl;
            std::cout << "Ring_Allreduce:  " << ring_dur << " ms." << std::endl;
        }
    }
    else
        if (my_rank == 0)
            std::cout << "Wrong!" << std::endl;

    delete[] mpi_sendbuf;
    delete[] mpi_recvbuf;
    delete[] naive_sendbuf;
    delete[] naive_recvbuf;
    delete[] ring_sendbuf;
    delete[] ring_recvbuf;
    MPI_Finalize();
    return 0;
}
