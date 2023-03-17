#include "mpi.h"
int main(void)
{
	int comm_sz, my_rank;
	MPI_Init(NULL,NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	printf("process %d of %d\n",my_rank,comm_sz);
	//MPI_Finalize();
	return 0;
}