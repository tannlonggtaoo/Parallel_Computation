#include "mpi.h"
int main(void)
{
	int rank, value;
	MPI_Init(NULL,NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	do
	{
		if (rank == 0)
		{
			scanf("%d", &value);
		}
		MPI_Bcast(&value, 1, MPI_INT, 0, MPI_COMM_WORLD);

		printf("Process %d got %d\n", rank, value);
	}while(value >= 0);
	MPI_Finalize();
	return 0;

}

