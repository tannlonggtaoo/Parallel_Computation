echo -e 'proc_num:\n' > record.txt
srun -N 2 -n 2 ./allreduce 10 100000000 >> record.txt
srun -N 2 -n 4 ./allreduce 10 100000000 >> record.txt
srun -N 2 -n 8 ./allreduce 10 100000000 >> record.txt
echo -e 'comm_num:\n' >> record.txt
srun -N 2 -n 4 ./allreduce 10 100000000 >> record.txt
srun -N 2 -n 4 ./allreduce 10 300000000 >> record.txt
srun -N 2 -n 4 ./allreduce 10 500000000 >> record.txt
srun -N 2 -n 4 ./allreduce 10 700000000 >> record.txt
echo -e 'node_num:\n' >> record.txt
srun -N 1 -n 2 ./allreduce 10 100000000 >> record.txt
srun -N 1 -n 4 ./allreduce 10 100000000 >> record.txt
srun -N 1 -n 8 ./allreduce 10 100000000 >> record.txt
srun -N 1 -n 4 ./allreduce 10 100000000 >> record.txt
srun -N 1 -n 4 ./allreduce 10 300000000 >> record.txt
srun -N 1 -n 4 ./allreduce 10 500000000 >> record.txt
srun -N 1 -n 4 ./allreduce 10 700000000 >> record.txt

