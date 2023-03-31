echo -e 'proc_num:\n' > record.txt
srun -N 2 -n 2 ./allreduce 10 100000000 >> record.txt
srun -N 2 -n 4 ./allreduce 10 100000000 >> record.txt
srun -N 2 -n 8 ./allreduce 10 100000000 >> record.txt
srun -N 2 -n 16 ./allreduce 10 100000000 >> record.txt
srun -N 2 -n 32 ./allreduce 10 100000000 >> record.txt
srun -N 2 -n 64 ./allreduce 10 100000000 >> record.txt
srun -N 2 -n 128 ./allreduce 10 100000000 >> record.txt
echo -e 'comm_num:\n' >> record.txt
srun -N 2 -n 16 ./allreduce 10 100000000 >> record.txt
srun -N 2 -n 16 ./allreduce 10 200000000 >> record.txt
srun -N 2 -n 16 ./allreduce 10 300000000 >> record.txt
srun -N 2 -n 16 ./allreduce 10 400000000 >> record.txt
srun -N 2 -n 16 ./allreduce 10 500000000 >> record.txt
srun -N 2 -n 16 ./allreduce 10 600000000 >> record.txt
srun -N 2 -n 16 ./allreduce 10 700000000 >> record.txt
echo -e 'node_num:\n' >> record.txt
srun -N 1 -n 2 ./allreduce 10 100000000 >> record.txt
srun -N 1 -n 4 ./allreduce 10 100000000 >> record.txt
srun -N 1 -n 8 ./allreduce 10 100000000 >> record.txt
srun -N 1 -n 16 ./allreduce 10 100000000 >> record.txt
srun -N 1 -n 16 ./allreduce 10 300000000 >> record.txt
srun -N 1 -n 16 ./allreduce 10 50000000 >> record.txt
srun -N 1 -n 16 ./allreduce 10 700000000 >> record.txt

