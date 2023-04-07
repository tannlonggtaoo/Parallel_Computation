#!/bin/bash

# run on 1 machine * 28 process, feel free to change it!
for ((i=100;i<=100000000;i*=10))
do
	N=1
	for ((n=1;n<=28;n+=3))
	do
		echo "$N*$n,$i"
		srun -N $N -n $n --cpu-bind sockets ~/PA1/odd_even_sort $i ~/PA1/data/100000000.dat
	done

	N=2
	for ((n=1;n<=56;n+=5))
	do
		echo "$N*$n,$i"
		srun -N $N -n $n --cpu-bind sockets ~/PA1/odd_even_sort $i ~/PA1/data/100000000.dat
	done
done
