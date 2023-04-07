#!/bin/bash

# run on 1 machine * 28 process, feel free to change it!
if [[ $2 -lt 500000 ]]
then
	N=1
else
	N=2
fi

if [[ $2 -lt 1500 ]]
then
	n=1
elif [[ $2 -lt 5000 ]]
then
	n=5
elif [[ $2 -lt 50000 ]]
then
	n=20
elif [[ $2 -lt 500000 ]]
then
	n=28
else
	n=56
fi

srun -N $N -n $n --cpu-bind sockets $*

