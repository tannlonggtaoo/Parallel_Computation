#!/bin/bash

# run on 1 machine * 28 process, feel free to change it!
echo 1*1
srun -N 1 -n 1 --cpu-bind sockets $*

echo 1*2
srun -N 1 -n 2 --cpu-bind sockets $*

echo 1*4
srun -N 1 -n 4 --cpu-bind sockets $*

echo 1*8
srun -N 1 -n 8 --cpu-bind sockets $*

echo 1*16
srun -N 1 -n 16 --cpu-bind sockets $*

echo 2*16
srun -N 2 -n 32 --cpu-bind sockets $*

