UNROLL_N ?= 1

.PHONY: clean

all: main

main: main.cpp
	icc $^ -o $@${UNROLL_N} -O3 -DUNROLL_N=${UNROLL_N}

clean:
	rm main

