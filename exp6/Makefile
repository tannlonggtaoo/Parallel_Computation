.PHONY: clean

all: test_gmem test_smem g1 g2 g4 g8 g16 sB2S1 sB2S2 sB2S4 sB2S8 sB2S16 sB2S32 sB4S1 sB4S2 sB4S4 sB4S8 sB4S16 sB4S32 sB8S1 sB8S2 sB8S4 sB8S8 sB8S16 sB8S32
	
%: %.cu
	nvcc $^ -o $@ -O2 -code sm_60 -arch compute_60

clean:
	rm test_gmem test_smem

