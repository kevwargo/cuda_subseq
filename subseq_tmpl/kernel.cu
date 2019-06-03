
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__device__ __host__ unsigned newton(unsigned n, unsigned k)
{
	unsigned top = 1;
	unsigned bottom = 1;
	if (k >= n) {
		return 1;
	}

	for (unsigned i = k + 1; i <= n; i++) {
		top *= i;
	}
	for (unsigned i = 2; i <= n - k; i++) {
		bottom *= i;
	}
	return top / bottom;
}

__global__ void check_subseq(unsigned patlen, unsigned seqlen, int *pat, int *seq, unsigned n)
{

}

int main()
{

}

