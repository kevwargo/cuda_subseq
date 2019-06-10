
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <stdio.h>
#include <string.h>
#include <iostream>

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

__global__ void check_subseq(unsigned n, unsigned k, int *seq, int *pat, int *combinations, unsigned size)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int data[k];

	if (idx >= size) {
		return;
	}


}

int main(int argc, char **argv)
{
	unsigned n, k, comb_count;
	if (argc < 3) {
		std::cerr << "Not enough args" << std::endl;
		return 1;
	}
	n = strlen(argv[1]);
	k = strlen(argv[2]);

	thrust::host_vector<int> h_combinations;
	thrust::device_vector<int> d_combinations;
	thrust::host_vector<int> h_seq;
	thrust::device_vector<int> d_seq;
	thrust::host_vector<int> h_pat;
	thrust::device_vector<int> d_pat;

	for (int i = 0; i < n; i++) {
		h_seq.push_back(argv[1][i]);
	}
	for (int i = 0; i < k; i++) {
		h_pat.push_back(argv[2][i]);
	}
	d_seq = h_seq;
	d_pat = h_pat;
	comb_count = newton(n, k);
	d_combinations.reserve(comb_count * k);

	check_subseq << <1, comb_count >> > (n, k, d_seq.data().get(), d_pat.data().get(), d_combinations.data().get(), comb_count);

	h_combinations = d_combinations;

	return 0;
}

