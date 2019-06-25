
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

__global__ void check_subseq(int *data, int *comb_base, int *comb_idx_base, unsigned char *comb_map, unsigned n, unsigned k, unsigned comb_count)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int *combination = comb_base + idx * k;
    int *comb_idx = comb_idx_base + idx * k;
    int *seq = data;
    int *pat = data + n;

	if (idx >= comb_count) {
		return;
	}

    int i = 0;
    int p = idx;
    for (int x = 0; x < k; x++) {
        int c = newton(n-i-1, k-x-1);
        while (c <= p) {
            p -= c;
            i++;
            c = newton(n-i-1, k-x-1);
        }
        combination[x] = seq[i];
        comb_idx[x] = i;
        i++;
    }

    for (i = 0; i < k; i++)
        for (int j = 0; j < k; j++)
            if (i != j && ((pat[i] == pat[j]) ^ (combination[i] == combination[j])))
                return;
    comb_map[idx] = 1;
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
    if (k > n) {
        std::cerr << "Pattern is longer than sequence" << std::endl;
        return 1;
    }

	thrust::host_vector<int> h_combinations;
	thrust::device_vector<int> d_combinations;
	thrust::host_vector<int> h_comb_idx;
	thrust::device_vector<int> d_comb_idx;
	thrust::host_vector<unsigned char> h_comb_map;
	thrust::device_vector<unsigned char> d_comb_map;
	thrust::host_vector<int> h_data;
	thrust::device_vector<int> d_data;

	for (unsigned int i = 0; i < n; i++) {
		h_data.push_back(argv[1][i]);
	}
	for (unsigned int i = 0; i < k; i++) {
		h_data.push_back(argv[2][i]);
	}
	d_data = h_data;
	comb_count = newton(n, k);
	d_combinations.resize(comb_count * k, 0);
	d_comb_idx.resize(comb_count * k, 0);
	d_comb_map.resize(comb_count, 0);

	check_subseq << <1, comb_count >> > (d_data.data().get(), d_combinations.data().get(), d_comb_idx.data().get(), d_comb_map.data().get(), n, k, comb_count);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "kernel error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

	h_combinations = d_combinations;
	h_comb_idx = d_comb_idx;
    h_comb_map = d_comb_map;
    for (unsigned int i = 0; i < comb_count; i++) {
        if (! h_comb_map[i])
            continue;
        printf("%d:", i);
        for (unsigned int j = 0; j < k; j++) 
            printf(" %c=%c[%d]", argv[2][j], h_combinations[i*k + j], h_comb_idx[i*k + j]);
        putchar('\n');
    }

	return 0;
}
