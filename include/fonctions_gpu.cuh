#ifndef HPCOMBI_PERM_FONCTIONS_GPU_CUH
#define HPCOMBI_PERM_FONCTIONS_GPU_CUH
#if COMPILE_CUDA==1	
	#define CUDA_DEVICE 1
	#include <cuda_runtime.h>
	#include <cuda.h>
	#include "vector_gpu.cuh"
	#include "RennerGpu.hpp"
	
	void hash_gpu(const uint32_t* __restrict__, const int, uint64_t*, const int, const int, int);
	void hpcombi_gpu(Vector_cpugpu<int8_t>&, Vector_gpu<uint32_t>&, const uint32_t* __restrict__, Vector_cpugpu<uint64_t>&, 
				const int, const int, const int8_t, size_t);
	void hash_id_gpu(Vector_cpugpu<uint64_t>&, Vector_gpu<uint32_t>&, const int);
	bool equal_gpu(const Key&, const Key&, uint32_t*, int8_t*, const int, const int8_t, Vector_cpugpu<int>&);
	void malloc_gen(uint32_t*& __restrict__, const uint32_t* __restrict__, const int, const int8_t);
	void free_gen(uint32_t*& __restrict__);
	void malloc_words(int8_t*& __restrict__, const int);
	void free_words(int8_t*& __restrict__);
	size_t cudaSetDevice_cpu();

#endif  // USE_CUDA
#endif  // HPCOMBI_PERM_FONCTIONS_GPU_CUH
