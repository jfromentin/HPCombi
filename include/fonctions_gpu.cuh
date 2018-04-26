#ifndef HPCOMBI_PERM_FONCTIONS_GPU_CUH
#define HPCOMBI_PERM_FONCTIONS_GPU_CUH
#if COMPILE_CUDA==1	
	#define CUDA_DEVICE 0
	#include <cuda_runtime.h>
	#include "vector_gpu.cuh"
	#include "RennerGpu.hpp"
	
	void hash_gpu(const uint32_t* __restrict__ x, const int block_size, uint64_t* hashed, const int size, const int nb_vect, int kernel_num);
	void hpcombi_gpu(Vector_cpugpu<int8_t>* words, Vector_gpu<uint32_t>* d_x, Vector_gpu<uint32_t>* d_y, const uint32_t* __restrict__ d_gen, Vector_cpugpu<uint64_t>* hashed, 
				const int size, const int size_word, const int8_t nb_gen);
	void hash_id_gpu(Vector_cpugpu<uint64_t>* hashed, Vector_gpu<uint32_t>* d_x, int block_size, const int size);
	bool equal_gpu(const key* key1, const key* key2, int block_size, const int size, const int size_word, const int8_t nb_gen);
	void malloc_gen(uint32_t** __restrict__ d_gen, const uint32_t* __restrict__ gen, const int size, const int8_t nb_gen);
	void free_gen(uint32_t** __restrict__ d_gen);
	void cudaSetDevice_cpu();

	// GPU error catching
	#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
	inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
	{
	   if (code != cudaSuccess) 
	   {
	      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	      if (abort) exit(code);
	   }
	}
	
#endif  // USE_CUDA
#endif  // HPCOMBI_PERM_FONCTIONS_GPU_CUH
