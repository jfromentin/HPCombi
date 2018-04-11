#ifndef HPCOMBI_PERM_FONCTIONS_GPU_CUH
#define HPCOMBI_PERM_FONCTIONS_GPU_CUH
#if COMPILE_CUDA==1

	//~ #include <cuda.h>
	//~ #include <helper_cuda.h>
	//~ #include <helper_functions.h>
	#include <cuda_runtime.h>
	template <typename T>
	void shufl_gpu(T* __restrict__ x, const T* __restrict__ y, const size_t size, float * timers);
	void hash_gpu(const uint32_t* __restrict__ x, const int block_size, uint64_t* hashed, const int size, const int nb_vect, int kernel_num);
	void hpcombi_gpu(const int* __restrict__ words, const uint32_t* __restrict__ d_gen, uint64_t* hashed, 
				int block_size, const int size, const int size_word, const int nb_words, const int nb_gen);
	void hash_id_gpu(uint64_t* hashed, int block_size, const int size);
	bool equal_gpu(const int* __restrict__ word1, const int* __restrict__ word2, const uint32_t* __restrict__ d_gen, 
				int block_size, const int size, const int size_word, const int nb_gen);
	void malloc_gen(uint32_t** __restrict__ d_gen, const uint32_t* __restrict__ gen, const int size, const int number);
	void free_gen(uint32_t** __restrict__ d_gen);

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
	

	// Memory pool
	class MemGpu {
	
		public :
		uint8_t *h_x8, *h_y8;
		uint16_t *h_x16, *h_y16;
		uint32_t *h_x32, *h_y32;

		uint8_t *d_x8, *d_y8;
		uint16_t *d_x16, *d_y16;
		uint32_t *d_x32, *d_y32;
	
		MemGpu (size_t size) {
			cudaSetDevice(0);
			gpuErrchk( cudaHostAlloc((void**)&h_x8, size*sizeof(uint8_t), 0) );
			gpuErrchk( cudaHostAlloc((void**)&h_y8, size*sizeof(uint8_t), 0) );
			
			gpuErrchk( cudaHostAlloc((void**)&h_x16, size*sizeof(uint16_t), 0) );
			gpuErrchk( cudaHostAlloc((void**)&h_y16, size*sizeof(uint16_t), 0) );
			
			gpuErrchk( cudaHostAlloc((void**)&h_x32, size*sizeof(uint32_t), 0) );
			gpuErrchk( cudaHostAlloc((void**)&h_y32, size*sizeof(uint32_t), 0) );

			
			gpuErrchk( cudaMalloc((void**)&d_x8, size*sizeof(uint8_t)) );
			gpuErrchk( cudaMalloc((void**)&d_y8, size*sizeof(uint8_t)) );
	
			gpuErrchk( cudaMalloc((void**)&d_x16, size*sizeof(uint16_t)) );
			gpuErrchk( cudaMalloc((void**)&d_y16, size*sizeof(uint16_t)) );
	
			gpuErrchk( cudaMalloc((void**)&d_x32, size*sizeof(uint32_t)) );
			gpuErrchk( cudaMalloc((void**)&d_y32, size*sizeof(uint32_t)) );
		}
		~MemGpu () {
			gpuErrchk( cudaFreeHost(h_x8) );
			gpuErrchk( cudaFreeHost(h_y8) );
			gpuErrchk( cudaFreeHost(h_x16) );
			gpuErrchk( cudaFreeHost(h_y16) );
			gpuErrchk( cudaFreeHost(h_x32) );
			gpuErrchk( cudaFreeHost(h_y32) );

			gpuErrchk( cudaFree(d_x8) );
			gpuErrchk( cudaFree(d_y8) );
			gpuErrchk( cudaFree(d_x16) );
			gpuErrchk( cudaFree(d_y16) );
			gpuErrchk( cudaFree(d_x32) );
			gpuErrchk( cudaFree(d_y32) );
			
		}
	};
	
#endif  // USE_CUDA
#endif  // HPCOMBI_PERM_FONCTIONS_GPU_CUH
