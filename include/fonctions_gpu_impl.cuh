#ifndef HPCOMBI_PERM_FONCTIONS_GPU_IMPL_CUH
#define HPCOMBI_PERM_FONCTIONS_GPU_IMPL_CUH

#include <stdint.h>
#include <stdio.h>
#include <iostream>
#include <type_traits>
#include <typeinfo>

#include "fonctions_gpu.cuh"

#define CUDA_DEVICE 0

template <typename T>
void shufl_gpu(T* __restrict__ x, const T* __restrict__ y, const size_t size, float * timers);


// Instantiating template functions
template void shufl_gpu<uint8_t>(uint8_t* x, const uint8_t* y, const size_t size, float * timers);
template void shufl_gpu<uint16_t>(uint16_t* x, const uint16_t* y, const size_t size, float * timers);
template void shufl_gpu<uint32_t>(uint32_t* x, const uint32_t* y, const size_t size, float * timers);

// Allocating memory
//~ MemGpu memGpu(131072);
MemGpu memGpu(32);

template <typename T>
void shufl_gpu(T* __restrict__ x, const T* __restrict__ y, const size_t size, float * timers)
{
	//Creation des timers	
	cudaEvent_t start_all, stop_all;
	cudaEventCreate(&start_all);
	cudaEventCreate(&stop_all);
	cudaEventRecord(start_all);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaSetDevice(0);
	float tmp=0;

	//~ printf("size : %d\n", size);
	
	// Memory allocation on GPU
	cudaEventRecord(start);
	T *d_x, *d_y;
	//~ T *h_x, *h_y;

	if (std::is_same<uint8_t, T>::value){
		d_x = (T*)memGpu.d_x8;
		d_y = (T*)memGpu.d_y8;
	}
	else if (std::is_same<uint16_t, T>::value){
		d_x = (T*)memGpu.d_x16;
		d_y = (T*)memGpu.d_y16;
	}
	else if (std::is_same<uint32_t, T>::value){
		d_x = (T*)memGpu.d_x32;
		d_y = (T*)memGpu.d_y32;
	}
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(timers+2, start, stop);

	// Definition of grid and block sizes
	dim3 block(512,1);
	dim3 grid((size+block.x-1)/block.x,1);

	// Copy CPU to GPU
	cudaEventRecord(start);
	gpuErrchk( cudaMemcpy(d_x, x, size*sizeof(T), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_y, y, size*sizeof(T), cudaMemcpyHostToDevice) );
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(timers+1, start, stop);
	
		// Computation
		cudaEventRecord(start);
		//~ permute_gpu<T><<<grid, block, block.x*sizeof(T)>>>(d_x, d_y, size); // Algorithm using sfhl and shared memory
		permute_gpu_gen<T><<<grid, block>>>(d_x, d_y, size); // Simple algorithm
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(timers, start, stop);
		//~ printf("Computation %.3f ms\n", milliseconds);
	
	//Copy GPU to CPU
	cudaEventRecord(start);
	//~ memset(z, 0, size*sizeof(T));
	gpuErrchk( cudaMemcpy(x, d_y, size*sizeof(T), cudaMemcpyDeviceToHost) );
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&tmp, start, stop);

	// Update timers
	timers[1] += tmp;
	timers[1] += timers[0];
	timers[2] += timers[1];

}




void hash_gpu(const uint32_t* __restrict__ x, const int block_size, uint64_t* hashed, const int size, const int nb_vect, int kernel_num){
	cudaSetDevice(0);
	uint32_t* d_x;
	uint64_t* d_hashed;
	
	gpuErrchk( cudaMalloc((void**)&d_x, size * nb_vect * sizeof(uint32_t)) );
	gpuErrchk( cudaMalloc((void**)&d_hashed, nb_vect * sizeof(uint64_t)) );

	gpuErrchk( cudaMemcpy(d_x, x, size * nb_vect * sizeof(uint32_t), cudaMemcpyHostToDevice) );
	dim3 block;
	dim3 grid;
	if(kernel_num == 1){
		block.x = block_size;
		block.y = 1;
		grid.x = nb_vect;
		grid.y = 1;
	}
	else if(kernel_num > 1){
		block.x = 32;
		block.y = block_size;
		grid.x = 1;
		grid.y = (nb_vect + block.y-1)/block.y;
	}
		hpcombi<<<grid, block>>>(d_x, d_hashed, size, nb_vect);
	gpuErrchk( cudaDeviceSynchronize() );
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaMemcpy(hashed, d_hashed, nb_vect * sizeof(uint64_t), cudaMemcpyDeviceToHost) );

	cudaFree(d_x);
	cudaFree(d_hashed);

}


void hpcombi_gpu(const int* __restrict__ words, const uint32_t* __restrict__ d_gen, uint64_t* hashed, 
				int block_size, const int size, const int size_word, const int nb_words, const int nb_gen){
	cudaSetDevice(CUDA_DEVICE);
	uint32_t* d_x;
	int* d_words;
	uint64_t* d_hashed;	

	gpuErrchk( cudaMalloc((void**)&d_words, size_word*nb_words * sizeof(int)) ); // TODO do not reallocate
	gpuErrchk( cudaMalloc((void**)&d_x, size * nb_words*nb_gen * sizeof(uint32_t)) ); // TODO do not reallocate
	gpuErrchk( cudaMalloc((void**)&d_hashed, nb_words*nb_gen * sizeof(uint64_t)) ); // TODO do not reallocate

	dim3 blockInit(128, 1);
	dim3 gridInit(( size*nb_words*nb_gen + blockInit.x-1 )/blockInit.x, 1);
		initId_kernel<<<gridInit, blockInit>>>(d_x, size, nb_words, nb_gen);
	gpuErrchk( cudaDeviceSynchronize() );
	gpuErrchk( cudaPeekAtLastError() );

	gpuErrchk( cudaMemcpy(d_words, words, size_word*nb_words * sizeof(int), cudaMemcpyHostToDevice) );
	for(int i=0; i<5; i++){
		int gridy = (nb_words*nb_gen + block_size-1)/block_size;		
		if(gridy > 65535 && block_size < 32){
			block_size *= 2;
		}
		else if(gridy > 65535){
			printf("To much words (%d)\n", gridy);
			exit(1);
		}
	}
	dim3 block(32, block_size);
	dim3 grid(1, (nb_words*nb_gen + block.y-1)/block.y);
	//~ printf("Blockx : %d, Blocky : %d, Gridx : %d, Gridy : %d\n", block.x, block.y, grid.x, grid.y);
		//~ hpcombi_kernel<<<grid, block>>>(d_x, d_gen, d_words, d_hashed, size, nb_words, size_word, nb_gen);
		permute_all_kernel<<<grid, block>>>(d_x, d_gen, d_words, size, nb_words, size_word, nb_gen);
	gpuErrchk( cudaDeviceSynchronize() );
	gpuErrchk( cudaPeekAtLastError() );
		hash_kernel3<<<grid, block>>>(d_x, d_hashed, size, nb_words*nb_gen);
	gpuErrchk( cudaDeviceSynchronize() );
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaMemcpy(hashed, d_hashed, nb_words*nb_gen * sizeof(uint64_t), cudaMemcpyDeviceToHost) );

	cudaFree(d_x);

	cudaFree(d_words);
	cudaFree(d_hashed);
}

bool equal_gpu(const int* __restrict__ word1, const int* __restrict__ word2, const uint32_t* __restrict__ d_gen, 
				int block_size, const int size, const int size_word, const int nb_gen){
	cudaSetDevice(CUDA_DEVICE);
	uint32_t* d_x;
	int* d_words;
	bool equal;
	bool* d_equal;

	gpuErrchk( cudaMalloc((void**)&d_words, size_word*2 * sizeof(int)) ); // TODO do not reallocate
	gpuErrchk( cudaMalloc((void**)&d_x, size * 2 * sizeof(uint32_t)) ); // TODO do not reallocate
	gpuErrchk( cudaMalloc((void**)&d_equal, sizeof(bool)) );

	dim3 blockInit(128, 1);
	dim3 gridInit(( size*2 + blockInit.x-1 )/blockInit.x, 1);
		initId_kernel<<<gridInit, blockInit>>>(d_x, size, 2, 1);
	gpuErrchk( cudaDeviceSynchronize() );
	gpuErrchk( cudaPeekAtLastError() );

	gpuErrchk( cudaMemcpy(d_words, word1, size_word * sizeof(int), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_words+size_word, word2, size_word * sizeof(int), cudaMemcpyHostToDevice) );

	dim3 block(32, block_size);
	dim3 grid(1, (2 + block.y-1)/block.y);
		equal_kernel<<<grid, block>>>(d_x, d_gen, d_words, d_equal, size, size_word);
	gpuErrchk( cudaDeviceSynchronize() );
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaMemcpy(&equal, d_equal, sizeof(bool), cudaMemcpyDeviceToHost) );

	cudaFree(d_x);
	cudaFree(d_words);
	cudaFree(d_equal);
	
	return equal;
}

void hash_id_gpu(uint64_t* hashed, int block_size, const int size){
	cudaSetDevice(CUDA_DEVICE);
	uint32_t* d_x;
	uint64_t* d_hashed;	

	gpuErrchk( cudaMalloc((void**)&d_x, size * sizeof(uint32_t)) );
	gpuErrchk( cudaMalloc((void**)&d_hashed, 1 * sizeof(uint64_t)) );

	dim3 blockInit(128, 1);
	dim3 gridInit(( size + blockInit.x-1 )/blockInit.x, 1);
		initId_kernel<<<gridInit, blockInit>>>(d_x, size, 1, 1);
	gpuErrchk( cudaDeviceSynchronize() );
	gpuErrchk( cudaPeekAtLastError() );

	dim3 block(32, block_size);
	dim3 grid(1, (1 + block.y-1)/block.y);
		hash_kernel3<<<grid, block>>>(d_x, d_hashed, size, 1);
	gpuErrchk( cudaDeviceSynchronize() );
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaMemcpy(hashed, d_hashed, 1 * sizeof(uint64_t), cudaMemcpyDeviceToHost) );

	cudaFree(d_x);
	cudaFree(d_hashed);
}

void malloc_gen(uint32_t** __restrict__ d_gen, const uint32_t* __restrict__ gen, const int size, const int nb_gen){
	cudaSetDevice(CUDA_DEVICE);
	gpuErrchk( cudaMalloc((void**)d_gen, size*nb_gen * sizeof(uint32_t)) );
	gpuErrchk( cudaMemcpy(*d_gen, gen, size*nb_gen * sizeof(uint32_t), cudaMemcpyHostToDevice) );	
}

void free_gen(uint32_t** __restrict__ d_gen){
	cudaFree(*d_gen);
}
#endif  // HPCOMBI_PERM_FONCTIONS_GPU_IMPL_CUH
