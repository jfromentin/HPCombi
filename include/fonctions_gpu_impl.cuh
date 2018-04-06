#ifndef HPCOMBI_PERM_FONCTIONS_GPU_IMPL_CUH
#define HPCOMBI_PERM_FONCTIONS_GPU_IMPL_CUH

#include <stdint.h>
#include <stdio.h>
#include <iostream>
#include <type_traits>
#include <typeinfo>

#include "fonctions_gpu.cuh"

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




void hash_gpu(const uint32_t* __restrict__ x, const int block_size, uint64_t* hashed, const size_t size, const size_t nb_vect) {
	cudaSetDevice(0);
	uint32_t* d_x;
	uint64_t* d_hashed;
	
	gpuErrchk( cudaMalloc((void**)&d_x, size * nb_vect * sizeof(uint32_t)) );
	gpuErrchk( cudaMalloc((void**)&d_hashed, nb_vect * sizeof(uint64_t)) );

	gpuErrchk( cudaMemcpy(d_x, x, size * nb_vect * sizeof(uint32_t), cudaMemcpyHostToDevice) );
	//~ dim3 block(block_size,1);
	//~ dim3 grid(nb_vect,1);
	dim3 block(32, block_size);
	dim3 grid(1,(nb_vect + block_size-1)/block_size);
		hpcombi<<<grid, block>>>(d_x, d_hashed, size, nb_vect);
	gpuErrchk( cudaDeviceSynchronize() );
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaMemcpy(hashed, d_hashed, nb_vect * sizeof(uint64_t), cudaMemcpyDeviceToHost) );

	cudaFree(d_x);
	cudaFree(d_hashed);

}
#endif  // HPCOMBI_PERM_FONCTIONS_GPU_IMPL_CUH
