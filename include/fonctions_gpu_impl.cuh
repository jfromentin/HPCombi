#ifndef HPCOMBI_PERM_FONCTIONS_GPU_IMPL_CUH
#define HPCOMBI_PERM_FONCTIONS_GPU_IMPL_CUH

#include <stdint.h>
#include <stdio.h>
#include <iostream>
#include <type_traits>
#include <typeinfo>
#include <cuda_profiler_api.h>

#include "fonctions_gpu.cuh"

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

// GPU Infos
size_t cudaSetDevice_cpu(){
	int count=0;
	cudaGetDeviceCount(&count);
	printf("count %d\n", count);
	if(count>1)
		cudaSetDevice(CUDA_DEVICE);
	else
		cudaSetDevice(0);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, CUDA_DEVICE);
	printf("%s :\n", prop.name); 
	printf("	Global memory available : %.2f Go\n", static_cast<float>(prop.totalGlobalMem)*1e-9); 
	printf("	Shared memory per block : %.2f Ko\n", static_cast<float>(prop.sharedMemPerBlock)*1e-3);
	printf("	Registers per block : %d\n", prop.regsPerBlock);
	printf("	Clock Rate : %.2f GHz\n", static_cast<float>(prop.clockRate)*1e-6);
	printf("	MultiProcessor Count : %d\n", prop.multiProcessorCount);
	printf("	Max Threads Per Block : %d\n", prop.maxThreadsPerBlock);
	printf("	Max Threads Dim : %d x %d x %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
	printf("	Max Grid Size : %d x %d x %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
	printf("\n");
	return prop.totalGlobalMem;
}

void hpcombi_gpu(Vector_cpugpu<int8_t>& words, Vector_gpu<uint32_t>& workSpace, const uint32_t* __restrict__ d_gen, Vector_cpugpu<uint64_t>& hashed, 
				const int size, const int size_word, const int8_t nb_gen, size_t memory){
	//~ cudaProfilerStart();
	//~ cudaSetDevice(CUDA_DEVICE);
	//~ float timer;
	memory /= 1.05;
	int nb_words = words.size()/size_word;
    hashed.resize(static_cast<size_t>(nb_words)*nb_gen * NB_HASH_FUNC, 2);	
	words.copyHostToDevice();
	
	size_t constMem = words.capacity()*sizeof(int8_t) 
					+ sizeof(uint32_t)*nb_gen*size_word 
					+ hashed.resize(static_cast<size_t>(nb_words)*nb_gen * NB_HASH_FUNC, 0);
	memory -= constMem;
	size_t varMem = workSpace.resize(static_cast<size_t>(size) * nb_words*nb_gen, 0, memory);
	//~ printf("varMem : %.2f Mo\n", (float)varMem*1e-6);
	int div = (varMem + memory-1) / memory;
	int paquetMax = (nb_words + div-1) / div;
	int paquet = paquetMax;

	//~ if(workSpace.capacity < static_cast<size_t>(size) * paquetMax*nb_gen){
		//~ varMem = workSpace.resize(static_cast<size_t>(size) * paquetMax*nb_gen, 0, memory);
		//~ printf("div : %d, paquet : %d\n", div, paquetMax);
		//~ printf("Allocating : %.2f Mo, available : %.2f Mo, constMem : %.2f Mo\n", (float)varMem*1e-6, (float)memory*1e-6, (float)constMem*1e-6);
	//~ }
	varMem = workSpace.resize(static_cast<size_t>(size) * paquetMax*nb_gen, 1, memory);
	
	for(int pass=0; pass<div; pass++){
		if(pass == div-1)
			paquet = nb_words-paquetMax*pass;
		//~ if(div > 1)
			//~ printf("pass %d/%d, paquet : %d\n", pass+1, div, paquet);
		if(paquet > 0){
			int threadPerPerm = min(size, 16384);
			dim3 blockPerm, gridPerm;
			for(int i=0; i<10; i++){
				blockPerm.x = min(threadPerPerm, 1024);
				blockPerm.y = max(1024/threadPerPerm, 1);
				gridPerm.x = (threadPerPerm + blockPerm.x -1)/blockPerm.x;
				gridPerm.y = (paquet*nb_gen + blockPerm.y-1)/blockPerm.y;
				if(threadPerPerm < size && gridPerm.y < 65536 && gridPerm.x < pow(2,31))
					break;
				threadPerPerm /= 2;
			}
			//~ printf("blockPerm.x : %d, blockPerm.y : %d\n", blockPerm.x, blockPerm.y);
			//~ printf("gridPerm.x : %d, gridPerm.y : %d\n", gridPerm.x, gridPerm.y);
				//~ compose_kernel<<<gridPerm, blockPerm, size_blocky*sizeof(int8_t)>>>(workSpace.device(), d_gen, words.device(), size, paquet, size_word, nb_gen);		
				compose_kernel<<<gridPerm, blockPerm>>>(workSpace.device(), d_gen, words.device() + pass*paquetMax*size_word, size, paquet, size_word, nb_gen);
			gpuErrchk( cudaDeviceSynchronize() );
			gpuErrchk( cudaPeekAtLastError() );

			dim3 blockHash(32, 4), gridHash(1, 1);
			for(int i=0; i<6; i++){
				if(blockHash.x <= size)
					break;
				blockHash.x /= 2;
			}
			for(int i=0; i<6; i++){
				gridHash.y = (paquet*nb_gen + blockHash.y-1)/blockHash.y;		
				if(gridHash.y > 65536 && blockHash.y*blockHash.x < 1024)
					blockHash.y *= 2;
			}
				hash_kernel<<<gridHash, blockHash>>>(workSpace.device(), hashed.device() + pass*paquetMax*nb_gen, size, paquet*nb_gen);
			
			gpuErrchk( cudaDeviceSynchronize() );
			gpuErrchk( cudaPeekAtLastError() );
			
			hashed.copyDeviceToHost(pass*paquetMax*nb_gen, paquet*nb_gen);
		}
	}
	//~ cudaProfilerStop();
}

bool equal_gpu(const Key& key1, const Key& key2, uint32_t* d_gen, int8_t* d_words, const int size, const int8_t nb_gen, Vector_cpugpu<int>& equal){
	//~ cudaProfilerStart();
	gpuErrchk( cudaMemcpy(d_words, key1.data(), NODE * sizeof(int8_t), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_words + NODE, key2.data(), NODE * sizeof(int8_t), cudaMemcpyHostToDevice) );

	const dim3 block(128, 1);
	const dim3 grid((min(size, 16384) + block.x-1)/block.x, 1);
		equal_kernel<<<grid, block>>>(d_gen, d_words, equal.device(), size, NODE, nb_gen);
	gpuErrchk( cudaDeviceSynchronize() );
	gpuErrchk( cudaPeekAtLastError() );
	equal.copyDeviceToHost();

	const bool out = (equal[0] == size) ? true:false;
	//~ cudaProfilerStop();
	return out;
}

void hash_id_gpu(Vector_cpugpu<uint64_t>& hashed, Vector_gpu<uint32_t>& workSpace, const int size){
	//~ cudaSetDevice(CUDA_DEVICE);
	workSpace.resize(static_cast<size_t>(size));
	
	const dim3 blockInit(32, 4);
	const dim3 gridInit(1, ( 1 + blockInit.y-1 )/blockInit.y);
		initId_kernel<<<gridInit, blockInit>>>(workSpace.device(), size, 1);
	gpuErrchk( cudaDeviceSynchronize() );
	gpuErrchk( cudaPeekAtLastError() );

	const dim3 block(32, 4);
	const dim3 grid(1, (1 + block.y-1)/block.y);
		hash_kernel<<<grid, block>>>(workSpace.device(), hashed.device(), size, 1);
	gpuErrchk( cudaDeviceSynchronize() );
	gpuErrchk( cudaPeekAtLastError() );
	hashed.copyDeviceToHost();
}

void malloc_gen(uint32_t*& __restrict__ d_gen, const uint32_t* __restrict__ gen, const int size, const int8_t nb_gen){
	//~ cudaSetDevice(CUDA_DEVICE);
	gpuErrchk( cudaMalloc((void**)&d_gen, size*nb_gen * sizeof(uint32_t)) );
	gpuErrchk( cudaMemcpy(d_gen, gen, size*nb_gen * sizeof(uint32_t), cudaMemcpyHostToDevice) );	
}

void malloc_words(int8_t*& __restrict__ d_words, const int size){
	//~ cudaSetDevice(CUDA_DEVICE);
	gpuErrchk( cudaMalloc((void**)&d_words, size*2 * sizeof(uint8_t)) );
}

void free_gen(uint32_t*& __restrict__ d_gen){
	gpuErrchk( cudaFree(d_gen) );
}

void free_words(int8_t*& __restrict__ d_words){
	gpuErrchk( cudaFree(d_words) );
}
#endif  // HPCOMBI_PERM_FONCTIONS_GPU_IMPL_CUH
