#ifndef HPCOMBI_PERM_FONCTIONS_GPU_IMPL_CUH
#define HPCOMBI_PERM_FONCTIONS_GPU_IMPL_CUH

#include <stdint.h>
#include <stdio.h>
#include <iostream>
#include <type_traits>
#include <typeinfo>
#include <cuda_profiler_api.h>

#include "fonctions_gpu.cuh"

size_t cudaSetDevice_cpu(){ 
	cudaSetDevice(CUDA_DEVICE);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, CUDA_DEVICE);
	printf("%s :\n", prop.name); 
	printf("	Global memory available : %.2f Go\n", (float)prop.totalGlobalMem*1e-9); 
	printf("	Shared memory per block : %.2f Ko\n", (float)prop.sharedMemPerBlock*1e-3);
	printf("	Registers per block : %d\n", prop.regsPerBlock);
	printf("	Clock Rate : %.2f GHz\n", (float)prop.clockRate*1e-6);
	printf("	MultiProcessor Count : %d\n", prop.multiProcessorCount);
	printf("	Max Threads Per Block : %d\n", prop.maxThreadsPerBlock);
	printf("	Max Threads Dim : %d x %d x %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
	printf("	Max Grid Size : %d x %d x %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
	printf("\n");
	return prop.totalGlobalMem;
}

void hpcombi_gpu(Vector_cpugpu<int8_t>* words, Vector_gpu<uint32_t>* d_x, const uint32_t* __restrict__ d_gen, Vector_cpugpu<uint64_t>* hashed, 
				const int size, const int size_word, const int8_t nb_gen, size_t memory){
	//~ cudaProfilerStart();
	//~ cudaSetDevice(CUDA_DEVICE);
	//~ float timer;
	memory /= 4.1;
	int nb_words = words->size/size_word;
    hashed->resize((size_t)nb_words*nb_gen * NB_HASH_FUNC, 2);	
	words->copyHostToDevice();
	
	size_t constMem = words->capacity*sizeof(int8_t) + sizeof(uint32_t)*nb_gen*size_word + hashed->resize((size_t)nb_words*nb_gen * NB_HASH_FUNC, 0);
	memory -= constMem;
	size_t varMem = d_x->resize((size_t)size * nb_words*nb_gen, 0, memory);
	//~ printf("varMem : %.2f Mo\n", (float)varMem*1e-6);
	int div = (varMem + memory-1) / memory;
	int paquetMax = (nb_words + div-1) / div;
	int paquet = paquetMax;

	//~ if(d_x->capacity < (size_t)size * paquetMax*nb_gen){
		//~ varMem = d_x->resize((size_t)size * paquetMax*nb_gen, 0, memory);
		//~ printf("div : %d, paquet : %d\n", div, paquetMax);
		//~ printf("Allocating : %.2f Mo, available : %.2f Mo, constMem : %.2f Mo\n", (float)varMem*1e-6, (float)memory*1e-6, (float)constMem*1e-6);
	//~ }
	varMem = d_x->resize((size_t)size * paquetMax*nb_gen, 1, memory);
	
	for(int pass=0; pass<div; pass++){
		if(pass == div-1)
			paquet = nb_words-paquetMax*pass;
		if(div > 1)
			printf("pass %d/%d, paquet : %d\n", pass+1, div, paquet);
		if(paquet > 0){
			int threadPerPerm = min(size, 16384);
			int size_blockx, size_blocky;
			int size_gridx, size_gridy;
			for(int i=0; i<10; i++){
				size_blockx = min(threadPerPerm, 1024);
				size_blocky = max(1024/threadPerPerm, 1);
				size_gridx = (threadPerPerm + size_blockx -1)/size_blockx;
				size_gridy = (paquet*nb_gen + size_blocky-1)/size_blocky;
				if(threadPerPerm < size && size_gridy < 65536 && size_gridx < pow(2,31))
					break;
				threadPerPerm /= 2;
			}
			dim3 blockPerm(size_blockx, size_blocky);
			dim3 gridPerm(size_gridx, size_gridy);
			//~ printf("blockPerm.x : %d, blockPerm.y : %d\n", blockPerm.x, blockPerm.y);
			//~ printf("gridPerm.x : %d, gridPerm.y : %d\n", gridPerm.x, gridPerm.y);
				//~ permute_all_kernel<<<gridPerm, blockPerm, size_blocky*sizeof(int8_t)>>>(d_x->device, d_gen, words->device, size, paquet, size_word, nb_gen);		
				permute_all_kernel<<<gridPerm, blockPerm>>>(d_x->device, d_gen, words->device + pass*paquetMax*size_word, size, paquet, size_word, nb_gen);
			gpuErrchk( cudaDeviceSynchronize() );
			gpuErrchk( cudaPeekAtLastError() );
		
			int gridy;
			int size_block_hash = 32;
			int block_size = 4;
			for(int i=0; i<5; i++){
				if(size_block_hash <= size)
					break;
				size_block_hash /= 2;
			}
			for(int i=0; i<5; i++){
				gridy = (paquet*nb_gen + block_size-1)/block_size;		
				if(gridy > 65536 && block_size*size_block_hash < 1024)
					block_size *= 2;
			}
			dim3 blockHash(size_block_hash, block_size);
			dim3 gridHash(1, (paquet*nb_gen + blockHash.y-1)/blockHash.y);
				hash_kernel<<<gridHash, blockHash>>>(d_x->device, hashed->device + pass*paquetMax*nb_gen, size, paquet*nb_gen);
			
			gpuErrchk( cudaDeviceSynchronize() );
			gpuErrchk( cudaPeekAtLastError() );
			
			hashed->copyDeviceToHost(pass*paquetMax*nb_gen, paquet*nb_gen);
		}
	}
	//~ cudaProfilerStop();
}

bool equal_gpu(const key* key1, const key* key2){
	//~ cudaProfilerStart();
	const int8_t* word1 = &(key1->word[0]);
	const int8_t* word2 = &(key2->word[0]);
	int size = key1->size;
	uint32_t* d_gen = key1->d_gen;
	//~ cudaSetDevice(CUDA_DEVICE);
	int8_t* d_words = key1->d_words;
	gpuErrchk( cudaMemcpy(d_words, word1, NODE * sizeof(int8_t), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_words + NODE, word2, NODE * sizeof(int8_t), cudaMemcpyHostToDevice) );

	dim3 block(128, 1);
	dim3 grid((min(size, 16384) + block.x-1)/block.x, 1);
		equal_kernel<<<grid, block>>>(d_gen, d_words, key1->equal->device, size, NODE, key1->nb_gen);
	gpuErrchk( cudaDeviceSynchronize() );
	gpuErrchk( cudaPeekAtLastError() );
	key1->equal->copyDeviceToHost();

	bool out = (key1->equal->host[0] == size) ? true:false;
	//~ cudaProfilerStop();
	return out;
}

void hash_id_gpu(Vector_cpugpu<uint64_t>* hashed, Vector_gpu<uint32_t>* d_x, const int size){
	//~ cudaSetDevice(CUDA_DEVICE);
	d_x->resize((size_t)size);
	
	dim3 blockInit(32, 4);
	dim3 gridInit(1, ( 1 + blockInit.y-1 )/blockInit.y);
		initId_kernel<<<gridInit, blockInit>>>(d_x->device, size, 1);
	gpuErrchk( cudaDeviceSynchronize() );
	gpuErrchk( cudaPeekAtLastError() );

	dim3 block(32, 4);
	dim3 grid(1, (1 + block.y-1)/block.y);
		hash_kernel<<<grid, block>>>(d_x->device, hashed->device, size, 1);
	gpuErrchk( cudaDeviceSynchronize() );
	gpuErrchk( cudaPeekAtLastError() );
	hashed->copyDeviceToHost();
}

void malloc_gen(uint32_t** __restrict__ d_gen, const uint32_t* __restrict__ gen, const int size, const int8_t nb_gen){
	//~ cudaSetDevice(CUDA_DEVICE);
	gpuErrchk( cudaMalloc((void**)d_gen, size*nb_gen * sizeof(uint32_t)) );
	gpuErrchk( cudaMemcpy(*d_gen, gen, size*nb_gen * sizeof(uint32_t), cudaMemcpyHostToDevice) );	
}

void malloc_words(int8_t** __restrict__ d_words, const int size){
	//~ cudaSetDevice(CUDA_DEVICE);
	gpuErrchk( cudaMalloc((void**)d_words, size*2 * sizeof(uint8_t)) );
}

void free_gen(uint32_t** __restrict__ d_gen){
	gpuErrchk( cudaFree(*d_gen) );
}

void free_words(int8_t** __restrict__ d_words){
	gpuErrchk( cudaFree(*d_words) );
}
#endif  // HPCOMBI_PERM_FONCTIONS_GPU_IMPL_CUH
