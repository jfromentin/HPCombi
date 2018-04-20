#ifndef HPCOMBI_PERM_FONCTIONS_GPU_IMPL_CUH
#define HPCOMBI_PERM_FONCTIONS_GPU_IMPL_CUH

#include <stdint.h>
#include <stdio.h>
#include <iostream>
#include <type_traits>
#include <typeinfo>
#include <cuda_profiler_api.h>

#include "fonctions_gpu.cuh"

void cudaSetDevice_cpu(){ cudaSetDevice(CUDA_DEVICE); }

void hpcombi_gpu(Vector_cpugpu<int>* words, Vector_gpu<uint32_t>* d_x, Vector_gpu<uint32_t>* d_y, const uint32_t* __restrict__ d_gen, Vector_cpugpu<uint64_t>* hashed, 
				const int size, const int size_word, const int nb_gen){
	//~ cudaProfilerStart();
	//~ cudaSetDevice(CUDA_DEVICE);
	//~ float timer;
	int nb_words = words->size/size_word;

	d_x->resize(size * nb_words*nb_gen);
	d_y->resize(size * nb_words*nb_gen);

	dim3 blockInit(32, 4);
	dim3 gridInit(1, ( nb_words*nb_gen + blockInit.y-1 )/blockInit.y);
		initId_kernel<<<gridInit, blockInit>>>(d_x->device, size, nb_words*nb_gen);
	gpuErrchk( cudaDeviceSynchronize() );
	gpuErrchk( cudaPeekAtLastError() );
	words->copyHostToDevice();

	//~ cudaEvent_t start, stop;
	//~ cudaEventCreate(&start);
	//~ cudaEventCreate(&stop);
	int threadPerPerm = 1024;
	int size_block;
	int size_grid;
	for(int i=0; i<10; i++){
		size_block = 1024/threadPerPerm;
		size_grid = (nb_words*nb_gen + size_block-1)/size_block;
		if(threadPerPerm <= size && size_grid <= 65535)
			break;
		threadPerPerm /= 2;
	}
	//~ printf("threadPerPerm : %d, size_block : %d, size_grid : %d\n", threadPerPerm, size_block, size_grid);
	dim3 blockPerm(threadPerPerm, size_block);
	dim3 gridPerm(1, size_grid);
	//~ cudaEventRecord(start);		
		permute_all_kernel<<<gridPerm, blockPerm>>>(d_x->device, d_y->device, d_gen, words->device, size, nb_words, size_word, nb_gen);		
	//~ cudaEventRecord(stop);
	//~ cudaEventSynchronize(stop);
	//~ cudaEventElapsedTime(&timer, start, stop);
		
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
		gridy = (nb_words*nb_gen + block_size-1)/block_size;		
		if(gridy > 65535 && block_size*size_block_hash < 1024)
			block_size *= 2;
	}
	dim3 blockHash(size_block_hash, block_size);
	dim3 gridHash(1, (nb_words*nb_gen + blockHash.y-1)/blockHash.y);
	//~ cudaEventRecord(start);
		hash_kernel<<<gridHash, blockHash>>>(d_x->device, hashed->device, size, nb_words*nb_gen);		
	//~ cudaEventRecord(stop);
	//~ cudaEventSynchronize(stop);
	//~ cudaEventElapsedTime(&timer, start, stop);
	
	gpuErrchk( cudaDeviceSynchronize() );
	gpuErrchk( cudaPeekAtLastError() );
	hashed->copyDeviceToHost();
	//~ cudaProfilerStop();
}

bool equal_gpu(const key* key1, const key* key2, int block_size, const int size, const int size_word, const int nb_gen){
	const int* word1 = &(key1->word[0]);
	const int* word2 = &(key2->word[0]);
	//~ cudaPointerAttributes attr;
    //~ cudaPointerGetAttributes( &attr, key1->word.device );
    //~ print_ptr_attr( attr );
    //~ cudaPointerGetAttributes( &attr, key1->word.host );
    //~ print_ptr_attr( attr );
	uint32_t* d_gen = key1->d_gen;
	
	//~ cudaSetDevice(CUDA_DEVICE);
	uint32_t* d_x;
	int* d_word1;
	int* d_word2;
	int equal;
	int* d_equal;
	gpuErrchk( cudaMalloc((void**)&d_word1, size_word * sizeof(int)) ); // TODO do not reallocate
	gpuErrchk( cudaMalloc((void**)&d_word2, size_word * sizeof(int)) ); // TODO do not reallocate
	gpuErrchk( cudaMalloc((void**)&d_x, size * 2 * sizeof(uint32_t)) ); // TODO do not reallocate
	gpuErrchk( cudaMalloc((void**)&d_equal, sizeof(int)) ); // TODO do not reallocate

	dim3 blockInit(32, 4);
	dim3 gridInit(1, ( nb_gen*2 + blockInit.y-1 )/blockInit.y);
		initId_kernel<<<gridInit, blockInit>>>(d_x, size, nb_gen*2);
	gpuErrchk( cudaDeviceSynchronize() );
	gpuErrchk( cudaPeekAtLastError() );

	//~ key1->word.copyHostToDevice();
	//~ key2->word.copyHostToDevice();
	gpuErrchk( cudaMemcpy(d_word1, word1, size_word * sizeof(int), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_word2, word2, size_word * sizeof(int), cudaMemcpyHostToDevice) );

	dim3 block(128, 1);
	dim3 grid((size + block.x-1)/block.x, 1);
		//~ equal_kernel<<<grid, block>>>(d_x, d_gen, key1->word.device, key2->word.device, d_equal, size, size_word, nb_gen);
		equal_kernel<<<grid, block>>>(d_x, d_gen, d_word1, d_word2, d_equal, size, size_word, nb_gen);
	gpuErrchk( cudaDeviceSynchronize() );
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaMemcpy(&equal, d_equal, sizeof(int), cudaMemcpyDeviceToHost) );

	gpuErrchk( cudaFree(d_x) );
	gpuErrchk( cudaFree(d_word1) );
	gpuErrchk( cudaFree(d_word2) );
	gpuErrchk( cudaFree(d_equal) );
	bool out = (equal == size) ? true:false;
	
	return out;
}

void hash_id_gpu(Vector_cpugpu<uint64_t>* hashed, Vector_gpu<uint32_t>* d_x, int block_size, const int size){
	//~ cudaSetDevice(CUDA_DEVICE);
	d_x->resize(size);
	
	dim3 blockInit(32, 4);
	dim3 gridInit(1, ( 1 + blockInit.y-1 )/blockInit.y);
		initId_kernel<<<gridInit, blockInit>>>(d_x->device, size, 1);
	gpuErrchk( cudaDeviceSynchronize() );
	gpuErrchk( cudaPeekAtLastError() );

	dim3 block(32, block_size);
	dim3 grid(1, (1 + block.y-1)/block.y);
		hash_kernel<<<grid, block>>>(d_x->device, hashed->device, size, 1);
	gpuErrchk( cudaDeviceSynchronize() );
	gpuErrchk( cudaPeekAtLastError() );
	hashed->copyDeviceToHost();
}

void malloc_gen(uint32_t** __restrict__ d_gen, const uint32_t* __restrict__ gen, const int size, const int nb_gen){
	//~ cudaSetDevice(CUDA_DEVICE);
	gpuErrchk( cudaMalloc((void**)d_gen, size*nb_gen * sizeof(uint32_t)) );
	gpuErrchk( cudaMemcpy(*d_gen, gen, size*nb_gen * sizeof(uint32_t), cudaMemcpyHostToDevice) );	
}

void free_gen(uint32_t** __restrict__ d_gen){
	gpuErrchk( cudaFree(*d_gen) );
}
#endif  // HPCOMBI_PERM_FONCTIONS_GPU_IMPL_CUH
