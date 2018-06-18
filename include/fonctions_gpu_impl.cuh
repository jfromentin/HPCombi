#ifndef HPCOMBI_PERM_FONCTIONS_GPU_IMPL_CUH
#define HPCOMBI_PERM_FONCTIONS_GPU_IMPL_CUH

#include <stdint.h>
#include <stdio.h>
#include <iostream>
#include <type_traits>
#include <typeinfo>
#include <cuda_profiler_api.h>
#include <chrono>
using namespace std::chrono;

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
	printf("GPU count : %d\n", count);
	cudaSetDevice(count-1);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, count-1);
	printf("%s :\n", prop.name); 
	printf("	Global memory available : %.2f Go\n", static_cast<float>(prop.totalGlobalMem)*1e-9); 
	printf("	Shared memory per block : %.2f Ko\n", static_cast<float>(prop.sharedMemPerBlock)*1e-3);
	printf("	Registers per block : %d\n", prop.regsPerBlock);
	printf("	Clock Rate : %.2f GHz\n", static_cast<float>(prop.clockRate)*1e-6);
	printf("	MultiProcessor Count : %d\n", prop.multiProcessorCount);
	printf("	Max Threads Per Block : %d\n", prop.maxThreadsPerBlock);
	printf("	Max Threads Dim : %d x %d x %d\n", 
				prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
	printf("	Max Grid Size : %d x %d x %d\n", 
				prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
	printf("\n");
	return prop.totalGlobalMem;
}


template <typename T>
void compHash_gpu(Vector_cpugpu<int8_t>& words, Vector_gpu<T>& workSpace, 
				const T* __restrict__ d_gen, Vector_cpugpu<uint64_t>& hashed, 
				const int size, const int size_word, const int8_t nb_gen, size_t memory){
	//~ cudaProfilerStart();
	//~ float time;
	//~ cudaEvent_t start, stop;
	//~ cudaEventCreate(&start);
	//~ cudaEventCreate(&stop);
  auto tstartCpu = high_resolution_clock::now();
	memory /= 1.05;
	int nb_words = words.size()/size_word;
	
	// Divide the size of words to ensure enough memory space is available on the GPU.
	size_t constMem = words.capacity()*sizeof(int8_t) 
					+ sizeof(T)*nb_gen*size_word 
					+ hashed.resize(static_cast<size_t>(nb_words)*nb_gen * NB_HASH_FUNC, 0);
	memory -= constMem;
	size_t varMem = workSpace.resize(static_cast<size_t>(size) * nb_words*nb_gen, 0, memory);
	int div = (varMem + memory-1) / memory;
	int paquetMax = (nb_words + div-1) / div;
	int paquet = paquetMax;

	// Resizing array acoordingly to avaible memory in GPU and needs.
	workSpace.resize(static_cast<size_t>(size) * paquetMax*nb_gen, 1, memory);
    hashed.resize(static_cast<size_t>(nb_words)*nb_gen * NB_HASH_FUNC, 2);
    
	words.copyHostToDevice();
	
	//~ cudaEventRecord(start);
	if(div>1)
		std::cout << div << " passage(s)" << std::endl;
	for(int pass=0; pass<div; pass++){
		if(pass == div-1)
			paquet = nb_words-paquetMax*pass;
		if(paquet > 0){
			// Grid an dblock configuration must adapt to the number of word to compute 
			// and to the size of the transformations.
			// If the number of word to compute is low, transformations are spread through lots of threads.
			// Where as if number of word to compute is high, transformations are spread trhough few threads.
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
				compose_kernel<T><<<gridPerm, blockPerm>>>(workSpace.device(), d_gen, words.device() + pass*paquetMax*size_word, size, paquet, size_word, nb_gen);
			gpuErrchk( cudaDeviceSynchronize() );
			gpuErrchk( cudaPeekAtLastError() );

			// blockHash.y must be 1, if not reduction through tidy thread is required in kernel.
			dim3 blockHash(64, 1), gridHash(1, 1);
			for(int i=0; i<6; i++){
				if(blockHash.y <= size)
					break;
				blockHash.y /= 2;
			}
			for(int i=0; i<6; i++){
				gridHash.x = (paquet*nb_gen + blockHash.x-1)/blockHash.x;		
				if(gridHash.x > pow(2,31) && blockHash.x*blockHash.y < 1024)
					blockHash.x *= 2;
			}
				hash_kernel<T><<<gridHash, blockHash>>>(workSpace.device(), hashed.device() + pass*paquetMax*nb_gen * NB_HASH_FUNC, size, paquet*nb_gen);
				

				//~ dim3 blockPre(64, 1), gridPre((paquet*nb_gen + blockPre.x-1)/blockPre.x, 1);
				pre_insert_kernel<T><<<gridHash, blockHash>>>(workSpace.device(), hashed.device() + pass*paquetMax*nb_gen * NB_HASH_FUNC, size, paquet*nb_gen);
			
			gpuErrchk( cudaDeviceSynchronize() );
			gpuErrchk( cudaPeekAtLastError() );
			
			hashed.copyDeviceToHost(pass*paquetMax*nb_gen * NB_HASH_FUNC, paquet*nb_gen * NB_HASH_FUNC);
		}
	}
  auto tfinCpu = high_resolution_clock::now();
  auto tmCpu = duration_cast<duration<double>>(tfinCpu - tstartCpu);
  timeCH += tmCpu.count();
	//~ cudaEventRecord(stop);
	//~ cudaEventSynchronize(stop);
	//~ cudaEventElapsedTime(&time, start, stop);
	//~ time /= 1e3;
	//~ std::cout << "Time in hpcombi : " << (int)time/3600 << ":" << (int)time%3600/60 << ":" << ((int)time%3600)%60 << std::endl;
	//~ cudaProfilerStop();
}


template <typename T>
bool equal_gpu(const Key& key1, const Key& key2, T* d_gen, int8_t* d_words,
				Vector_cpugpu<int>& equal,	const int size, const int8_t nb_gen){
	// key1.data() are in paged memory.
	// Time could be saved if they where allocated in pinned memory.
	auto tstartCpu = high_resolution_clock::now();
	gpuErrchk( cudaMemcpy(d_words, key1.data(), NODE * sizeof(int8_t), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_words + NODE, key2.data(), NODE * sizeof(int8_t), cudaMemcpyHostToDevice) );

	const dim3 block(128, 1);
	const dim3 grid((min(size, 16384) + block.x-1)/block.x, 1);
		equal_kernel<T><<<grid, block>>>(d_gen, d_words, equal.device(), size, NODE, nb_gen);
	gpuErrchk( cudaDeviceSynchronize() );
	gpuErrchk( cudaPeekAtLastError() );
	equal.copyDeviceToHost();
	const bool out = (equal[0] == size) ? true:false;

  auto tfinCpu = high_resolution_clock::now();
  auto tmCpu = duration_cast<duration<double>>(tfinCpu - tstartCpu);
  timeEq += tmCpu.count();
	return out;
}


//~ bool equal_cpu(const Key& key1, const Key& key2, uint64_t* gen, const int size){
  //~ auto tstartCpu = high_resolution_clock::now();
  
  //~ bool result = true;
  //~ uint64_t* tmp1 = (uint64_t*)malloc(size * sizeof(uint64_t));
  //~ uint64_t* tmp2 = (uint64_t*)malloc(size * sizeof(uint64_t));
  //~ for(int i=0; i<size; i++){
    //~ tmp1[i] = i;
    //~ tmp2[i] = i;
    //~ for(int j=0; j<NODE; j++){
      //~ if(key1[j]>-1)
        //~ tmp1[i] = gen[tmp1[i] + static_cast<uint64_t>(key1[j])*size];
      //~ if(key2[j]>-1)
        //~ tmp2[i] = gen[tmp2[i] + static_cast<uint64_t>(key2[j])*size];
    //~ }
    //~ if(tmp1[i] != tmp2[i]){
      //~ result = false;
      //~ break;
    //~ }
  //~ }

  //~ auto tfinCpu = high_resolution_clock::now();
  //~ auto tmCpu = duration_cast<duration<double>>(tfinCpu - tstartCpu);
  //~ timeEq += tmCpu.count();
  //~ return result;
//~ }


bool equal_cpu(const Key& key1, const Key& key2, uint64_t* gen, const int size){
  auto tstartCpu = high_resolution_clock::now();
  
  bool result = true;
  uint64_t* tmp1 = (uint64_t*)malloc(size * sizeof(uint64_t));
  uint64_t* tmp2 = (uint64_t*)malloc(size * sizeof(uint64_t));
  uint64_t* tmp3 = (uint64_t*)malloc(size * sizeof(uint64_t));
  uint64_t* tmp4 = (uint64_t*)malloc(size * sizeof(uint64_t));
  for(int i=0; i<size; i++){
    tmp1[i] = i;
    tmp2[i] = i;
	}
    for(int j=0; j<NODE; j++){
        if(key1[j]>-1){
          for(int i=0; i<size; i++)
            tmp3[i] = gen[tmp1[i] + static_cast<uint64_t>(key1[j])*size];        
          for(int i=0; i<size; i++)
            tmp1[i] = tmp3[i];
        }
        if(key2[j]>-1){
          for(int i=0; i<size; i++)
            tmp4[i] = gen[tmp2[i] + static_cast<uint64_t>(key2[j])*size];
          for(int i=0; i<size; i++)
            tmp2[i] = tmp4[i];
        }
    }
    
    for(int i=0; i<size; i++){
      if(tmp1[i] != tmp2[i]){
        result = false;
        break;
      }
    }

  auto tfinCpu = high_resolution_clock::now();
  auto tmCpu = duration_cast<duration<double>>(tfinCpu - tstartCpu);
  timeEq += tmCpu.count();
  free(tmp1);
  free(tmp2);
  free(tmp3);
  free(tmp4);
  return result;
}

template <typename T>
void hash_id_gpu(Vector_cpugpu<uint64_t>& hashed, Vector_gpu<T>& workSpace, const int size){
	workSpace.resize(static_cast<size_t>(size));
		
	const dim3 blockInit(32, 4);
	const dim3 gridInit(1, ( 1 + blockInit.y-1 )/blockInit.y);
		initId_kernel<T><<<gridInit, blockInit>>>(workSpace.device(), size, 1);
	gpuErrchk( cudaDeviceSynchronize() );
	gpuErrchk( cudaPeekAtLastError() );

	const dim3 block(32, 4);
	const dim3 grid(1, (1 + block.y-1)/block.y);
		hash_kernel<T><<<grid, block>>>(workSpace.device(), hashed.device(), size, 1);
	gpuErrchk( cudaDeviceSynchronize() );
	gpuErrchk( cudaPeekAtLastError() );
	
	hashed.copyDeviceToHost();
}

template <typename T>
void malloc_gen(T*& __restrict__ d_gen, const uint64_t* __restrict__ gen, 
				const int size, const int8_t nb_gen){
	T* tmp;
	gpuErrchk( cudaMallocHost((void**)&tmp, size*nb_gen * sizeof(T)) );
	for(int i=0; i<size*nb_gen; i++)
		tmp[i] = gen[i];
	gpuErrchk( cudaMalloc((void**)&d_gen, size*nb_gen * sizeof(T)) );
	gpuErrchk( cudaMemcpy(d_gen, tmp, size*nb_gen * sizeof(T), cudaMemcpyHostToDevice) );	
}

void malloc_words(int8_t*& __restrict__ d_words, const int size){
	gpuErrchk( cudaMalloc((void**)&d_words, size*2 * sizeof(uint8_t)) );
}

template <typename T>
void free_gen(T*& __restrict__ d_gen){
	gpuErrchk( cudaFree(d_gen) );
}

void free_words(int8_t*& __restrict__ d_words){
	gpuErrchk( cudaFree(d_words) );
}



// Explicit instantiation uint64_t
template void compHash_gpu<uint64_t>(Vector_cpugpu<int8_t>& words, Vector_gpu<uint64_t>& workSpace, 
					const uint64_t* __restrict__ d_gen, Vector_cpugpu<uint64_t>& hashed, 
					const int size, const int size_word, const int8_t nb_gen, size_t memory);
template void hash_id_gpu<uint64_t>(Vector_cpugpu<uint64_t>& hashed, Vector_gpu<uint64_t>& workSpace, const int size);
template bool equal_gpu<uint64_t>(const Key& key1, const Key& key2, uint64_t* d_gen, int8_t* d_words,
					Vector_cpugpu<int>& equal, const int size, const int8_t nb_gen);
template void malloc_gen<uint64_t>(uint64_t*& __restrict__ d_gen, const uint64_t* __restrict__ gen, 
					const int size, const int8_t nb_gen);
template void free_gen<uint64_t>(uint64_t*& __restrict__ d_gen);
// Explicit instantiation uint32_t
template void compHash_gpu<uint32_t>(Vector_cpugpu<int8_t>& words, Vector_gpu<uint32_t>& workSpace, 
					const uint32_t* __restrict__ d_gen, Vector_cpugpu<uint64_t>& hashed, 
					const int size, const int size_word, const int8_t nb_gen, size_t memory);
template void hash_id_gpu<uint32_t>(Vector_cpugpu<uint64_t>& hashed, Vector_gpu<uint32_t>& workSpace, const int size);
template bool equal_gpu<uint32_t>(const Key& key1, const Key& key2, uint32_t* d_gen, int8_t* d_words,
					Vector_cpugpu<int>& equal, const int size, const int8_t nb_gen);
template void malloc_gen<uint32_t>(uint32_t*& __restrict__ d_gen, const uint64_t* __restrict__ gen, 
					const int size, const int8_t nb_gen);
template void free_gen<uint32_t>(uint32_t*& __restrict__ d_gen);
// Explicit instantiation uint16_t
template void compHash_gpu<uint16_t>(Vector_cpugpu<int8_t>& words, Vector_gpu<uint16_t>& workSpace, 
					const uint16_t* __restrict__ d_gen, Vector_cpugpu<uint64_t>& hashed, 
					const int size, const int size_word, const int8_t nb_gen, size_t memory);
template void hash_id_gpu<uint16_t>(Vector_cpugpu<uint64_t>& hashed, Vector_gpu<uint16_t>& workSpace, const int size);
template bool equal_gpu<uint16_t>(const Key& key1, const Key& key2, uint16_t* d_gen, int8_t* d_words,
					Vector_cpugpu<int>& equal, const int size, const int8_t nb_gen);
template void malloc_gen<uint16_t>(uint16_t*& __restrict__ d_gen, const uint64_t* __restrict__ gen, 
					const int size, const int8_t nb_gen);
template void free_gen<uint16_t>(uint16_t*& __restrict__ d_gen);
// Explicit instantiation uint8_t
template void compHash_gpu<uint8_t>(Vector_cpugpu<int8_t>& words, Vector_gpu<uint8_t>& workSpace, 
					const uint8_t* __restrict__ d_gen, Vector_cpugpu<uint64_t>& hashed, 
					const int size, const int size_word, const int8_t nb_gen, size_t memory);
template void hash_id_gpu<uint8_t>(Vector_cpugpu<uint64_t>& hashed, Vector_gpu<uint8_t>& workSpace, const int size);
template bool equal_gpu<uint8_t>(const Key& key1, const Key& key2, uint8_t* d_gen, int8_t* d_words,
					Vector_cpugpu<int>& equal, const int size, const int8_t nb_gen);
template void malloc_gen<uint8_t>(uint8_t*& __restrict__ d_gen, const uint64_t* __restrict__ gen, 
					const int size, const int8_t nb_gen);
template void free_gen<uint8_t>(uint8_t*& __restrict__ d_gen);

#endif  // HPCOMBI_PERM_FONCTIONS_GPU_IMPL_CUH
