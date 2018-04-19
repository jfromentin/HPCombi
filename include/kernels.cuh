#ifndef HPCOMBI_PERM_KERNELS_CUH
#define HPCOMBI_PERM_KERNELS_CUH

#include <stdint.h>
#include <stdio.h>
                              
__global__ void permute_all_kernel(uint32_t* __restrict__ d_x, const uint32_t* __restrict__ d_gen, const int* __restrict__ d_words,
                              const int size, const int nb_words, const int size_word, const int nb_gen);

__global__ void hash_kernel(uint32_t* __restrict__ d_x, uint64_t* d_hashed, const int size, const int nb_vect);
__global__ void initId_kernel(uint32_t * __restrict__ d_x, const int size, const int nb_vect);
__global__ void equal_kernel(uint32_t* __restrict__ d_x, const int* __restrict__ d_word1, const int* __restrict__ d_word2, int* d_equal, const int size, const int size_word, const int nb_gen);


__global__ void permute_all_kernel(uint32_t* __restrict__ d_x, const uint32_t* __restrict__ d_gen, const int* __restrict__ d_words, 
                              const int size, const int nb_words, const int size_word, const int nb_gen){
  const int tidy = blockIdx.y * blockDim.y + threadIdx.y;
  const int lane = threadIdx.x;
  const int coefPerThread = (size+warpSize-1) / warpSize;
  const int offset_d_x =  tidy * size;
  int offset_d_gen;
  int index;
  
  if(tidy/nb_gen<nb_words){
    for(int j=0; j<size_word; j++){
      offset_d_gen =  d_words[j + (tidy/nb_gen) * size_word];
      if(offset_d_gen > -1){
        for(int coef=0; coef<coefPerThread; coef++){
          index = lane + warpSize * coef;
          if (index < size){
            d_x[index + offset_d_x] = d_x[ d_gen[index + offset_d_gen * size] + offset_d_x];
          }
        }
      }
    }
    for(int coef=0; coef<coefPerThread; coef++){
      index = lane + warpSize * coef;
      if (index < size){
        d_x[index + offset_d_x] = d_x[ d_gen[index + (tidy%nb_gen) * size] + offset_d_x];
      }
    }
  }
}



__global__ void equal_kernel(uint32_t* __restrict__ d_x, const uint32_t* __restrict__ d_gen, const int* __restrict__ d_word1, const int* __restrict__ d_word2, 
                              int* d_equal, const int size, const int size_word, const int nb_gen){
  // Global thread id and warp id
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int nb_threads = blockDim.x*gridDim.x;
  const int wid = threadIdx.x / warpSize;
  const int lane = threadIdx.x % warpSize;
  static __shared__ int shared[32];
  const int coefPerThread = (size + nb_threads-1) / nb_threads;

  int offset_d_gen1, offset_d_gen2;
  int index;
  if(tid == 0)
    d_equal[0] = 0;
  
  for(int j=0; j<size_word; j++){
    offset_d_gen1 =  d_word1[j];
    offset_d_gen2 =  d_word2[j];
    if(offset_d_gen1 > nb_gen || offset_d_gen2 > nb_gen)
      printf("offset_d_gen : %d, %d\n", offset_d_gen1, offset_d_gen2 );
      for(int coef=0; coef<coefPerThread; coef++){
        index = tid + nb_threads * coef;    
        if (index < size){
          if(offset_d_gen1 > -1)
            d_x[index] = d_x[ d_gen[index + offset_d_gen1 * size]];
          if(offset_d_gen2 > -1)
            d_x[index + size] = d_x[ d_gen[index + offset_d_gen2 * size] + size];
      }
    }
  }
  
  int equal=0;
  for(int coef=0; coef<coefPerThread; coef++){
    index = tid + nb_threads * coef;
    if (index < size){
      if(d_x[index] == d_x[index + size]){
        equal = 1;
      }      
    }
  }
  
  // Reduction
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
      equal += __shfl_down(equal, offset);

  if(blockDim.x > 32){
    if (lane == 0)
      shared[wid] = equal;
      
    __syncthreads();
    
    if (wid == 0) {
      equal = shared[lane];     
      for (int offset = warpSize/2; offset > 0; offset /= 2) 
          equal += __shfl_down(equal, offset);
    }
  }
  if(threadIdx.x == 0){
    atomicAdd(d_equal, equal);
  }
}

__global__ void hash_kernel(uint32_t* __restrict__ d_x, uint64_t* d_hashed, const int size, const int nb_vect){
  // kernel with less operation. Each threads compute a part of the polynome with Horner method.
  // A warp computes a hash.

  const int tidy = blockIdx.y * blockDim.y + threadIdx.y;
  const int lane = threadIdx.x;
  const int coefPerThread = (size+warpSize-1) / warpSize;
  const uint64_t prime = 0x9e3779b97f4a7bb9;
  uint64_t out=1;
  uint64_t coef=0;
    
  for (int j=0; j<coefPerThread*lane; j++)
    out *= prime;
  if(lane + warpSize * 0 < size && tidy < nb_vect){
    out *= d_x[tidy * size + lane + warpSize * 0];
  }
  else
    out = 0;

  for(int i=1; i<coefPerThread; i++){
    if(lane + warpSize * i < size && tidy < nb_vect)
      coef = d_x[tidy * size + lane + warpSize * i];

    out += coef;
    out *= prime;

    coef = 0;
    }

  // Reduction
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
      out += __shfl_down(out, offset);

  if(lane == 0)
    d_hashed[tidy] = out >> 32;
}


__global__ void initId_kernel(uint32_t * __restrict__ d_x, const int size, const int nb_vect){
  const int lane = threadIdx.x;
	const int coefPerThread = (size + warpSize-1) / warpSize;
  const int tidy = blockIdx.y * blockDim.y + threadIdx.y;
  int index;
  for (int j=0; j<coefPerThread; j++){
    index = lane + warpSize*j;
    if(index < size && tidy < nb_vect)
      d_x[index + tidy*size] = index;    
  }
}

#endif  // HPCOMBI_PERM_KERNELS_CUH
