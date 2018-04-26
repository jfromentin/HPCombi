#ifndef HPCOMBI_PERM_KERNELS_CUH
#define HPCOMBI_PERM_KERNELS_CUH

#include <stdint.h>
#include <stdio.h>
                              
__global__ void permute_all_kernel(uint32_t* __restrict__ d_x, const uint32_t* __restrict__ d_gen, const int8_t* __restrict__ d_words,
                              const int size, const int nb_words, const int size_word, const int8_t nb_gen);

__global__ void hash_kernel(uint32_t* __restrict__ d_x, uint64_t* d_hashed, const int size, const int nb_vect);
__global__ void initId_kernel(uint32_t * __restrict__ d_x, const int size, const int nb_vect);
__global__ void equal_kernel(uint32_t* __restrict__ d_x, const int8_t* __restrict__ d_word1, const int8_t* __restrict__ d_word2, int* d_equal, const int size, const int size_word, const int8_t nb_gen);


__global__ void permute_all_kernel(uint32_t* __restrict__ d_x, uint32_t* __restrict__ d_y, const uint32_t* __restrict__ d_gen, const int8_t* __restrict__ d_words, 
                              const int size, const int nb_words, const int size_word, const int8_t nb_gen){
  const int tidy = blockIdx.y*blockDim.y + threadIdx.y;
  const int wordId = tidy/nb_gen;
  const int coefPerThread = (size + blockDim.x-1) / blockDim.x;
  const int offset_d_x =  tidy * size;
  int offset_d_gen;
  int index;
  if(wordId < nb_words){
    for(int j=0; j<size_word; j++){
      offset_d_gen =  d_words[j + wordId * size_word];
      if(offset_d_gen > -1){
        
        for(int coef=0; coef<coefPerThread; coef++){
          index = threadIdx.x + blockDim.x*coef;
          if (index < size){
            d_y[index + offset_d_x] = d_x[ d_gen[index + offset_d_gen*size] + offset_d_x];
          }
        }
        __syncthreads();
        for(int coef=0; coef<coefPerThread; coef++){
          index = threadIdx.x + blockDim.x*coef;
          if (index < size){
            d_x[index + offset_d_x] = d_y[index + offset_d_x];
          }
        }
        
      }
    }
    
    for(int coef=0; coef<coefPerThread; coef++){
      index = threadIdx.x + blockDim.x * coef;
      if (index < size){
        d_y[index + offset_d_x] = d_x[d_gen[index + (tidy%nb_gen)*size] + offset_d_x];
      }
    }
    __syncthreads();
    for(int coef=0; coef<coefPerThread; coef++){
      index = threadIdx.x + blockDim.x * coef;
      if (index < size){
        d_x[index + offset_d_x] = d_y[ index + offset_d_x];
      }
    }
    
  }
}



__global__ void equal_kernel(uint32_t* __restrict__ d_x, const uint32_t* __restrict__ d_gen, const int8_t* __restrict__ d_word1, const int8_t* __restrict__ d_word2, 
                              int* d_equal, const int size, const int size_word, const int8_t nb_gen){
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


#define NB_HASH_FUNC 4
__global__ void hash_kernel(uint32_t* __restrict__ d_x, uint64_t* d_hashed, const int size, const int nb_vect){
  // kernel with less operation. Each threads compute a part of the polynome with Horner method.
  // A warp computes a hash.

  const int tidy = blockIdx.y * blockDim.y + threadIdx.y;
  const int coefPerThread = (size+blockDim.x-1) / blockDim.x;
  uint64_t primes[NB_HASH_FUNC] = {13, 17, 19, 23};
  //~ primes[0] = 13;
  //~ primes[1] = 17;
  //~ primes[2] = 19;
  //~ primes[3] = 23;
  uint64_t out[NB_HASH_FUNC];
  for(int k=0; k<NB_HASH_FUNC; k++)
    out[k] = 1;
  uint64_t coef=0;
    
  for(int k=0; k<NB_HASH_FUNC; k++)
    for (int j=0; j<coefPerThread*threadIdx.x; j++)
      out[k] *= primes[k];

  if(threadIdx.x + blockDim.x * 0 < size && tidy < nb_vect){
    coef = d_x[tidy*size + threadIdx.x + blockDim.x*0];
    for(int k=0; k<NB_HASH_FUNC; k++)
      out[k] *= coef;
  }
  else
    for(int k=0; k<NB_HASH_FUNC; k++)
      out[k] = 0;

  coef=0;
  for(int i=1; i<coefPerThread; i++){
    if(threadIdx.x + blockDim.x * i < size && tidy < nb_vect)
      coef = d_x[tidy*size + threadIdx.x + blockDim.x*i];

    for(int k=0; k<NB_HASH_FUNC; k++){
      out[k] += coef;
      out[k] *= primes[k];
    }

    coef = 0;
  }

  // Reduction
  for(int k=0; k<NB_HASH_FUNC; k++)
    for (int offset = blockDim.x/2; offset > 0; offset /= 2) 
        out[k] += __shfl_down(out[k], offset);

  if(threadIdx.x == 0)
    for(int k=0; k<NB_HASH_FUNC; k++)
      d_hashed[NB_HASH_FUNC*tidy + k] = out[k] >> 32;
}

//~ __global__ void hash_kernel(uint32_t* __restrict__ d_x, uint64_t* d_hashed, const int size, const int nb_vect){
  //~ // kernel with less operation. Each threads compute a part of the polynome with Horner method.
  //~ // A warp computes a hash.

  //~ const int tidy = blockIdx.y * blockDim.y + threadIdx.y;
  //~ const int coefPerThread = (size+blockDim.x-1) / blockDim.x;
  //~ const uint64_t prime = 0x9e3779b97f4a7bb9;
  //~ uint64_t out=1;
  //~ uint64_t coef=0;
    
  //~ for (int j=0; j<coefPerThread*threadIdx.x; j++)
    //~ out *= prime;
  //~ if(threadIdx.x + blockDim.x * 0 < size && tidy < nb_vect){
    //~ out *= d_x[tidy*size + threadIdx.x + blockDim.x*0];
  //~ }
  //~ else
    //~ out = 0;

  //~ for(int i=1; i<coefPerThread; i++){
    //~ if(threadIdx.x + blockDim.x * i < size && tidy < nb_vect)
      //~ coef = d_x[tidy*size + threadIdx.x + blockDim.x*i];

    //~ out += coef;
    //~ out *= prime;

    //~ coef = 0;
  //~ }

  //~ // Reduction
  //~ for (int offset = blockDim.x/2; offset > 0; offset /= 2) 
      //~ out += __shfl_down(out, offset);

  //~ if(threadIdx.x == 0)
    //~ d_hashed[tidy] = out >> 32;
//~ }


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


#define get_key(entry)            ((unsigned)((entry) >> 32))
#define get_value(entry)          ((unsigned)((entry) & 0xffffffff))
#define make_entry(key, value)    ((((unsigned long long)key) << 32) + (value))
#define KEY_EMPTY                 0xffffffff
#define NOT_FOUND                 0xffffffff
#define PRIME                     ((unsigned)334214459)
#define TABLE_SIZE                ((unsigned)1e5)
#define MAX_ITER                  ((unsigned)20)

__device__ unsigned hash_function_1(unsigned key){ return (5*key+11)%PRIME; }
__device__ unsigned hash_function_2(unsigned key){ return (7*key+13)%PRIME; }
__device__ unsigned hash_function_3(unsigned key){ return (11*key+19)%PRIME; }
__device__ unsigned hash_function_4(unsigned key){ return (13*key+23)%PRIME; }

__device__ bool insert(uint32_t * table, uint32_t key, uint64_t* hashed){
  
  uint64_t location = hashed[0];
  for(int i=0; i<MAX_ITER; i++){
    key = atomicExch(&table[location], key);
    if(key == KEY_EMPTY)
      return true;
      
    if(location == hashed[0])
      location = hashed[1];
    else if(location == hashed[1])
      location = hashed[2];
    else if(location == hashed[2])
      location = hashed[3];
    else
      location = hashed[0];
  }  
    
  return false;
}

__device__ bool insert_all(unsigned long long* table, unsigned* keys, unsigned* values){
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned key = keys[tid];
  unsigned value = values[tid];
  unsigned long long entry = make_entry(key, value);
  
  unsigned location = hash_function_1(key);
  
  for(int i=0; i<MAX_ITER; i++){
    entry = atomicExch(&table[location], entry);
    key = get_key(entry);
    if(key == KEY_EMPTY)
      return true;
      
    unsigned location_1 = hash_function_1(key)%TABLE_SIZE;
    unsigned location_2 = hash_function_2(key)%TABLE_SIZE;
    unsigned location_3 = hash_function_3(key)%TABLE_SIZE;
    unsigned location_4 = hash_function_4(key)%TABLE_SIZE;
      
    if(location == location_1)
      location = location_2;
    else if(location == location_2)
      location = location_3;
    else if(location == location_3)
      location = location_4;
    else
      location = location_1;
  }  
    
  return false;
}


__device__ unsigned retriev(unsigned key, unsigned long long* table){
  
  unsigned location_1 = hash_function_1(key)%TABLE_SIZE;
  unsigned location_2 = hash_function_2(key)%TABLE_SIZE;
  unsigned location_3 = hash_function_3(key)%TABLE_SIZE;
  unsigned location_4 = hash_function_4(key)%TABLE_SIZE;
  
  unsigned long long entry;
  
  if(get_key(table[location_1]) != key)
    if(get_key(table[location_2]) != key)
      if(get_key(table[location_3]) != key)
        if(get_key(table[location_4]) != key)
          entry = make_entry(0, NOT_FOUND);
  return get_value(entry);
  
}
#endif  // HPCOMBI_PERM_KERNELS_CUH
