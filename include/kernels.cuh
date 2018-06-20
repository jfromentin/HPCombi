#ifndef HPCOMBI_PERM_KERNELS_CUH
#define HPCOMBI_PERM_KERNELS_CUH

#include <stdint.h>
#include <stdio.h>
#include "RennerGpu.hpp"


/** @brief Compose transformations.
* @details 1st stage) words array contains nb_words suites of generators (named by there indexes) 
* 		that are applyed to the identity transformation.
* 		2d stage) For each resulting transformation one more generator is applyed 
* 		to compute next generation set of transformation.
* 		Hence nb_gen transformation are compute for each result of the first stage.
* 		for a total of nb_words*nb_gen transformations. 
* @param workSpace Array allocated on GPU of size size*nb_words*nb_gen containing the transformations.
* @param d_gen Array allocaded on GPU of size size*nb_gen containing the generators.
* @param d_words Array allocaded on GPU of size size_word*nb_words containing the words.
* @param size Size of one transformations.
* @param nb_words Number of words.
* @param size_word Size of one word.
* @param nb_gen Number of generators.
*
*/      
template <typename T>
__global__ void compose_kernel(T* __restrict__ workSpace, const T* __restrict__ d_gen, const int8_t* __restrict__ d_words, 
                              const int size, const int nb_words, const int size_word, const int8_t nb_gen);
                              
/** @brief Compute the hash values of transformations.
* @details Let be the polynome P : sum e_i * X^i for i in [1, size], 
* 		where e_i are the transformation coeficients.
* 		The hash value of a tranformation is the value of the polynome P for X= aPrimeNumber.
* 		It is computed with the Horner method.
* @param workSpace Array allocated on GPU of size size*nb_trans containing the transformations.
* @param d_hashed Array allocaded on GPU of size nb_trans containing the hash value.
* @param size Size of one transformations.
* @param nb_trans Number of transformations to hash.
*
*/   
template <typename T>
__global__ void hash_kernel(T* __restrict__ workSpace, uint64_t* d_hashed, const int size, const int nb_trans);
                           
/** @brief Initialize workSpace to identity.
* @param workSpace Array allocated on GPU of size size*nb_trans containing the transformations.
* @param size Size of one transformations.
* @param nb_trans Number of transformations to hash.
*
*/   
template <typename T>
__global__ void initId_kernel(T * __restrict__ workSpace, const int size, const int nb_trans);
                           
/** @brief Check equality of the resulting transformation of two words.
* 		1st stage) d_words array contains two suites of generators (named by there indexes) 
* 		that are applyed to the identity transformation.
* 		2d stage) The resulting transformation are compared element by element
* and the number of equal coeficient is stored in d_equal.
* @param d_gen Array allocaded on GPU of size size*nb_gen containing the generators.
* @param d_words Array allocaded on GPU of size size_word*nb_words containing the words.
* @param d_equal Number of equal coeficients.
* 		Must be equal to size for the transformation to be equals.
* @param size Size of one transformations.
* @param size_word Size of one word.
* @param nb_gen Number of generators.
*/ 
template <typename T>
__global__ void equal_kernel(const T* __restrict__ d_gen, const int8_t* __restrict__ d_words, 
                              int* d_equal, const int size, const int size_word, const int8_t nb_gen);


template <typename T>
__global__ void compose_kernel(T* __restrict__ workSpace, const T* __restrict__ d_gen, const int8_t* __restrict__ d_words, 
                              const int size, const int nb_words, const int size_word, const int8_t nb_gen){
  const int tidy = blockIdx.y*blockDim.y + threadIdx.y;
  const int tidx = blockIdx.x*blockDim.x + threadIdx.x;
  const int wordId = tidy/nb_gen;
  const int nb_threads = blockDim.x*gridDim.x;
  const int coefPerThread = (size + nb_threads-1) / nb_threads;
  const size_t offset = static_cast<size_t>(tidy) * size;
  int indexInit, index;
  int8_t offset_d_gen;
  
  if(wordId < nb_words){
    for(int coef=0; coef<coefPerThread; coef++){
      indexInit = tidx + nb_threads*coef;
      index = indexInit;
      if (indexInit < size){
        // Compute all perm in word  
        for(int j=0; j<size_word; j++){
          offset_d_gen = d_words[j + wordId * size_word];
          if(offset_d_gen > -1)
            index = d_gen[index + offset_d_gen*size];
        }
        // Compute last perm
        index = d_gen[index + (tidy%nb_gen)*size];
        workSpace[static_cast<size_t>(indexInit) + offset] = index;
      }
    }
  }
}


template <typename T>
__global__ void equal_kernel(const T* __restrict__ d_gen, const int8_t* __restrict__ d_words, 
                              int* d_equal, const int size, const int size_word, const int8_t nb_gen){
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int nb_threads = blockDim.x*gridDim.x;
  const int wid = threadIdx.x / warpSize;
  const int lane = threadIdx.x % warpSize;
  static __shared__ int shared[32];
  const int coefPerThread = (size + nb_threads-1) / nb_threads;
  int equal=0;
  int8_t offset_d_gen1, offset_d_gen2;
  int indexPerm1, indexPerm2;
  
  // Initialize d_equal
  if(tid == 0)
    d_equal[0] = 0;
	// Initialize shared memory
  if(threadIdx.x <32)
    shared[threadIdx.x] = 0;
       
  // Compute all perm in words
  for(int coef=0; coef<coefPerThread; coef++){
    indexPerm1 = tid + nb_threads * coef;
    indexPerm2 = indexPerm1;
    if (indexPerm1 < size){
      for(int j=0; j<size_word; j++){
        offset_d_gen1 = d_words[j];
        offset_d_gen2 = d_words[j + size_word];
        if (offset_d_gen1 > -1)
            indexPerm1 = d_gen[indexPerm1 + offset_d_gen1 * size];
        if (offset_d_gen2 > -1)
            indexPerm2 = d_gen[indexPerm2 + offset_d_gen2 * size]; 
      }
      // Compare results
      if(indexPerm1 == indexPerm2)
        equal += 1;
    }
  }

  // Reduction of comparison result
  for (unsigned int offset = warpSize/2; offset > 0; offset /= 2) 
      equal += __shfl_down(equal, offset); // Cuda > 7
      //~ equal += __shfl_down_sync(0xffffffff, equal, offset); // Cuda > 9

  if(size > 32 && blockDim.x > 32){
    if (lane == 0)
      shared[wid] = equal;
    __syncthreads();    
    if (wid == 0) {
      equal = shared[lane];
      __syncthreads();   
      for (unsigned int offset = warpSize/2; offset > 0; offset /= 2) 
          equal += __shfl_down(equal, offset); // Cuda > 7
          //~ equal += __shfl_down_sync(0xffffffff, equal, offset); // Cuda > 9
    }
  }
  if(threadIdx.x == 0){
    atomicAdd(d_equal, equal);
	}
}



#define NB_HASH_FUNC 2

//~ template <typename T>
//~ __global__ void pre_insert_kernel(T* __restrict__ workSpace, uint64_t* d_hashed,
                                  //~ const int size, const int nb_trans,
                                  //~ int* d_equal){
	//~ const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	//~ const uint64_t hash1 = d_hashed[tid*NB_HASH_FUNC];
	//~ const uint64_t hash2 = d_hashed[tid*NB_HASH_FUNC + 1];
  //~ uint64_t hash3, hash4;
  //~ int j=0;
  //~ if(tid < nb_trans){
    //~ for(int i=0; i<tid; i++){
      //~ hash3 = d_hashed[i*NB_HASH_FUNC];
      //~ hash4 = d_hashed[i*NB_HASH_FUNC + 1];
      //~ if(hash1 == hash3 && hash2 == hash4){
        //~ j = 0;
        //~ for(j=0; j<size; j++){
          //~ if(workSpace[i*size + j] != workSpace[tid*size + j])
            //~ break;          
        //~ }
        //~ if(j==size){
          //~ d_hashed[tid*NB_HASH_FUNC] = UINT64_MAX;
          //~ d_hashed[tid*NB_HASH_FUNC + 1] = UINT64_MAX;
          //~ break;
        //~ }
      //~ }
    //~ }
  //~ }
//~ }

template <typename T>
__global__ void pre_insert_kernel(T* __restrict__ workSpace, uint64_t* d_hashed,
                                  const int size, const int nb_trans){
	const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t hashRef1, hashRef2, hashComp1, hashComp2;
  int equal = 0;
  if(tidx < nb_trans){
    hashRef1 = d_hashed[tidx*NB_HASH_FUNC + 0];
    hashRef2 = d_hashed[tidx*NB_HASH_FUNC + 1];
    for(int i=0; i<tidx; i++){
      hashComp1 = d_hashed[i*NB_HASH_FUNC + 0];
      hashComp2 = d_hashed[i*NB_HASH_FUNC + 1];
      if(hashRef1 == hashComp1 && hashRef2 == hashComp2){
        equal = 0;
        for(int j=0; j<size; j++){
          if(workSpace[i*size + j] == workSpace[tidx*size + j])
            equal += 1;          
        }
        if(equal==size){
          d_hashed[tidx*NB_HASH_FUNC + 0] = UINT64_MAX;
          d_hashed[tidx*NB_HASH_FUNC + 1] = UINT64_MAX;
          break;
        }
      }
    }
  }
}





template <typename T>
__global__ void hash_kernel(T* __restrict__ workSpace, uint64_t* d_hashed, const int size, const int nb_trans){
  // kernel with less operation. Each threads compute a part of the polynome with Horner method.
  // A warp computes a hash.
  // Can compute several hash number based on different prime numbers.

  const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  const int coefPerThread = (size+blockDim.y-1) / blockDim.y;
  //~ uint64_t primes[NB_HASH_FUNC] = {13, 17, 19, 23};
  uint64_t primes[NB_HASH_FUNC] = {0x9e3779b97f4a7bb9, 19};
  uint64_t out[NB_HASH_FUNC];
  
  // Compute the lower power of prime that thread is using.
  //~ for(int k=0; k<NB_HASH_FUNC; k++){
    out[0] = 1;
    out[1] = 5381;
    for (int j=0; j<coefPerThread*threadIdx.y; j++)
      out[0] *= primes[0];
  //~ }
  
  uint64_t coef=0;
  // Initiale compute stage
  if(threadIdx.y + blockDim.y * 0 < size && tidx < nb_trans){
    coef = workSpace[tidx*size + threadIdx.y + blockDim.y*0];
    //~ for(int k=0; k<NB_HASH_FUNC; k++)
      out[0] *= coef;
      out[1] = ((out[1] << 5) + out[1]) + coef;
  }
  else{
    //~ for(int k=0; k<NB_HASH_FUNC; k++)
      out[0] = 0;  
      out[1] = 5384;
  } 
  coef=0;
  // Compute all stage for the Horner method
  for(int i=1; i<coefPerThread; i++){
    if(threadIdx.y + blockDim.y * i < size && tidx < nb_trans)
      coef = workSpace[tidx*size + threadIdx.y + blockDim.y*i];

    //~ for(int k=0; k<NB_HASH_FUNC; k++){
      out[0] += coef;
      out[0] *= primes[0];
      out[1] = ((out[1] << 5) + out[1]) + coef;
    //~ }

    coef = 0;
  }

  if(threadIdx.y == 0)
    //~ for(int k=0; k<NB_HASH_FUNC; k++)
      d_hashed[NB_HASH_FUNC*tidx + 0] = out[0] >> 32;
      d_hashed[NB_HASH_FUNC*tidx + 1] = out[1];
}


//~ template <typename T>
//~ __global__ void hash_kernel(T* __restrict__ workSpace, uint64_t* d_hashed, const int size, const int nb_trans){
  //~ // kernel with less operation. Each threads compute a part of the polynome with Horner method.
  //~ // A warp computes a hash.
  //~ // Can compute several hash number based on different prime numbers.

  //~ const int tidy = blockIdx.x * blockDim.x + threadIdx.x;
  //~ const int coefPerThread = (size+blockDim.y-1) / blockDim.y;
  //~ uint64_t out[NB_HASH_FUNC] = {5381};  
  //~ uint64_t coef=0;
  //~ for(int i=1; i<coefPerThread; i++){
    //~ if(threadIdx.y + blockDim.y * i < size && tidy < nb_trans)
      //~ coef = workSpace[tidy*size + threadIdx.y + blockDim.y*i];
    //~ for(int k=0; k<NB_HASH_FUNC; k++)
      //~ out[k] = ((out[k] << 5) + out[k]) + coef;
    //~ coef = 0;
  //~ }

  //~ if(threadIdx.y == 0)
    //~ for(int k=0; k<NB_HASH_FUNC; k++)
      //~ d_hashed[NB_HASH_FUNC*tidy + k] = out[k];
      
      
//~ }

template <typename T>
__global__ void initId_kernel(T * __restrict__ workSpace, const int size, const int nb_trans){
  const int coefPerThread = (size + warpSize-1) / warpSize;
  const int tidy = blockIdx.y * blockDim.y + threadIdx.y;
  int index;
  for (int j=0; j<coefPerThread; j++){
    index = threadIdx.x + warpSize*j;
    if(index < size && tidy < nb_trans)
      workSpace[index + tidy*size] = index;    
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
//~ __device__ unsigned hash_function_1(unsigned key, uint64_t* d_hashed){ return (5*(*(d_hashed + key))+11)%PRIME; }
//~ __device__ unsigned hash_function_2(unsigned key, uint64_t* d_hashed){ return (7*(*(d_hashed + key))+13)%PRIME; }
//~ __device__ unsigned hash_function_3(unsigned key, uint64_t* d_hashed){ return (11*(*(d_hashed + key))+19)%PRIME; }
//~ __device__ unsigned hash_function_4(unsigned key, uint64_t* d_hashed){ return (13*(*(d_hashed + key))+23)%PRIME; }

template <typename T>
__device__ bool insert(T * table, T key, uint64_t* hashed){
  
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
