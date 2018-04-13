#ifndef HPCOMBI_PERM_KERNELS_CUH
#define HPCOMBI_PERM_KERNELS_CUH

#include <stdint.h>
#include <stdio.h>

template <typename T>
__global__ void permute_gpu (T * __restrict__ d_x, T * __restrict__ d_y, const int size);
template <typename T>
__global__ void permute_gpu_gen (T * __restrict__ d_x, T * __restrict__ d_y, const int size);

__global__ void hpcombi(uint32_t* __restrict__ d_x, uint64_t* d_hashed, const int size, const int nb_vect);
__global__ void hpcombi_kernel(uint32_t* __restrict__ d_x, const uint32_t* __restrict__ d_gen, const uint32_t* __restrict__ d_words, uint64_t* d_hashed, 
                              const int size, const int nb_words, const int size_word, const int nb_gen);
                              
__global__ void permute_all_kernel(uint32_t* __restrict__ d_x, const uint32_t* __restrict__ d_gen, const int* __restrict__ d_words,
                              const int size, const int nb_words, const int size_word, const int nb_gen);
__device__ void hash_kernel1(uint32_t* __restrict__ d_x, uint64_t* d_hashed, const int size, const int nb_vect);
__device__ void hash_kernel2(uint32_t* __restrict__ d_x, uint64_t* d_hashed, const int size, const int nb_vect);
__device__ void hash_kernel3_device(uint32_t* __restrict__ d_x, uint64_t* d_hashed, const int size, const int nb_vect);
__global__ void hash_kernel3(uint32_t* __restrict__ d_x, uint64_t* d_hashed, const int size, const int nb_vect);
__device__ void permute_kernel(uint32_t* __restrict__ d_x, const uint32_t* __restrict__ d_y, const int size, const int num);
__global__ void initId_kernel(uint32_t * __restrict__ d_x, const int size, const int nb_words, const int nb_gen);
__global__ void equal_kernel(uint32_t* __restrict__ d_x, const int* __restrict__ d_words, int* d_equal, const int size, const int size_word, const int nb_gen);
__global__ void equal_kernel2(uint32_t* __restrict__ d_x, const int* __restrict__ d_words, int* d_equal, const int size, const int size_word, const int nb_gen);



template <typename T>
__global__ void permute_gpu (T * __restrict__ d_x, T * __restrict__ d_y, const int size) {
  // Global thread id and warp id
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int wid = threadIdx.x/warpSize;
  
  // Copy in register
  const T x_reg = d_x[tid];
  const T y_reg = d_y[tid];
  T tmp = 0;
  
  // Mask for shuffle greater than warp size
  const bool mask_shared = (y_reg/warpSize == wid) ? 1:0;
  const bool mask_global = (y_reg/blockDim.x == blockIdx.x) ? 1:0;
  
  // Shared memory for shuffle greater than warp size lesser than block size
  // extern __shared__ T my_shared[]; // Incompatible with use of template
  extern __shared__ __align__(sizeof(T)) unsigned char my_shared[];
  T *shared = reinterpret_cast<T *>(my_shared);
  
  if (tid < size){
	// Copy in shared memory for shared memory shuffle
	shared[threadIdx.x] = x_reg;
	
	 // Warp shuffle
	 // y_reg is automaticaly set to y_reg%warpSize 
     tmp = __shfl(x_reg, y_reg); // Todo try with tmp save in register
										
	// Waitting for all thread to finish shared memory writing
	__syncthreads();	
	//Shared memory shuffle
	if(mask_global == false){
		d_y[tid] = d_x[d_y[tid]];
	}
	else if(mask_shared == false){
		d_y[tid] = shared[y_reg%blockDim.x];
	}
	else{
		d_y[tid] = tmp;
	}
	
  }
}


template <typename T>
__global__ void permute_gpu_gen (T * __restrict__ d_x, T * __restrict__ d_y, const int size) {
  // Global thread id and warp id
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;  
  if (tid < size){
	d_y[tid] = d_x[d_y[tid]];
  }
}


__global__ void hpcombi(uint32_t* __restrict__ d_x, uint64_t* d_hashed, const int size, const int nb_vect){
	hash_kernel3_device(d_x, d_hashed, size, nb_vect);
}

__global__ void hpcombi_kernel(uint32_t* __restrict__ d_x, const uint32_t* __restrict__ d_gen, const int* __restrict__ d_words, uint64_t* d_hashed, 
                              const int size, const int nb_words, const int size_word, const int nb_gen){
                                
  //~ if(threadIdx.x +size< nb_gen*size)
    //~ printf("Thread : %d, d_gen[thread] : %u\n", threadIdx.x, d_gen[threadIdx.x+size]);
  for(int i=0; i<nb_words; i++){
    for(int k=0; k<nb_gen; k++){
      for(int j=0; j<size_word; j++){
        if(d_words[j + i*size_word]>-1)
          permute_kernel(d_x + i*nb_gen*size + k*size, d_gen + d_words[j + i*size_word]*size, size, i*nb_gen + k);
      }
      permute_kernel(d_x + i*nb_gen*size + k*size, d_gen + k*size, size, i*nb_gen + k);
    }
  }
  hash_kernel3_device(d_x, d_hashed, size, nb_words*nb_gen);
}

__global__ void permute_all_kernel(uint32_t* __restrict__ d_x, const uint32_t* __restrict__ d_gen, const int* __restrict__ d_words, 
                              const int size, const int nb_words, const int size_word, const int nb_gen){
  const int tidy = blockIdx.y * blockDim.y + threadIdx.y;
  const int lane = threadIdx.x;
  const int coefPerThread = (size+warpSize-1) / warpSize;

  const int offset_d_x =  tidy * size;
  int offset_d_gen;
  int index;
  //~ if(tidy==0&&lane==0)
  //~ printf("permute pointeur : %u, size : %d\n", d_gen[17], size);
  //~ printf("permute pointeur : %p\n", d_gen);
  
  if(tidy/nb_gen<nb_words){
    for(int j=0; j<size_word; j++){
      offset_d_gen =  d_words[j + (tidy/nb_gen) * size_word];
      //~ if(offset_d_gen>=nb_gen)
        //~ printf("offset_d_gen : %d\n", offset_d_gen);
      //~ if(tidy==4&&lane==0)
        //~ printf("j : %d, offset_d_gen : %d\n", j, offset_d_gen);
      if(offset_d_gen > -1){
        for(int coef=0; coef<coefPerThread; coef++){
          index = lane + warpSize * coef;
          if (index < size){
            //~ d_x[index + offset_d_x] = d_gen[ d_x[index + offset_d_x] + offset_d_gen];
                      //~ if(d_gen[index + offset_d_gen * size] >= size )
      //~ printf("d_gen[] : %d\n", d_gen[index + offset_d_gen * size]);
            d_x[index + offset_d_x] = d_x[ d_gen[index + offset_d_gen * size] + offset_d_x];
            //~ if(tidy==0)
            //~ printf("j : %d, tidy : %d, gen : %d, lane : %d, d_x : %u\n", j, tidy, tidy%nb_gen, lane, d_x[index + offset_d_x]);
          }
        }
      }
    }
    for(int coef=0; coef<coefPerThread; coef++){
      index = lane + warpSize * coef;
      if (index < size){
        //~ d_x[index + offset_d_x] = d_gen[ d_x[index + offset_d_x] + tidy%nb_gen * size];
        d_x[index + offset_d_x] = d_x[ d_gen[index + (tidy%nb_gen) * size] + offset_d_x];
        //~ if( tidy >= 1*powf(nb_gen, size_word) && tidy < 2*powf(nb_gen, size_word) )
        //~ printf("tidy : %d, gen : %d, lane : %d, d_x : %u\n", tidy, tidy%nb_gen, lane, d_x[index + offset_d_x]);
      }
    }
  }
}



__global__ void equal_kernel2(uint32_t* __restrict__ d_x, const uint32_t* __restrict__ d_gen, const int* __restrict__ d_words, 
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
  //~ if(tid==0)
  //~ printf("equal pointeur : %u, size : %d\n", d_gen[17], size);
  //~ printf("equal pointeur : %p\n", d_gen);
  
  for(int j=0; j<size_word; j++){
    offset_d_gen1 =  d_words[j];
    offset_d_gen2 =  d_words[j + size_word];
    if(offset_d_gen1 > nb_gen || offset_d_gen2 > nb_gen)
      printf("offset_d_gen : %d, %d\n", offset_d_gen1, offset_d_gen2 );
      for(int coef=0; coef<coefPerThread; coef++){
        index = tid + nb_threads * coef;    
        if (index < size){ 
          //~ if(d_gen[index + offset_d_gen1 * size] > size || d_gen[index + offset_d_gen2 * size] > size)
      //~ printf("d_gen[] : %u, %u, index : %d\n", d_gen[index + offset_d_gen1 * size], d_gen[index + offset_d_gen2 * size], index + offset_d_gen1 * size );
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
  //~ if(tid == 0)
    //~ printf("equal : %d, coefPerThread : %d\n", equal, coefPerThread);
  if(threadIdx.x == 0){
    //~ printf("equal block : %d, block : %d\n", equal, blockIdx.x);
    atomicAdd(d_equal, equal);
  }    
  //~ if(tid == 0){
    //~ if(d_equal[0] == size)
      //~ d_equal[0] = 1;
    //~ else
      //~ d_equal[0] = 0;
  //~ }
}


__global__ void equal_kernel(uint32_t* __restrict__ d_x, const uint32_t* __restrict__ d_gen, const int* __restrict__ d_words, 
                              int* d_equal, const int size, const int size_word, const int nb_gen){
  const int tidy = blockIdx.y * blockDim.y + threadIdx.y;
  const int lane = threadIdx.x;
  const int coefPerThread = (size+warpSize-1) / warpSize;
  int equal=0;

  const int offset_d_x =  tidy * size;
  int offset_d_gen;
  int index;
  
  if(tidy<2){
    for(int j=0; j<size_word; j++){
      offset_d_gen =  d_words[j + tidy * size_word];
      if(offset_d_gen>nb_gen)
        printf("offset_d_gen : %d\n", offset_d_gen);
      if(offset_d_gen > -1){
        for(int coef=0; coef<coefPerThread; coef++){
          index = lane + warpSize * coef;
          if (index < size){
                                  //~ if(d_gen[index + offset_d_gen * size] > size )
      //~ printf("d_gen[] : %d\n", d_gen[index + offset_d_gen * size]);
            d_x[index + offset_d_x] = d_x[ d_gen[index + offset_d_gen * size] + offset_d_x];
          }
        }
      }
    }
  }
	if(tidy == 0){
    if(lane == 0)
      d_equal[0] = 1;
    for(int coef=0; coef<coefPerThread; coef++){
      index = lane + warpSize * coef;
      if (index < size){
        equal = (d_x[index] == d_x[index + size]) ? 1:0;
      }
      for (int offset = warpSize/2; offset > 0; offset /= 2) 
        equal += __shfl_down(equal, offset);
      if(lane == 0){
        
          //~ for(int i=0; i<size_word*2; i++){
            //~ printf("%d|", d_words[i]);
            //~ if(i%size_word == size_word-1)
          //~ printf("\n");
          //~ }
          //~ printf("\n");
          
          //~ for(int i=0; i<size*2; i++){
            //~ printf("%u|", d_x[i]);
            //~ if(i%size == size-1)
          //~ printf("\n");
          //~ }
          //~ printf("\n");
          
        if(equal != 32 && equal != size){
          d_equal[0] = 0;
          //~ printf("equal : %d, sum : %d\n", *d_equal, equal);
          break;
        }
        else{
          
          
        }
      }
      
    }
	  
  }

//~ if(tidy == 0 and lane == 0)
//~ printf("equal : %d\n", d_equal);
  
}


__device__ void hash_kernel1(uint32_t* __restrict__ d_x, uint64_t* d_hashed, const int size, const int nb_vect){
  // Kernel with more operation. Every threads compute X^n.
  // A block computes a hash
  
  //~ const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int wid = threadIdx.x / warpSize;
  const int lane = threadIdx.x % warpSize;
  const int offset_global = blockDim.x;
  const int coefPerThread = (size+blockDim.x-1) / blockDim.x;
  static __shared__ uint64_t shared[32];
  const uint64_t prime = 0x9e3779b97f4a7bb9;
  uint64_t tmp=1;
  uint64_t out=0;
  uint64_t coef=0;

  if (wid == 0)
    shared[lane] = 0;
    
  for (int j=0; j<threadIdx.x; j++)
    tmp *= prime;

  for(int i=0; i<coefPerThread; i++){
    if(threadIdx.x + offset_global * i < size && blockIdx.x < nb_vect)
      coef = d_x[blockIdx.x * size + threadIdx.x + offset_global * i];

    out += coef*tmp;
    for (int j=0; j<offset_global; j++)
      tmp *= prime;

    coef = 0;
    }

  // Reduction
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
      out += __shfl_down(out, offset);

  if(blockDim.x > 32){
    if (lane == 0)
      shared[wid] = out;
      
    __syncthreads();
    
    if (wid == 0) {
      out = shared[lane];     
      for (int offset = warpSize/2; offset > 0; offset /= 2) 
          out += __shfl_down(out, offset);
    }
  }	
  if(threadIdx.x == 0)
    d_hashed[blockIdx.x] = out;
}

__device__ void hash_kernel2(uint32_t* __restrict__ d_x, uint64_t* d_hashed, const int size, const int nb_vect){
  // Kernel with more operation. Every threads compute X^n.
  // A warp computes a hash.
  
  //~ const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  const int tidy = blockIdx.y * blockDim.y + threadIdx.y;
  //~ const int lane = threadIdx.x % warpSize;
  const int lane = threadIdx.x;
  const int coefPerThread = (size+warpSize-1) / warpSize;
  const uint64_t prime = 0x9e3779b97f4a7bb9;
  uint64_t tmp=1;
  uint64_t out=0;
  uint64_t coef=0;
    
  for (int j=0; j<lane; j++)
    tmp *= prime;

  for(int i=0; i<coefPerThread; i++){
    if(lane + warpSize * i < size && tidy < nb_vect)
      coef = d_x[tidy * size + lane + warpSize * i];

    out += coef*tmp;
    for (int j=0; j<warpSize; j++)
      tmp *= prime;

    coef = 0;
    }

  // Reduction
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
      out += __shfl_down(out, offset);

  if(lane == 0)
    d_hashed[tidy] = out;
}

__global__ void hash_kernel3(uint32_t* __restrict__ d_x, uint64_t* d_hashed, const int size, const int nb_vect){
  // kernel with less operation. Each threads compute a part of the polynome with Horner method.
  // A warp computes a hash.
  
  //~ const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  const int tidy = blockIdx.y * blockDim.y + threadIdx.y;
  //~ const int lane = threadIdx.x % warpSize;
  const int lane = threadIdx.x;
  const int coefPerThread = (size+warpSize-1) / warpSize;
  const uint64_t prime = 0x9e3779b97f4a7bb9;
  //~ uint64_t tmp=1;
  uint64_t out=1;
  uint64_t coef=0;
    
  for (int j=0; j<coefPerThread*lane; j++)
    out *= prime;
	//~ out *= d_x[tidy * size + lane * coefPerThread + 0];
  if(lane + warpSize * 0 < size && tidy < nb_vect){
    out *= d_x[tidy * size + lane + warpSize * 0];
    //~ printf("lane : %d, out : %u\n", lane, out);
  }
  else
    out = 0;

  for(int i=1; i<coefPerThread; i++){
    //~ if(lane * coefPerThread + i < size && tidy < nb_vect)
      //~ coef = d_x[tidy * size + lane * coefPerThread + i];
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
    d_hashed[tidy] = out;
}


__device__ void hash_kernel3_device(uint32_t* __restrict__ d_x, uint64_t* d_hashed, const int size, const int nb_vect){
  // kernel with less operation. Each threads compute a part of the polynome with Horner method.
  // A warp computes a hash.
  
  //~ const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  const int tidy = blockIdx.y * blockDim.y + threadIdx.y;
  //~ const int lane = threadIdx.x % warpSize;
  const int lane = threadIdx.x;
  const int coefPerThread = (size+warpSize-1) / warpSize;
  const uint64_t prime = 0x9e3779b97f4a7bb9;
  //~ uint64_t tmp=1;
  uint64_t out=1;
  uint64_t coef=0;
    
  for (int j=0; j<coefPerThread*lane; j++)
    out *= prime;
	//~ out *= d_x[tidy * size + lane * coefPerThread + 0];
  if(lane + warpSize * 0 < size && tidy < nb_vect){
    out *= d_x[tidy * size + lane + warpSize * 0];
    //~ printf("lane : %d, out : %u\n", lane, out);
  }
  else
    out = 0;

  for(int i=1; i<coefPerThread; i++){
    //~ if(lane * coefPerThread + i < size && tidy < nb_vect)
      //~ coef = d_x[tidy * size + lane * coefPerThread + i];
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
    d_hashed[tidy] = out;
}


__device__ void permute_kernel(uint32_t* __restrict__ d_x, const uint32_t* __restrict__ d_y, const int size, const int num){
  const int tidy = blockIdx.y * blockDim.y + threadIdx.y;
  //~ const int wid = threadIdx.x / warpSize;
  //~ const int lane = threadIdx.x % warpSize;
  const int lane = threadIdx.x;
  const int coefPerThread = (size+warpSize-1) / warpSize;  

  if(tidy == num){
    for(int i=0; i<coefPerThread; i++){
      //~ if (lane * coefPerThread + i < size)
        //~ d_y[lane * coefPerThread + i] = d_x[d_y[lane * coefPerThread + i]];
      if (lane + warpSize * i < size){
        //~ if (d_x[lane + warpSize * i] > size)
        //~ printf("value : %u, lane : %d, size : %lu, tidy : %d\n", d_x[lane + warpSize * i], lane, size, tidy);
        d_x[lane + warpSize * i] = d_y[d_x[lane + warpSize * i]];
        }
    }
  }
}

__global__ void initId_kernel(uint32_t * __restrict__ d_x, const int size, const int nb_words, const int nb_gen){
  const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  //~ printf("tidx : %lu\n", size*nb_words*nb_gen);
  if(tidx < size*nb_words*nb_gen){
    d_x[tidx] = tidx%size;
    //~ printf("tidx : %d, out : %u, tidxsize : %u\n", tidx, d_x[tidx], tidx%size);
  }
}

#endif  // HPCOMBI_PERM_KERNELS_CUH
