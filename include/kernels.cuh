#ifndef HPCOMBI_PERM_KERNELS_CUH
#define HPCOMBI_PERM_KERNELS_CUH

#include <stdint.h>
#include <stdio.h>

template <typename T>
__global__ void permute_gpu (T * __restrict__ d_x, T * __restrict__ d_y, const size_t size);
template <typename T>
__global__ void permute_gpu_gen (T * __restrict__ d_x, T * __restrict__ d_y, const size_t size);

__global__ void hash_kernel(uint32_t* __restrict__ d_x, uint64_t* d_hashed, const size_t size);



template <typename T>
__global__ void permute_gpu (T * __restrict__ d_x, T * __restrict__ d_y, const size_t size) {
  // Global thread id and warp id
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t wid = threadIdx.x/warpSize;
  
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
__global__ void permute_gpu_gen (T * __restrict__ d_x, T * __restrict__ d_y, const size_t size) {
  // Global thread id and warp id
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;  
  if (tid < size){
	d_y[tid] = d_x[d_y[tid]];
  }
}

#define NB_COEF 2

__global__ void hash_kernel(uint32_t* __restrict__ d_x, uint64_t* d_hashed, const size_t size) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t wid = threadIdx.x / warpSize;
  const size_t lane = threadIdx.x % warpSize;
  const size_t offset = blockDim.x * gridDim.x;
  const size_t coefPerThread = (size+blockDim.x-1) / blockDim.x;
  static __shared__ uint64_t shared[32];
  const uint64_t prime = 0x9e3779b97f4a7bb9;
  uint64_t tmp1=1;
  uint64_t tmp2=1;
  uint64_t out=0;
  uint64_t coef=0;

  if (wid == 0)
    shared[lane] = 0;
    
  for (size_t j=0; j<tid; j++)
    tmp1 *= prime;
   //~ tmp1 = __mul64hi(tmp1, prime);
    
  tmp2 = tmp1;
  for(size_t i=0; i<coefPerThread; i++){
    if(tid + offset * i < size)
      coef = d_x[tid + offset * i];           

    for (size_t j=0; j<offset; j++)
      tmp2 *= prime;
      //~ tmp2 = __mul64hi(tmp2, prime);

    out += coef*tmp1;
    tmp1 = tmp2;
    coef = 0;
    }

  // Reduction
  for (size_t offset = warpSize/2; offset > 0; offset /= 2) 
      out += __shfl_down(out, offset);

  if(blockDim.x > 32){
    if (lane == 0)
      shared[wid] = out;
      
    __syncthreads();
    
    if (wid == 0) {
      out = shared[lane];     
      for (size_t offset = warpSize/2; offset > 0; offset /= 2) 
          out += __shfl_down(out, offset);
    }
  }

	//~ ((v1 * prime + v0) * prime) >> 64;
  //~ long long int 	__mul64hi ( long long int x, long long int y )
	
  if(tid == 0)
    *d_hashed = out;
	
	
}
#endif  // HPCOMBI_PERM_KERNELS_CUH
