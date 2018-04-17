#ifndef VECTOR_GPU_IMPL_CUH
#define VECTOR_GPU_IMPL_CUH
#if COMPILE_CUDA==1

#include <stdint.h>
#include <stdio.h>
#include "vector_gpu.cuh"

template <typename T>
Vector_gpu<T>::Vector_gpu(){
	capacity = 32;
	size = 0;
	gpuErrchk( cudaMallocHost((void**)&host, capacity * sizeof(T)) );
	gpuErrchk( cudaMalloc((void**)&device, capacity * sizeof(T)) );		
}

template <typename T>
Vector_gpu<T>::Vector_gpu(size_t capacityIn){
	capacity = capacityIn;
	size = 0;
	gpuErrchk( cudaMallocHost((void**)&host, capacity * sizeof(T)) );
	gpuErrchk( cudaMalloc((void**)&device, capacity * sizeof(T)) );		
}

template <typename T>
Vector_gpu<T>::~Vector_gpu(){
	printf("free\n");
	if(host != NULL)
	gpuErrchk( cudaFreeHost(host) );
	if(device != NULL)
	gpuErrchk( cudaFree(device) );	
	host = NULL;
	device = NULL;	
}

template <typename T>
void Vector_gpu<T>::realloc(){
	printf("Resizing, %lu\n", capacity);
	T* oldHost = host;
	T* oldDevice = device;
	gpuErrchk( cudaMallocHost((void**)&host, capacity * sizeof(T)) );
	gpuErrchk( cudaMalloc((void**)&device, capacity * sizeof(T)) );
	for(size_t i=0; i<size; i++){
		host[i] = oldHost[i];
	}
	printf("oldHost : %p, oldDevice : %p\n", oldHost, oldDevice);
	printf("host : %p, device : %p\n", host, device);
	gpuErrchk( cudaFreeHost(oldHost) );
	gpuErrchk( cudaFree(oldDevice) );
	oldHost = NULL;
	oldDevice = NULL;
}

template <typename T>
void Vector_gpu<T>::push_back(T new_elem){
	printf("pushbak1\n");
	if(capacity < size+1){
		capacity *= 2;
		realloc();
	}
	host[size] = new_elem;
	size++;	
}

template <typename T>
void Vector_gpu<T>::push_back(T* new_array, size_t size_array){
	printf("pushbak2\n");
	if(capacity < size+size_array){
		while(capacity < size+size_array)
			capacity *= 2;
		realloc();
	}
	for(size_t i=0; i<size_array; i++){
		host[size+i] = new_array[i];
	}
	size += size_array;
}

template <typename T>
void Vector_gpu<T>::copyHostToDevice(){
	gpuErrchk( cudaMemcpy(device, host, size * sizeof(T), cudaMemcpyHostToDevice) );
}

template <typename T>
void Vector_gpu<T>::copyDeviceToHost(){
	gpuErrchk( cudaMemcpy(host, device, size * sizeof(T), cudaMemcpyDeviceToHost) );
}

template <typename T>
void Vector_gpu<T>::clear(){
	size = 0;
}

template <typename T>
void Vector_gpu<T>::swap(Vector_gpu<T> other){
	printf("swap\n");
	//~ if(other.size > capacity){
		//~ capacity = other.capacity;
		//~ printf("swap1\n");
		//~ realloc();
	//~ }
	//~ else if(size > other.capacity){
		//~ other.capacity = capacity;
		//~ printf("swap2\n");
		//~ other.realloc();
	//~ }
	T* tmpP;
	size_t tmp;
	
	tmpP = host;
	host = other.host;
	other.host = tmpP;
	
	tmpP = device;
	device = other.device;
	other.device = tmpP;
	
	tmp = capacity;
	capacity = other.capacity;
	other.capacity = tmp;
	
	tmp = size;
	size = other.size;
	other.size = tmp;
	tmpP = NULL;
}



#endif  // USE_CUDA
#endif  // VECTOR_GPU_IMPL_CUH
