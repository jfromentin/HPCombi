#ifndef VECTOR_GPU_IMPL_CUH
#define VECTOR_GPU_IMPL_CUH
#if COMPILE_CUDA==1

#include <stdint.h>
#include <stdio.h>
#include "vector_gpu.cuh"

template <typename T>
Vector_cpugpu<T>::Vector_cpugpu(){
	capacity = 32;
	size = 0;
	gpuErrchk( cudaMallocHost((void**)&host, capacity * sizeof(T)) );
	gpuErrchk( cudaMalloc((void**)&device, capacity * sizeof(T)) );		
}

template <typename T>
Vector_cpugpu<T>::Vector_cpugpu(size_t capacityIn){
	capacity = capacityIn;
	size = 0;
	gpuErrchk( cudaMallocHost((void**)&host, capacity * sizeof(T)) );
	gpuErrchk( cudaMalloc((void**)&device, capacity * sizeof(T)) );		
}

template <typename T>
Vector_cpugpu<T>::~Vector_cpugpu(){
	capacity = 0;
	size = 0;
	gpuErrchk( cudaFreeHost(host) );
	gpuErrchk( cudaFree(device) );	
	host = NULL;
	device = NULL;
}

template <typename T>
void Vector_cpugpu<T>::realloc(){
	//~ printf("Resizing, %lu\n", capacity);
	T* oldHost = host;
	T* oldDevice = device;
	gpuErrchk( cudaMallocHost((void**)&host, capacity * sizeof(T)) );
	gpuErrchk( cudaMalloc((void**)&device, capacity * sizeof(T)) );
	for(size_t i=0; i<size; i++){
		host[i] = oldHost[i];
	}
	gpuErrchk( cudaFreeHost(oldHost) );
	gpuErrchk( cudaFree(oldDevice) );
	oldHost = NULL;
	oldDevice = NULL;
}

template <typename T>
void Vector_cpugpu<T>::push_back(T new_elem){
	resize(size + 1);
	host[size] = new_elem;
	size++;	
}

template <typename T>
void Vector_cpugpu<T>::push_back(T* new_array, size_t size_array){
	resize(size + size_array);
	for(size_t i=0; i<size_array; i++){
		host[size+i] = new_array[i];
	}
	size += size_array;
}

template <typename T>
void Vector_cpugpu<T>::resize(size_t newCapacity, int force){
	if(capacity < newCapacity){
		while(capacity < newCapacity)
			capacity *= 2;
		realloc();
	}
	if(force == 1)
		size = newCapacity;
}

template <typename T>
void Vector_cpugpu<T>::copyHostToDevice(){
	gpuErrchk( cudaMemcpy(device, host, size * sizeof(T), cudaMemcpyHostToDevice) );
}

template <typename T>
void Vector_cpugpu<T>::copyDeviceToHost(){
	gpuErrchk( cudaMemcpy(host, device, size * sizeof(T), cudaMemcpyDeviceToHost) );
}

template <typename T>
void Vector_cpugpu<T>::clear(){
	size = 0;
}

template <typename T>
void Vector_cpugpu<T>::swap(Vector_cpugpu<T>* other){
	T* tmpP;
	size_t tmp;
	
	tmpP = host;
	host = other->host;
	other->host = tmpP;
	
	tmpP = device;
	device = other->device;
	other->device = tmpP;
	
	
	tmp = capacity;
	capacity = other->capacity;
	other->capacity = tmp;
	
	tmp = size;
	size = other->size;
	other->size = tmp;
	
	tmpP = NULL;
}



//########################################################

template <typename T>
Vector_gpu<T>::Vector_gpu(){
	capacity = 32;
	gpuErrchk( cudaMalloc((void**)&device, capacity * sizeof(T)) );		
}

template <typename T>
Vector_gpu<T>::Vector_gpu(size_t capacityIn){
	capacity = capacityIn;
	gpuErrchk( cudaMalloc((void**)&device, capacity * sizeof(T)) );		
}

template <typename T>
Vector_gpu<T>::~Vector_gpu(){
	capacity = 0;
	gpuErrchk( cudaFree(device) );
	device = NULL;
}

template <typename T>
void Vector_gpu<T>::realloc(){
	//~ printf("Resizing, %lu\n", capacity);
	T* oldDevice = device;
	gpuErrchk( cudaMalloc((void**)&device, capacity * sizeof(T)) );
	gpuErrchk( cudaFree(oldDevice) );
	oldDevice = NULL;
}

template <typename T>
void Vector_gpu<T>::resize(size_t newCapacity){
	if(capacity < newCapacity){
		while(capacity < newCapacity)
			capacity *= 2;
		realloc();
	}
}

#endif  // USE_CUDA
#endif  // VECTOR_GPU_IMPL_CUH
