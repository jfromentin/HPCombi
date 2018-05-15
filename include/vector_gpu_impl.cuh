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
	//~ printf("Realloc : %f Go\n", (float)(capacity* sizeof(T))*1e-9);
	gpuErrchk( cudaFree(device) );
	gpuErrchk( cudaMalloc((void**)&device, capacity * sizeof(T)) );
	T* oldHost = host;
	gpuErrchk( cudaMallocHost((void**)&host, capacity * sizeof(T)) );
	for(size_t i=0; i<size; i++){
		host[i] = oldHost[i];
	}
	gpuErrchk( cudaFreeHost(oldHost) );
	oldHost = NULL;

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
size_t Vector_cpugpu<T>::resize(size_t sizeIn, int runType, size_t maxMem){
	size_t newCapacity = capacity;
	size_t maxCapacity = maxMem/sizeof(T);
	if(sizeIn > maxCapacity && runType > 0){
		printf("Not enough memory\n");
		exit(1);
	}
	if(newCapacity < sizeIn){
		while(newCapacity < sizeIn)
			newCapacity *= 2;
		if(runType > 0){
			if(newCapacity > maxCapacity)
				printf("Asked for %.2f Mo, but allocating only  %.2f Mo\n", (float)newCapacity*sizeof(T) * 1e-6, (float)maxCapacity*sizeof(T) * 1e-6);
			newCapacity = min(newCapacity, maxCapacity);
			capacity = newCapacity;
			realloc();
		}
	}
	if(runType == 2)
		size = sizeIn;
	return newCapacity*sizeof(T);
}

template <typename T>
void Vector_cpugpu<T>::copyHostToDevice(size_t offset, size_t copySize) const {
	if(offset < size){
		if(copySize == 0)
			copySize = size;
		else
			copySize = min(copySize, size-offset);
		gpuErrchk( cudaMemcpy(device + offset, host + offset, copySize * sizeof(T), cudaMemcpyHostToDevice) );
	}
	else if(copySize != 0)
		printf("Offset is bigger than size in copyHostToDevice\n");
}

template <typename T>
void Vector_cpugpu<T>::copyDeviceToHost(size_t offset, size_t copySize) const {
	if(offset < size){
		if(copySize == 0)
			copySize = size;
		else
			copySize = min(copySize, size-offset);
		gpuErrchk( cudaMemcpy(host + offset, device + offset, copySize * sizeof(T), cudaMemcpyDeviceToHost) );
	}
	else if(copySize != 0)
		printf("Offset is bigger than size in copyDeviceToHost\n");
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

template <typename T>
void Vector_cpugpu<T>::print() const{
	for(int i=0; i<size; i++)
		std::cout << host[i] << "|";
	std::cout << std::endl;
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
	//~ printf("Realloc : %f Go\n", (float)(capacity* sizeof(T))*1e-9);
	gpuErrchk( cudaFree(device) );
	gpuErrchk( cudaMalloc((void**)&device, capacity * sizeof(T)) );
}

template <typename T>
size_t Vector_gpu<T>::resize(size_t sizeIn, int runType, size_t maxMem){
	size_t newCapacity = capacity;
	size_t maxCapacity = maxMem/sizeof(T);
	if(sizeIn > maxCapacity && runType > 0){
		printf("Not enough memory, max capacity : %.2f Mo, required : %.2f Mo\n", (float)maxCapacity*sizeof(T) * 1e-6, (float)sizeIn*sizeof(T) * 1e-6);
		exit(1);
	}
	if(newCapacity < sizeIn){
		//~ printf("newCapacity : %.2f Mo, maxMem : %.2f Mo\n", (float)newCapacity*sizeof(T) * 1e-6, (float)maxMem * 1e-6);
		float scale = 2;
		if(newCapacity*sizeof(T) > maxMem/2)
			scale = 1.3;
		while(newCapacity < sizeIn)
			newCapacity *= scale;
		if(runType > 0){
			if(newCapacity > maxCapacity)
				printf("Asked for %.2f Mo, but allocating only  %.2f Mo\n", (float)newCapacity*sizeof(T) * 1e-6, (float)maxCapacity*sizeof(T) * 1e-6);
			newCapacity = min(newCapacity, maxCapacity);
			capacity = newCapacity;
			realloc();
		}
	}
	return newCapacity*sizeof(T);
}

#endif  // USE_CUDA
#endif  // VECTOR_GPU_IMPL_CUH
