#ifndef VECTOR_GPU_IMPL_CUH
#define VECTOR_GPU_IMPL_CUH
#if COMPILE_CUDA==1

#include <stdint.h>
#include <stdio.h>
#include "vector_gpu.cuh"

using namespace std;

template <typename T>
Vector_cpugpu<T>::Vector_cpugpu(){
	capacityAtt = 32;
	sizeAtt = 0;
	gpuErrchk( cudaMallocHost((void**)&hostAtt, capacityAtt * sizeof(T)) );
	gpuErrchk( cudaMalloc((void**)&deviceAtt, capacityAtt * sizeof(T)) );		
}

template <typename T>
Vector_cpugpu<T>::Vector_cpugpu(size_t capacityAttIn){
	capacityAtt = capacityAttIn;
	sizeAtt = 0;
	gpuErrchk( cudaMallocHost((void**)&hostAtt, capacityAtt * sizeof(T)) );
	gpuErrchk( cudaMalloc((void**)&deviceAtt, capacityAtt * sizeof(T)) );		
}

template <typename T>
Vector_cpugpu<T>::~Vector_cpugpu(){
	capacityAtt = 0;
	sizeAtt = 0;
	gpuErrchk( cudaFreeHost(hostAtt) );
	gpuErrchk( cudaFree(deviceAtt) );	
	hostAtt = NULL;
	deviceAtt = NULL;
}

template <typename T>
void Vector_cpugpu<T>::realloc(){
	gpuErrchk( cudaFree(deviceAtt) );
	gpuErrchk( cudaMalloc((void**)&deviceAtt, capacityAtt * sizeof(T)) );
	T* oldHost = hostAtt;
	gpuErrchk( cudaMallocHost((void**)&hostAtt, capacityAtt * sizeof(T)) );
	for(size_t i=0; i<sizeAtt; i++){
		hostAtt[i] = oldHost[i];
	}
	gpuErrchk( cudaFreeHost(oldHost) );
	oldHost = NULL;
}

template <typename T>
void Vector_cpugpu<T>::fill(T input){
	sizeAtt = capacityAtt;
	for(int i=0; i<sizeAtt; i++)
		hostAtt[i] = input;
}

template <typename T>
void Vector_cpugpu<T>::push_back(T new_elem){
	resize(sizeAtt + 1);
	hostAtt[sizeAtt] = new_elem;
	sizeAtt++;	
}

template <typename T>
void Vector_cpugpu<T>::push_back(T* new_array, size_t sizeAtt_array){
	resize(sizeAtt + sizeAtt_array);
	for(size_t i=0; i<sizeAtt_array; i++){
		hostAtt[sizeAtt+i] = new_array[i];
	}
	sizeAtt += sizeAtt_array;
}

template <typename T>
size_t Vector_cpugpu<T>::resize(size_t sizeAttIn, int runType, size_t maxMem){
	size_t newCapacity = capacityAtt;
	size_t maxCapacity = maxMem/sizeof(T);
	if(sizeAttIn > maxCapacity && runType > 0){
		cout << "Not enough memory, max capacity : " << static_cast<float>(maxCapacity*sizeof(T)) * 1e-6 
			<< " Mo, required : " << static_cast<float>(sizeAttIn*sizeof(T)) * 1e-6 << " Mo" << endl;
		exit(1);
	}
	if(newCapacity < sizeAttIn){
		while(newCapacity < sizeAttIn)
			newCapacity *= 2;
		if(runType > 0){
			if(newCapacity > maxCapacity)
				cout << "Asked for " << static_cast<float>(newCapacity*sizeof(T)) * 1e-6 << " Mo, but allocating only " 
					<< static_cast<float>(maxCapacity*sizeof(T)) * 1e-6 << " Mo" << endl;
			newCapacity = min(newCapacity, maxCapacity);
			capacityAtt = newCapacity;
			realloc();
		}
	}
	if(runType == 2)
		sizeAtt = sizeAttIn;
	return newCapacity*sizeof(T);
}

template <typename T>
void Vector_cpugpu<T>::copyHostToDevice(size_t offset, size_t copySize) const {
	if(offset < sizeAtt){
		if(copySize == 0)
			copySize = sizeAtt;
		else
			copySize = min(copySize, sizeAtt-offset);
		gpuErrchk( cudaMemcpy(deviceAtt + offset, hostAtt + offset, copySize * sizeof(T), cudaMemcpyHostToDevice) );
	}
	else if(copySize != 0)
		cout << "Offset is bigger than sizeAtt in copyHostToDevice" << endl;
}

template <typename T>
void Vector_cpugpu<T>::copyDeviceToHost(size_t offset, size_t copySize) const {
	if(offset < sizeAtt){
		if(copySize == 0)
			copySize = sizeAtt;
		else
			copySize = min(copySize, sizeAtt-offset);
		gpuErrchk( cudaMemcpy(hostAtt + offset, deviceAtt + offset, copySize * sizeof(T), cudaMemcpyDeviceToHost) );
	}
	else if(copySize != 0)
		cout << "Offset is bigger than sizeAtt in copyDeviceToHost" << endl;
}

template <typename T>
void Vector_cpugpu<T>::clear(){
	sizeAtt = 0;
}

template <typename T>
void Vector_cpugpu<T>::swap(Vector_cpugpu<T>* other){
	T* tmpP;
	size_t tmp;
	
	tmpP = hostAtt;
	hostAtt = other->hostAtt;
	other->hostAtt = tmpP;
	
	tmpP = deviceAtt;
	deviceAtt = other->deviceAtt;
	other->deviceAtt = tmpP;	
	
	tmp = capacityAtt;
	capacityAtt = other->capacityAtt;
	other->capacityAtt = tmp;
	
	tmp = sizeAtt;
	sizeAtt = other->sizeAtt;
	other->sizeAtt = tmp;
	
	tmpP = NULL;
}


//########################################################

template <typename T>
Vector_gpu<T>::Vector_gpu(){
	capacityAtt = 32;
	gpuErrchk( cudaMalloc((void**)&deviceAtt, capacityAtt * sizeof(T)) );		
}

template <typename T>
Vector_gpu<T>::Vector_gpu(size_t capacityAttIn){
	capacityAtt = capacityAttIn;
	gpuErrchk( cudaMalloc((void**)&deviceAtt, capacityAtt * sizeof(T)) );		
}

template <typename T>
Vector_gpu<T>::~Vector_gpu(){
	capacityAtt = 0;
	gpuErrchk( cudaFree(deviceAtt) );
	deviceAtt = NULL;
}

template <typename T>
void Vector_gpu<T>::realloc(){
	gpuErrchk( cudaFree(deviceAtt) );
	gpuErrchk( cudaMalloc((void**)&deviceAtt, capacityAtt * sizeof(T)) );
}

template <typename T>
size_t Vector_gpu<T>::resize(size_t sizeAttIn, int runType, size_t maxMem){
	size_t newCapacity = capacityAtt;
	size_t maxCapacity = maxMem/sizeof(T);
	if(sizeAttIn > maxCapacity && runType > 0){
		cout << "Not enough memory, max capacity : " << static_cast<float>(maxCapacity*sizeof(T)) * 1e-6 
			<< " Mo, required : " << static_cast<float>(sizeAttIn*sizeof(T)) * 1e-6 << " Mo" << endl;
		exit(1);
	}
	if(newCapacity < sizeAttIn){
		float scale = 2;
		if(newCapacity*sizeof(T) > maxMem/2)
			scale = 1.3;
		while(newCapacity < sizeAttIn)
			newCapacity *= scale;
		if(runType > 0){
			if(newCapacity > maxCapacity)
				cout << "Asked for " << static_cast<float>(newCapacity*sizeof(T)) * 1e-6 << " Mo, but allocating only " 
					<< static_cast<float>(maxCapacity*sizeof(T)) * 1e-6 << " Mo" << endl;
			newCapacity = min(newCapacity, maxCapacity);
			capacityAtt = newCapacity;
			realloc();
		}
	}
	return newCapacity*sizeof(T);
}


#endif  // USE_CUDA
#endif  // VECTOR_GPU_IMPL_CUH

