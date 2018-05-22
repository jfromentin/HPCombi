#ifndef VECTOR_GPU_CUH
#define VECTOR_GPU_CUH
#if COMPILE_CUDA==1
#include <limits>
#include <typeinfo>
#include <iostream>

template <typename T>
class Vector_cpugpu {
	size_t capacityAtt;
	size_t sizeAtt;
	T* hostAtt;
	T* deviceAtt;
	void realloc();
	public:		
		Vector_cpugpu();	
		Vector_cpugpu(size_t capacityIn);	
		~Vector_cpugpu();
		void fill(T input);
		void push_back(T new_elem);	
		void push_back(T* new_array, size_t size_array);
		size_t resize(size_t newCapacity, int runType=1, size_t maxMem=std::numeric_limits<size_t>::max());
		void copyHostToDevice(size_t offset=0, size_t copySize=0) const;	
		void copyDeviceToHost(size_t offset=0, size_t copySize=0) const;
		void clear();
		void swap(Vector_cpugpu<T>* other);
		T &operator[](uint64_t i){ return hostAtt[i]; }
		const T &operator[](uint64_t i) const { return hostAtt[i]; }
		size_t capacity() const { return capacityAtt; }
		size_t size() const { return sizeAtt; }
		T* device() const { return deviceAtt; }
		T* host() const { return hostAtt; }
};

// Explicit instentiation
template class Vector_cpugpu<int>;
template class Vector_cpugpu<uint64_t>;
template class Vector_cpugpu<int8_t>;


template <typename T>
class Vector_gpu {
	size_t capacityAtt;
	T* deviceAtt;
	void realloc();
	public:		
		Vector_gpu();	
		Vector_gpu(size_t sizeIn);	
		~Vector_gpu();
		size_t resize(size_t sizeIn, int runType=1, size_t maxMem=std::numeric_limits<size_t>::max());
		size_t capacity() const { return capacityAtt; }
		T* device() const { return deviceAtt; }
};

// Explicit instentiation
template class Vector_gpu<uint32_t>;

template <typename T>
std::ostream &operator<<(std::ostream &stream, const Vector_cpugpu<T> &vect) {
  for(int i=0; i<vect.size; i++){
	if(std::is_same<uint8_t, T>::value || std::is_same<int8_t, T>::value)
		stream << static_cast<int>(vect.host[i]) << "|";
	}
  stream << std::endl;
  return stream;
}

#endif  // USE_CUDA
#endif  // VECTOR_GPU_CUH

