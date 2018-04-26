#ifndef VECTOR_GPU_CUH
#define VECTOR_GPU_CUH
#if COMPILE_CUDA==1

template <typename T>
class Vector_cpugpu {
	public:
		T* host;
		T* device;
		size_t capacity;
		size_t size;
		
		Vector_cpugpu();	
		Vector_cpugpu(size_t capacityIn);	
		~Vector_cpugpu();	
		void realloc();	
		void push_back(T new_elem);	
		void push_back(T* new_array, size_t size_array);
		void resize(size_t newCapacity, int force=0);
		void copyHostToDevice() const;	
		void copyDeviceToHost() const;
		void clear();
		void print() const;
		void swap(Vector_cpugpu<T>* other);
		T &operator[](uint64_t i){ return host[i]; }
};

// Explicit instentiation
template class Vector_cpugpu<int>;
template class Vector_cpugpu<uint64_t>;
template class Vector_cpugpu<int8_t>;


template <typename T>
class Vector_gpu {
	public:
		T* device;
		size_t capacity;
		
		Vector_gpu();	
		Vector_gpu(size_t sizeIn);	
		~Vector_gpu();	
		void realloc();
		void resize(size_t newCapacity);
};

// Explicit instentiation
template class Vector_gpu<uint32_t>;


#endif  // USE_CUDA
#endif  // VECTOR_GPU_CUH
