#ifndef VECTOR_GPU_CUH
#define VECTOR_GPU_CUH
#if COMPILE_CUDA==1

template <typename T>
class Vector_gpu {
	public:
		T* host;
		T* device;
		size_t capacity;
		size_t size;
		
		Vector_gpu();	
		Vector_gpu(size_t capacityIn);	
		~Vector_gpu();	
		void realloc();	
		void push_back(T new_elem);	
		void push_back(T* new_array, size_t size_array);	
		void copyHostToDevice();	
		void copyDeviceToHost();
		void clear();
		void swap(Vector_gpu<T> other);
};

// Explicit instentiation
template class Vector_gpu<int>;


#endif  // USE_CUDA
#endif  // VECTOR_GPU_CUH
