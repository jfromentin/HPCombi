#ifndef VECTOR_GPU_CUH
#define VECTOR_GPU_CUH
#if COMPILE_CUDA==1
#include <limits>
#include <typeinfo>
#include <iostream>

/** @class Vector_cpugpu
*	@brief Dynamic array allocated on both CPU and GPU.
* @details
*
*/
template <typename T>
class Vector_cpugpu {
	size_t capacityAtt; 	/**< Allocated space for the array */
	size_t sizeAtt; 		/**< Number of elemetn pushed in the array */
	T* hostAtt; 			/**< Pointer to host array */
	T* deviceAtt; 			/**< Pointer to device array */
	/** @brief Reallocate memory with capacity*2 on both CPU and GPU.
	* @details Only CPU data is copyed to the new allocated array.
	* 			One must call copyHostToDevice() member function to copy data on GPU.
	*/
	void realloc();
	public:
		/** @brief Allocate memory on both CPU and GPU 
		* Default size of the array : 32.
		*/
		Vector_cpugpu();
		/** @brief Allocate memory on both CPU and GPU 
		* @param capacityIn Size of the array.
		*/
		Vector_cpugpu(size_t capacityIn);
		/** @brief Free memory on both CPU and GPU 
		*/
		~Vector_cpugpu();
		/** @brief Fill host vector with provided number.
		* @details One must call copyHostToDevice() member function to copy data on GPU.
		* @param input Value to fill in.
		*/
		void fill(T input);
		/** @brief Add an element to host array.
		* @details One must call copyHostToDevice() member function to copy data on GPU.
		* @param new_elem Element to add.
		*/
		void push_back(T new_elem);
		/** @brief Add an array to host array.
		* @details One must call copyHostToDevice() member function to copy data on GPU.
		* @param new_array Array to add.
		* @param size_array Size of the array to add.
		*/
		void push_back(T* new_array, size_t size_array);
		/** @brief Checks if capacity of Vector_cpugpu is greater than newCapacity and reallocate if needed.
		* @param newCapacity Capaciy to have.
		* @param runType 
		* 0 : Only check capacity but do not reallocate.
		* 1 : Check capacity and reallocate.
		* 2 : Check capacity, reallocate and est size to capacity.
		* @param maxMem Maximum amount of memory allowed to allocate.
		* @return New size in bytes allocated (or that would be allocated if runtype=0)
		*/
		size_t resize(size_t newCapacity, int runType=1, size_t maxMem=std::numeric_limits<size_t>::max());
		/** @brief Copy data from CPU to GPU.
		* @param offset The staring block to copy
		* @param copySize The number of block to copy.
		* 		Setting it to 0 actualy sets it up to size.
		*/
		void copyHostToDevice(size_t offset=0, size_t copySize=0) const;
		/** @brief Copy data from GPU to CPU.
		*/
		void copyDeviceToHost(size_t offset=0, size_t copySize=0) const;
		/** @brief Sets size to 0 but do not free memory.
		* @param offset The staring block to copy
		* @param copySize The number of block to copy.
		* 		Setting it to 0 actualy sets it up to size.
		*/
		void clear();
		/** @brief Swaps two Vector_cpugpu.
		* @param other The other Vector_cpugpu to swap with.
		*/
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


/** @class Vector_gpu
*	@brief Dynamic array allocated on GPU.
* @details
*
*/
template <typename T>
class Vector_gpu {
	size_t capacityAtt;		/**< Allocated space for the array */
	T* deviceAtt;			/**< Pointer to device array */
	/** @brief Reallocate memory with capacity*2.
	*/
	void realloc();
	public:
		/** @brief Allocate memory on GPU 
		* Default size of the array : 32.
		*/
		Vector_gpu();
		/** @brief Allocate memory on GPU 
		* @param capacityAttIn Size of the array.
		*/
		Vector_gpu(size_t capacityAttIn);
		/** @brief Free memory on GPU 
		*/
		~Vector_gpu();
		/** @brief Checks if capacity of Vector_cpugpu is greater than newCapacity and reallocate if needed.
		* @param newCapacity Capaciy to have.
		* @param runType 
		* 0 : Only check capacity but do not reallocate.
		* 1 : Check capacity and reallocate.
		* 2 : Check capacity, reallocate and est size to capacity.
		* @param maxMem Maximum amount of memory allowed to allocate.
		* @return New size in bytes allocated (or that would be allocated if runtype=0)
		*/
		size_t resize(size_t sizeAttIn, int runType=1, size_t maxMem=std::numeric_limits<size_t>::max());
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

