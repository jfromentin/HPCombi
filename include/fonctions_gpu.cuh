#ifndef HPCOMBI_PERM_FONCTIONS_GPU_CUH
#define HPCOMBI_PERM_FONCTIONS_GPU_CUH
#if COMPILE_CUDA==1
	#include <cuda_runtime.h>
	#include <cuda.h>
	#include "vector_gpu.cuh"
	#include "RennerGpu.hpp"
	                           
	/** @brief Compose and hash transformations.
	* @details 1st stage) words array contains nb_words suites of generators (named by there indexes) 
	* 		that are applyed to the identity transformation.
	* 		2d stage) For each resulting transformation one more generator is applyed 
	* 		to compute next generation set of transformation.
	* 		Hence nb_gen transformation are compute for each result of the first stage.
	* 		for a total of nb_words*nb_gen transformations. 
	*		3d stage) The hash value are then computed and stored in hashed array.
	* @param words Array of size size_word*nb_words containing the words.
	* @param workSpace Array containing the transformations. 
	* 		Will be resized if needed to be of size size*nb_words*nb_gen.
	* @param d_gen Array allocaded on GPU of size size*nb_gen containing the generators.
	* @param hashed Array containing hash values.
	* 		Will be resized if needed to be of size nb_gen*nb_words.
	* @param size Size of one transformations.
	* @param size_word Size of one word.
	* @param nb_gen Number of generators.
	* @param memory Size in bytes of available memory on GPU.
	*/
	void hpcombi_gpu(Vector_cpugpu<int8_t>& words, Vector_gpu<uint32_t>& workSpace, 
					const uint32_t* __restrict__ d_gen, Vector_cpugpu<uint64_t>& hashed, 
					const int size, const int size_word, const int8_t nb_gen, size_t memory);

	/** @brief Initialize workSpace to identity and hashes identity.
	 * 		Needed to initialise hash table.
	* @param hashed Array containing hash values.
	* @param workSpace Array containing the transformations. 
	* 		Will be resized if needed to be of size size.
	* @param size Size of one transformations.
	*/ 					
	void hash_id_gpu(Vector_cpugpu<uint64_t>& hashed, Vector_gpu<uint32_t>& workSpace, const int size);
                           
	/** @brief Check equality of the resulting transformation of two words.
	* 		1st stage) key1 and key2 objects contains each one a word.
	* 		Those words are array the contains a suites of generators (named by there indexes) 
	* 		that are by applyed to the identity transformation.
	* 		2d stage) The resulting transformation are compared element by element
	* 		and the number of equal coeficient is stored in d_equal.
	* @param key1 Key to compare with key2.
	* @param key2 Key to compare with key1.
	* @param d_gen Array allocaded on GPU of size size*nb_gen containing the generators.
	* @param d_words Array allocaded on GPU of size 2*size_word in wich the words are copied.
	* @param d_equal Number of equal coeficient.
	* 		Must be equal to size for the transformation to be equals.
	* @param size Size of one transformations.
	* @param nb_gen Number of generators.
	* @param size_word Size of one word.
	*/ 	
	bool equal_gpu(const Key& key1, const Key& key2, uint32_t* d_gen, int8_t* d_words,
					Vector_cpugpu<int>& equal, const int size, const int8_t nb_gen);

	/** @brief Allocate memory on GPU for generators.
	* @param d_gen Allocated array on GPU containing the generators.
	* @param gen Array on CPU containing the generators.
	* @param size Size of one transformations.
	* @param nb_gen Number of generators.
	* @return 1 if tranformations are equal, 0 if not.
	*/
	void malloc_gen(uint32_t*& __restrict__ d_gen, const uint32_t* __restrict__ gen, 
					const int size, const int8_t nb_gen);

	/** @brief Allocate memory on GPU for words comparison.
	* This enables not to allocated memory on GPU for each comparison.
	* The memory for 2 words is pre-allocated.
	* @param d_words Allocated array on GPU.
	* @param size Size of one transformations.
	*/
	void malloc_words(int8_t*& __restrict__ d_words, const int size);
	void free_gen(uint32_t*& __restrict__ d_gen);
	void free_words(int8_t*& __restrict__ d_words);
	
	/** @brief Set the GPU to use and print informations about it.
	* @return The size of global memory available in bytes
	*/
	size_t cudaSetDevice_cpu();

#endif  // USE_CUDA
#endif  // HPCOMBI_PERM_FONCTIONS_GPU_CUH
