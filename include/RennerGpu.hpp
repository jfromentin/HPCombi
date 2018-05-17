#include <array>
#include <iostream>
#include "vector_gpu.cuh"

//~ #define HASH_SIZE 100000
#define NODE 50
//~ #define NB_GEN 6
//~ #define SIZE 100000
//~ #define BLOCK_SIZE 4
#define NB_HASH_FUNC 1

void print_word(std::array<int8_t, NODE> word);

class key
{
  public :
	  key(uint64_t hashed_in, std::array<int8_t, NODE> word_in){
	    hashed = hashed_in;
	    word = word_in;
	  }  
	  key(){}// For dens_hash_map
	  uint64_t hashed;
	  std::array<int8_t, NODE> word;
};



void print_ptr_attr( const cudaPointerAttributes& pa ) {
    std::cout << "Pointer attributes:\n";
    std::string mt = pa.memoryType == cudaMemoryTypeHost ? "cudaMemoryTypeHost"
                                                         : "cudaMemoryTypeDevice";
    std::cout << "  memoryType:    " << mt << std::endl;
    std::cout << "  device:        " << std::hex << pa.device << std::endl;
    std::cout << "  devicePointer: " << std::hex << pa.devicePointer << std::endl;
    std::cout << "  hostPointer:   " << pa.hostPointer << std::endl;
}
