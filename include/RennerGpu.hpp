#ifndef RENNERGPU
#define RENNERGPU
#include <array>
#include <iostream>
#include "vector_gpu.cuh"

//~ #define HASH_SIZE 100000
#define NODE (7*8)
//~ #define NB_GEN 6
//~ #define SIZE 100000
//~ #define BLOCK_SIZE 4
#define NB_HASH_FUNC 1

void print_word(std::array<int8_t, NODE>);

class Key
{
  private :
	  uint64_t hashedAtt;
	  std::array<int8_t, NODE> wordAtt;
  
  public :
    Key(const uint64_t hashed, const std::array<int8_t, NODE> word) : hashedAtt(hashed), wordAtt(word) {}
    Key(){}// For dense_hash_map
    int8_t &operator[](uint64_t i){ return wordAtt[i]; }
    const int8_t &operator[](uint64_t i) const { return wordAtt[i]; }
    const int8_t* data() const { return wordAtt.data(); }
    uint64_t hashed() const { return hashedAtt; }
    
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
#endif  // RENNERGPU
