#ifndef RENNERGPU
#define RENNERGPU
#include <array>
#include <iostream>
#include "vector_gpu.cuh"

//~ #define HASH_SIZE 100000
#define NODE (1*16)
//~ #define NB_GEN 6
//~ #define SIZE 100000
//~ #define BLOCK_SIZE 4
#define NB_HASH_FUNC 2
typedef unsigned __int128 uint128_t;
extern double timeEq;
extern double timeCH;

class Key
{
  private :
    uint128_t _hashed;
	  std::array<int8_t, NODE> _word;
  
  public :
    Key(const uint64_t hashed1, const uint64_t hashed2, const std::array<int8_t, NODE> word) : _word(word) {
      _hashed = (((uint128_t)hashed1 >> 64) + hashed2);
      }
    Key(const uint128_t hashed, const std::array<int8_t, NODE> word) : _hashed(hashed), _word(word) {}
    Key(){}// For dense_hash_map
    const int8_t* data() const { return _word.data(); }
    uint128_t hashed() const { return _hashed; }
    int8_t &operator[](uint64_t i){ return _word[i]; }
    const int8_t &operator[](uint64_t i) const { return _word[i]; }
    
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
