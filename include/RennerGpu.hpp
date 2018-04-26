#include <array>
#include <iostream>
#include "vector_gpu.cuh"

#define HASH_SIZE 100000
#define NODE 50
#define NB_GEN 6
#define SIZE 16
#define BLOCK_SIZE 4
#define NB_HASH_FUNC 4

void print_word(std::array<int8_t, NODE> word);

struct key
{  
  uint64_t hashed;
  std::array<int8_t, NODE> word;
  Vector_gpu<uint32_t>* d_x;
  Vector_cpugpu<int>* equal;
  uint32_t* d_gen;
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