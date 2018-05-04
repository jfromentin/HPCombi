#include <array>
#include <iostream>
#include "vector_gpu.cuh"

#define HASH_SIZE 100000
#define NODE 50
#define NB_GEN 6
#define SIZE 100000
#define BLOCK_SIZE 4
#define NB_HASH_FUNC 1

void print_word(std::array<int8_t, NODE> word);

class key
{
  public :
  void creatKey(uint64_t hashed_in, std::array<int8_t, NODE> word_in, uint32_t* d_gen_in, int8_t* d_words_in, Vector_cpugpu<int>* equal_in){
    hashed = hashed_in;
    word = word_in;
    d_gen = d_gen_in;
    d_words = d_words_in;
    equal = equal_in;
  }
  uint64_t hashed;
  std::array<int8_t, NODE> word;
  uint32_t* d_gen;
  int8_t* d_words;
  Vector_cpugpu<int>* equal;
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
