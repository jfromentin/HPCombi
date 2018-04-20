//****************************************************************************//
//       Copyright (C) 2016 Florent Hivert <Florent.Hivert@lri.fr>,           //
//                                                                            //
//  Distributed under the terms of the GNU General Public License (GPL)       //
//                                                                            //
//    This code is distributed in the hope that it will be useful,            //
//    but WITHOUT ANY WARRANTY; without even the implied warranty of          //
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU       //
//   General Public License for more details.                                 //
//                                                                            //
//  The full text of the GPL is available at:                                 //
//                                                                            //
//                  http://www.gnu.org/licenses/                              //
//****************************************************************************//

#include "perm16.hpp"
#include <array>
#include <cassert>
#include <cstdint>
#include <functional>  // less<>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#ifdef HPCOMBI_HAVE_DENSEHASHSET
#include <sparsehash/dense_hash_map>
#else
#include <unordered_map>
#endif
#include <x86intrin.h>
#include <chrono>

#include <iotools.hpp>


using namespace std;
using namespace HPCombi;
using namespace std::chrono;

struct eqstr
{  
  uint32_t* d_gen;
  int block_size=BLOCK_SIZE;
  int size=SIZE;
  int size_word=NODE;
  int nb_gen=NB_GEN;
  bool operator()(const key key1, const key key2) const
  {
    //~ return (key1.hashed == key2.hashed) && (equal_gpu(&key1, &key2, block_size, size, size_word, nb_gen));
    return key1.hashed == key2.hashed;
  }
};

struct hash_gpu_class
{
  bool operator()(const key key1) const
  {
    return key1.hashed;
  }
};


int main() {
  using namespace std::chrono;
  const int size = SIZE;
  int block_size = BLOCK_SIZE;
  const int nb_gen = NB_GEN;

  uint32_t* gen = (uint32_t*)malloc(size*nb_gen * sizeof(uint32_t));
  for(int i=0; i<size*nb_gen; i++){
    gen[i] = i%size;
  }
  gen[5] = 6;
  gen[6] = 5;
  gen[9] = 10;
  gen[10] = 9;
  gen[size + 4] = 5;
  gen[size + 5] = 4;
  gen[size + 10] = 11;
  gen[size + 11] = 10;
  gen[2*size + 3] = 4;
  gen[2*size + 4] = 3;
  gen[2*size + 11] = 12;
  gen[2*size + 12] = 11;
  gen[3*size + 2] = 3;
  gen[3*size + 3] = 2;
  gen[3*size + 12] = 13;
  gen[3*size + 13] = 12;
  gen[4*size + 1] = 2;
  gen[4*size + 2] = 1;
  gen[4*size + 13] = 14;
  gen[4*size + 14] = 13;
  gen[5*size + 0] = 1;
  gen[5*size + 1] = 0;
  gen[5*size + 14] = 15;
  gen[5*size + 15] = 14;
  //~ gen[5*size + 7] = 8;
  //~ gen[5*size + 8] = 7;
  //~ gen[5*size + 6] = 7;
  //~ gen[5*size + 7] = 6;
  //~ gen[5*size + 8] = 9;
  //~ gen[5*size + 9] = 8;
  //~ gen[6*size + 6] = 7;
  //~ gen[6*size + 7] = 6;
  //~ gen[6*size + 8] = 9;
  //~ gen[6*size + 9] = 8;
const PTransf16 s0  {0, 1, 2, 3, 4, 5, 6, 8, 7, 9,10,11,12,13,14,15};
const PTransf16 s1e {0, 1, 2, 3, 4, 5, 7, 6, 9, 8,10,11,12,13,14,15};
const PTransf16 s1f {0, 1, 2, 3, 4, 5, 8, 9, 6, 7,10,11,12,13,14,15};
const PTransf16 s2  {0, 1, 2, 3, 4, 6, 5, 7, 8,10, 9,11,12,13,14,15};
const PTransf16 s3  {0, 1, 2, 3, 5, 4, 6, 7, 8, 9,11,10,12,13,14,15};
const PTransf16 s4  {0, 1, 2, 4, 3, 5, 6, 7, 8, 9,10,12,11,13,14,15};
const PTransf16 s5  {0, 1, 3, 2, 4, 5, 6, 7, 8, 9,10,11,13,12,14,15};
const PTransf16 s6  {0, 2, 1, 3, 4, 5, 6, 7, 8, 9,10,11,12,14,13,15};
const PTransf16 s7  {1, 0, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,15,14};

  uint32_t* d_gen;
  malloc_gen(&d_gen, gen, size, nb_gen);

  google::dense_hash_map< key, std::array<int, NODE>, hash_gpu_class, eqstr> elems(5000);

  Vector_cpugpu<int> todo, newtodo;
  Vector_gpu<uint32_t> d_x(8192);
  Vector_gpu<uint32_t> d_y(8192);
  Vector_cpugpu<uint64_t> hashed(1024);
  std::array<int, NODE> empty_word;
  empty_word.fill(-10);
  
  key empty_key;
  empty_key.hashed = -1;
  empty_key.word = empty_word;
  //~ empty_key.word.push_back(&(empty_word[0]), NODE);
  empty_key.d_gen = d_gen;
  
  elems.set_empty_key(empty_key);

  uint64_t hashedId;
  hash_id_gpu(&hashedId, block_size, size);
  std::array<int, NODE> id_word;
  id_word.fill(-1);
  todo.push_back(&(id_word[0]), NODE);
    
  key id_key;
  id_key.hashed = hashedId;
  id_key.word = id_word;
  //~ id_key.word.push_back(&(id_word[0]), NODE);
  id_key.d_gen = d_gen;
  elems.insert({ id_key, id_word});


double timeGpu;
auto tstartGpu = high_resolution_clock::now();

  for(int i=0; i<NODE; i++){
    newtodo.clear(); 
    hashed.resize(todo.size/NODE*nb_gen, 1);
    hpcombi_gpu(&todo, &d_x, &d_y, d_gen, &hashed, block_size, size, NODE, nb_gen);
    
    for(int j=0; j<todo.size/NODE*nb_gen; j++){       
      std::array<int, NODE> newWord;
      for(int k=0; k<NODE; k++)
        newWord[k] = todo.host[(j/nb_gen)*NODE + k];    
      newWord[i] = j%nb_gen;
      //~ print_word(newWord);
      key new_key;
      new_key.hashed = hashed.host[j];
      new_key.word = newWord;
      //~ new_key.word.push_back(&(newWord[0]), NODE);
      new_key.d_gen = d_gen;
      if(elems.insert({ new_key, newWord}).second){
        newtodo.push_back(&(newWord[0]), NODE);
        //~ print_word(newWord);
      }
      else{
      }
    }

    todo.swap(&newtodo);
    cout << i << ", todo = " << todo.size/NODE << ", elems = " << elems.size()
         << ", #Bucks = " << elems.bucket_count() << endl;
    if(todo.size/NODE == 0)
      break;
  }
  
  auto tfinGpu = high_resolution_clock::now();
  auto tmGpu = duration_cast<duration<double>>(tfinGpu - tstartGpu);
  timeGpu = tmGpu.count()*1e3;
  printf("Time GPU : %.3fms\n", timeGpu);
  
  cout << "elems =  " << elems.size() << endl;
  free(gen);
  free_gen(&d_gen);
}





void print_word(std::array<int, NODE> word){
  for(int i=0; i<NODE; i++)
    if(word[i]>-1)
      printf("%d|", word[i]);
  printf("\n");
  
  
}





































