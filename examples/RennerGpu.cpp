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

#ifdef HPCOMBI_HAVE_DENSEHASHMAP
google::dense_hash_map<PTransf16, std::pair<PTransf16, int>,
                       hash<PTransf16>, equal_to<PTransf16>> elems;
#else
unordered_map<PTransf16, std::pair<PTransf16, int>> elems;
#endif

void hash_cpu(const uint32_t* __restrict__ x, uint64_t* hashed, const int size, const int nb_vect) {
  const uint64_t prime = 0x9e3779b97f4a7bb9;
  uint64_t tmp;
  __int128 v0;
  __int128 v1;
  for(int k=0; k<nb_vect; k++){
    tmp=x[k*size];
    for(int i=1; i<size; i++){
      v0 = (__int128)x[i + k*size];
      v1 = (__int128)tmp;
      tmp += ((v1 * prime + v0) * prime) >> 64;
    }
    hashed[k] = tmp;
  }
}



//~ int main() {
  //~ using namespace std::chrono;
  //~ int size = 50000;
  //~ int nb_vect = 5;
  //~ int block_size = 32;
  //~ int nb_hash = 100;
  //~ int size_start = 2000;
  //~ int size_end = 250000;
  //~ int size_nb = 5;
  //~ int vect_start = 200;
  //~ int vect_end = 300;
  //~ int vect_nb = 2;
  //~ int coefPerThread;
  //~ int kernel_num = 3;
  
  //~ double timeGpu, timeCpu; 
  
  //~ string file_name = to_string(size_end) + to_string(size_nb) + to_string(vect_end) + to_string(vect_nb) + to_string(nb_hash) + to_string(kernel_num);
  //~ printf("kernel_num : %d\n", kernel_num);
  //~ for(nb_vect = vect_start; nb_vect<vect_end; nb_vect = ceil(nb_vect*pow(vect_end/vect_start, 1./vect_nb)) ){
    //~ printf("nb_vect : %d\n", nb_vect);

    //~ for(size = size_start; size<size_end; size = ceil(size *pow(size_end/size_start, 1./size_nb))){
      //~ printf("size : %d\n", size);
      //~ uint32_t* gen = (uint32_t*)malloc(size * nb_vect * sizeof(uint32_t));    
      //~ uint64_t* hashed = (uint64_t*)malloc(nb_vect * sizeof(uint64_t));     
       
      //~ for(int i=0; i<nb_vect; i++)
        //~ for(int j=0; j<size; j++)
          //~ gen[i*size + j] = (uint32_t)(j+i);
  
      //~ for(block_size = 4; block_size<=5; block_size*=2){
        //~ // GPU ############################
        //~ auto tstartGpu = high_resolution_clock::now();
          //~ for(int j=0; j<nb_hash; j++)
            //~ hash_gpu(gen, block_size, hashed, size, nb_vect, kernel_num);
        //~ auto tfinGpu = high_resolution_clock::now();
        //~ auto tmGpu = duration_cast<duration<double>>(tfinGpu - tstartGpu);
        //~ timeGpu = tmGpu.count()/nb_hash*1e6;
        //~ for(int i=0; i<nb_vect; i++)
          //~ printf("Hash GPU : %lu, index : %lu\n", hashed[i], hashed[i]%100000);
        //~ // CPU ############################
        //~ auto tstartCpu = high_resolution_clock::now();
          //~ for(int j=0; j<nb_hash; j++)
            //~ hash_cpu(gen, hashed, size, nb_vect);
        //~ auto tfinCpu = high_resolution_clock::now();
        //~ auto tmCpu = duration_cast<duration<double>>(tfinCpu - tstartCpu);
        //~ timeCpu = tmCpu.count()/nb_hash*1e6;
        //~ for(int i=0; i<nb_vect; i++)
          //~ printf("Hash CPU : %lu, index : %lu\n", hashed[i], hashed[i]%100000);
         
        //~ if(kernel_num == 1)
          //~ coefPerThread = (size+block_size-1) / block_size;
        //~ else if(kernel_num > 1)
          //~ coefPerThread = (size+32-1) / 32;
        //~ printf("coefPerThread : %d, nb vect : %d\n", coefPerThread, nb_vect);
        //~ printf("Speedup : %f\n", timeCpu/timeGpu);
        //~ printf("Block size : %d, size : %lu, time : %.3f us, time/vect : %.3f us\n", block_size, size, timeGpu, timeGpu/nb_vect);
        //~ printf("Block size : , size : %lu, time : %.3f us, time/vect : %.3f us\n", size, timeCpu, timeCpu/nb_vect);
        
        //~ write_hash(block_size, size, nb_vect, hashed, timeGpu, file_name);
        //~ write_hash(1, size, nb_vect, hashed, timeCpu, file_name);
      //~ }
      //~ printf("\n");
  
      //~ free(gen);
      //~ free(hashed);
    //~ }
  //~ }
  //~ printf("end\n");

//~ }



#define HASH_SIZE 10000
#define NODE 50

void print_word(std::array<int, NODE> word);


struct eqstr
{
  bool operator()(const std::pair<uint64_t, std::array<int, NODE>> pair1, const std::pair<uint64_t, std::array<int, NODE>> pair2) const
  {
    return pair1.first == pair2.first;
    //~ return equal_gpu(&(pair1.second[0]), &(pair2.second[0]), d_gen, block_size, size, size_word, nb_gen);
  }
};

struct hash_gpu_class
{
  bool operator()(const std::pair<uint64_t, std::array<int, NODE>> pair) const
  {
    return pair.first%HASH_SIZE;
  }
};


int main() {
  using namespace std::chrono;
  const int size = 16;
  int block_size = 4;
  const int nb_gen = 4;
  //~ const int node = 50;

  google::dense_hash_map< std::pair< uint64_t, std::array<int, NODE> >, std::array<int, NODE>, hash_gpu_class, eqstr> elems;

  uint32_t* gen = (uint32_t*)malloc(size*nb_gen * sizeof(uint32_t));
  for(int i=0; i<size*nb_gen; i++){
    gen[i] = i%size;
  }
  gen[9] = 10;
  gen[10] = 9;
  gen[size + 10] = 11;
  gen[size + 11] = 10;
  gen[2*size + 3] = 4;
  gen[2*size + 4] = 3;
  gen[3*size + 2] = 3;
  gen[3*size + 3] = 2;
  //~ gen[4*size + 1] = 2;
  //~ gen[4*size + 2] = 1;
  uint32_t* d_gen;
  malloc_gen(&d_gen, gen, size, nb_gen);



  vector< std::array<int, NODE> > todo, newtodo;
  std::array<int, NODE> id_word;
  id_word.fill(-1);
  todo.push_back(id_word);

  elems.set_empty_key(std::make_pair (-1, id_word));

  uint64_t hashedId;
  hash_id_gpu(&hashedId, block_size, size);  
  elems.insert({ {hashedId, id_word}, id_word});
  
  
  for(int i=0; i<NODE; i++){
    newtodo.clear();
    uint64_t* hashed = (uint64_t*)malloc(todo.size()*nb_gen * sizeof(uint64_t)); // Todo use of vector
    hpcombi_gpu(&(todo[0][0]), d_gen, hashed, block_size, size, NODE, todo.size(), nb_gen);
    
    for(int j=0; j<todo.size()*nb_gen; j++){
      std::array<int, NODE> newWord = todo[j/nb_gen];        
      newWord[i] = gen[j%nb_gen];
      //~ print_word(newWord);
      if(elems.insert({ {hashed[j], newWord}, newWord}).second){
        newtodo.push_back(newWord);
      //~ cout << "new   " << j << "  " << hashed[j]%HASH_SIZE << endl;
      }
      else{
        //~ cout << "old  " << j << "  " << hashed[j]%HASH_SIZE << endl;
        //~ print_word(elems[hashed[j]]);
      }
    }

    free(hashed);
    std::swap(todo, newtodo);
    cout << i << ", todo = " << todo.size() << ", elems = " << elems.size()
         << ", #Bucks = " << elems.bucket_count() << endl;
         
         
    //~ for(int i=0; i<todo.size()*NODE; i++){
      //~ printf("%d|", *(&(todo[0][0])+i));
      //~ if(i%NODE==NODE-1)
      //~ printf("\n");
    //~ }
    //~ printf("\n");
    
    if(todo.size() == 0)
      break;
  }
  cout << "elems =  " << elems.size() << endl;
  free(gen);
  free_gen(&d_gen);
}





void print_word(std::array<int, NODE> word){
  for(int i=0; i<NODE; i++)
    printf("%d|", word[i]);
  printf("\n");
  
  
}





































