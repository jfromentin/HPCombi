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

void hash_cpu(const uint32_t* __restrict__ x, uint64_t* hashed, const size_t size, const size_t nb_vect) {
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



int main() {
  //~ int lg = 0;
  using namespace std::chrono;
  size_t size = 50000;
  int nb_vect = 5;
  int block_size = 32;
  int nb_hash = 100;
  int size_start = 16;
  int size_end = 100000;
  int size_nb = 200;
  int vect_start = 1;
  int vect_end = 10000;
  int vect_nb = 200;
  int coefPerThread;
  int kernel_num = 3;
  
  double timeGpu, timeCpu; 
  
  string file_name = to_string(size_end) + to_string(size_nb) + to_string(vect_end) + to_string(vect_nb) + to_string(nb_hash) + to_string(kernel_num);
  
  for(nb_vect = vect_start; nb_vect<vect_end; nb_vect = ceil(nb_vect*pow(vect_end/vect_start, 1./vect_nb)) ){
    printf("nb_vect : %d\n", nb_vect);

    for(size = size_start; size<size_end; size = ceil(size *pow(size_end/size_start, 1./size_nb))){
      printf("size : %d\n", size);
      uint32_t* gen = (uint32_t*)malloc(size * nb_vect * sizeof(uint32_t));    
      uint64_t* hashed = (uint64_t*)malloc(nb_vect * sizeof(uint64_t));     
       
      for(int i=0; i<nb_vect; i++)
        for(int j=0; j<size; j++)
          gen[i*size + j] = (uint32_t)(j+i);  
  
      for(block_size = 4; block_size<=5; block_size*=2){
        // GPU ############################
        //~ auto tstartGpu = high_resolution_clock::now();
          //~ for(int j=0; j<nb_hash; j++)
            //~ hash_gpu(gen, block_size, hashed, size, nb_vect);
        //~ auto tfinGpu = high_resolution_clock::now();
        //~ auto tmGpu = duration_cast<duration<double>>(tfinGpu - tstartGpu);
        //~ timeGpu = tmGpu.count()/nb_hash*1e6;
        //~ for(int i=0; i<nb_vect; i++)
          //~ printf("Hash GPU : %lu, index : %lu\n", hashed[i], hashed[i]%100000);
        // CPU ############################
        auto tstartCpu = high_resolution_clock::now();
          for(int j=0; j<nb_hash; j++)
            hash_cpu(gen, hashed, size, nb_vect);
        auto tfinCpu = high_resolution_clock::now();
        auto tmCpu = duration_cast<duration<double>>(tfinCpu - tstartCpu);
        timeCpu = tmCpu.count()/nb_hash*1e6;
        for(int i=0; i<nb_vect; i++)
          printf("Hash CPU : %lu, index : %lu\n", hashed[i], hashed[i]%100000);
         
        if(kernel_num == 1)
          coefPerThread = (size+block_size-1) / block_size;
        else if(kernel_num == 2)
          coefPerThread = (size+32-1) / 32;
        //~ printf("coefPerThread : %d, nb vect : %d\n", coefPerThread, nb_vect);
        //~ printf("Speedup : %f\n", timeCpu/timeGpu);
        //~ printf("Block size : %d, size : %lu, time : %.3f us, time/vect : %.3f us\n", block_size, size, timeGpu, timeGpu/nb_vect);
        //~ printf("Block size : , size : %lu, time : %.3f us, time/vect : %.3f us\n", size, timeCpu, timeCpu/nb_vect);
        
        //~ write_hash(block_size, size, nb_vect, hashed, timeGPU, file_name);
        write_hash(1, size, nb_vect, hashed, timeCpu, file_name);
      }
      //~ printf("\n");
  
      free(gen);
      free(hashed);
    }
  }
  printf("end\n");

}
