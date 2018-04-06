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
  
  int kernel_num = 1;
  
  string file_name = to_string(size_end) + to_string(size_nb) + to_string(vect_end) + to_string(vect_nb) + to_string(nb_hash) + to_string(kernel_num);
  
  for(nb_vect = vect_start; nb_vect<vect_end; nb_vect = ceil(nb_vect*pow(vect_end/vect_start, 1./vect_nb)) ){
    printf("nb_vect : %d\n", nb_vect);

    for(size = size_start; size<size_end; size = ceil(size *pow(size_end/size_start, 1./size_nb))){
      //~ printf("size : %d\n", size);
      uint32_t* gen = (uint32_t*)malloc(size * nb_vect * sizeof(uint32_t));    
      uint64_t* hashed = (uint64_t*)malloc(nb_vect * sizeof(uint64_t));     
       
      for(int i=0; i<nb_vect; i++)
        for(int j=0; j<size; j++)
          gen[i*size + j] = (uint32_t)(j+i);  
  
      for(block_size = 32; block_size<=256; block_size*=2){
        for(int k=0; k<1; k++){
          auto tstart = high_resolution_clock::now();
            for(int j=0; j<nb_hash; j++)
              hash_gpu(gen, block_size, hashed, size, nb_vect);
          auto tfin = high_resolution_clock::now();
          auto tm = duration_cast<duration<double>>(tfin - tstart);
          double time = tm.count()/nb_hash*1e6;
          int coefPerThread = (size+block_size-1) / block_size;
          //~ for(int i=0; i<nb_vect; i++)
            //~ printf("Hash : %lu, index : %lu\n", hashed[i], hashed[i]%100000);
          //~ printf("coefPerThread : %d, nb vect : %d\n", coefPerThread, nb_vect);
          //~ printf("Block size : %d, size : %lu, time : %.3f us, time/vect : %.3f us\n", block_size, size, time, time/nb_vect);
          
          write_hash(block_size, size, nb_vect, hashed, time, file_name);
        }
      }
      //~ printf("\n");
  
      free(gen);
      free(hashed);
    }
  }

}
