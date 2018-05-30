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


#include "fonctions_gpu.cuh"
#include <math.h>
#include <array>
#include <string>
#include <vector>
#include <sparsehash/dense_hash_set>
#include <chrono>
#include <iotools.hpp>
#include <iomanip>

using namespace std;
using namespace std::chrono;


template <typename T>
class eqTransGPU
{
  private :
    T* d_gen;
    int size;
    Vector_cpugpu<int>* equal;
    int8_t* d_words;
    int8_t nb_gen;
  public :
    eqTransGPU(T* d_gen, int8_t* d_words, const int size, const int8_t nb_gen, Vector_cpugpu<int>& equal) :
        d_gen(d_gen), size(size), nb_gen(nb_gen), equal(&equal), d_words(d_words) {}
    bool operator()(const Key& key1, const Key& key2) const
    {
      return (key1.hashed() == key2.hashed()) && (equal_gpu<T>(key1, key2, d_gen, d_words, *equal, size, nb_gen));
      //~ return key1.hashed() == key2.hashed();
    }

};

class hash_gpu_class
{
  public :
    bool operator()(const Key& keyIn) const
    {
      return keyIn.hashed();
    }
};
template <typename T>
void renner(int size, int8_t nb_gen, uint64_t* gen);

int main(int argc, char* argv[]){
  using namespace std::chrono;
  int size = 10000;
  int8_t nb_gen = 6;

	uint64_t* gen;
  std::string fileName;
  if(argc > 1){
    fileName = std::string(argv[1]);
  }
  else
    fileName = "RenA2";
	fileName = fileName + ".txt";
  readRenner(fileName, &gen, &size, &nb_gen);
  printf("Size : %d, sizeof key : %lu bytes\n", size, sizeof(Key));

  if(size<pow(2,8)){
    printf("Using uint8_t for workSpace\n\n");
    renner<uint8_t>(size, nb_gen, gen);
  }
  else if(size<pow(2,16)){
    printf("Using uint16_t for workSpace\n\n");
    renner<uint16_t>(size, nb_gen, gen);
  }
  else if(size<pow(2,32)){
    printf("Using uint32_t for workSpace\n\n");
    renner<uint32_t>(size, nb_gen, gen);
  }
  else {
    printf("Using uint64_t for workSpace\n\n");
    renner<uint64_t>(size, nb_gen, gen);
  }
  free(gen);
}

template <typename T>
void renner(int size, int8_t nb_gen, uint64_t* gen){
  size_t memory = cudaSetDevice_cpu();
  T* d_gen;
  int8_t* d_words;
  malloc_gen<T>(d_gen, gen, size, nb_gen);
  malloc_words(d_words, NODE);

  Vector_cpugpu<int8_t> todo(pow(2, 12));
  Vector_cpugpu<int8_t> newtodo(pow(2, 12));
  Vector_gpu<T> workSpace(pow(2, 12));
  Vector_cpugpu<uint64_t> hashed(pow(2, 12));
  Vector_cpugpu<int> equal(1);
  equal.push_back(0);
  std::array<int8_t, NODE> empty_word;
  empty_word.fill(-10);
  
  Key empty_key(UINT64_MAX, empty_word);

  hash_gpu_class hashG;
  eqTransGPU<T> equalG(d_gen, d_words, size, nb_gen, equal);
  google::dense_hash_set< Key, hash_gpu_class, eqTransGPU<T> > elems(7000000, hashG, equalG);
  
  elems.set_empty_key(empty_key);

  hashed.resize(1 * NB_HASH_FUNC, 1);
  hash_id_gpu<T>(hashed, workSpace, size);
  std::array<int8_t, NODE> id_word;
  id_word.fill(-1);
  todo.push_back(&(id_word[0]), NODE);
    
  Key id_key(hashed[0], id_word);
  elems.insert(id_key);


  double timeGpu;
  double timeCpu=0;
  auto tstartGpu = high_resolution_clock::now();
  for(int i=0; i<NODE; i++){
    newtodo.clear();
    hpcombi_gpu<T>(todo, workSpace, d_gen, hashed, size, NODE, nb_gen, memory);
    
    for(int j=0; j<todo.size()/NODE*nb_gen; j++){
      
      std::array<int8_t, NODE> newWord;
      for(int k=0; k<NODE; k++)
        newWord[k] = todo.host()[(j/nb_gen)*NODE + k];    
      newWord[i] = j%nb_gen;
      Key new_key(hashed[j * NB_HASH_FUNC], newWord);
      auto tstartCpu = high_resolution_clock::now();
      if(elems.insert(new_key).second){
        newtodo.push_back(&(newWord[0]), NODE);
      }
      auto tfinCpu = high_resolution_clock::now();
      auto tmCpu = duration_cast<duration<double>>(tfinCpu - tstartCpu);
      timeCpu += tmCpu.count();
    }

    todo.swap(&newtodo);  
    auto tfinGpu = high_resolution_clock::now();
    auto tmGpu = duration_cast<duration<double>>(tfinGpu - tstartGpu);
    timeGpu = tmGpu.count();
    cout << i << ", todo = " << todo.size()/NODE << ", elems = " << elems.size()
         << ", #Bucks = " << elems.bucket_count() << ", table size = " 
         << elems.bucket_count()*sizeof(Key)*1e-6 << " Mo, time : " << std::setprecision(3) 
         << (int)timeGpu/3600 << ":" << (int)timeGpu%3600/60 << ":" << ((int)timeGpu%3600)%60 << endl;
    if(todo.size()/NODE == 0)
      break;
  }
  
  auto tfinGpu = high_resolution_clock::now();
  auto tmGpu = duration_cast<duration<double>>(tfinGpu - tstartGpu);
  timeGpu = tmGpu.count();
  printf("Total time : %.3f s, insert : %.3f s\n", timeGpu, timeCpu);
  
  cout << "elems =  " << elems.size() << endl;
  free_gen<T>(d_gen);
  free_words(d_words);

}







































