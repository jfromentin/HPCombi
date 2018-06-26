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

double timeEq=0;
double timeC=0;
double timeH=0;
double timeP=0;

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
    inline bool operator()(const Key& key1, const Key& key2) const
    {
      return (key1.hashed() == key2.hashed()) && (equal_gpu<T>(key1, key2, d_gen, d_words, *equal, size, nb_gen));
      //~ return key1.hashed() == key2.hashed();
    }

};


class eqTransCPU
{
  private :
    uint64_t* gen;
    int size;
  public :
    eqTransCPU(uint64_t* gen, const int size) : gen(gen), size(size) {}
    inline bool operator()(const Key& key1, const Key& key2) const
    {      
      return (key1.hashed() == key2.hashed()) && (equal_cpu(key1, key2, gen, size));
      //~ return key1.hashed() == key2.hashed();
    }

};

class hash_gpu_class
{
  public :
    inline bool operator()(const Key& keyIn) const
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
  printf("Size : %d, sizeof key : %lu bytes, max nodes : %d\n", size, sizeof(Key), NODE);

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
  
  Key empty_key(UINT64_MAX, UINT64_MAX, empty_word);

  hash_gpu_class hashG;
  eqTransGPU<T> equalG(d_gen, d_words, size, nb_gen, equal);
  google::dense_hash_set< Key, hash_gpu_class, eqTransGPU<T> > elems(7000000, hashG, equalG);
  //~ eqTransCPU equalC(gen, size);
  //~ google::dense_hash_set< Key, hash_gpu_class, eqTransCPU > elems(7000000, hashG, equalC);
  
  elems.set_empty_key(empty_key);

  hashed.resize(1 * NB_HASH_FUNC, 1);
  hash_id_gpu<T>(hashed, workSpace, size);
  std::array<int8_t, NODE> id_word;
  id_word.fill(-1);
  todo.push_back(&(id_word[0]), NODE);
    
  Key id_key(hashed[0], hashed[1], id_word);
  elems.insert(id_key);


  double timeTotal=0;
  double timeIns=0;
  auto tstart = high_resolution_clock::now();
  auto tfin = high_resolution_clock::now();
  auto tm = duration_cast<duration<double>>(tfin - tstart);
  
  auto tstartGpu = high_resolution_clock::now();
  int nb_double;
  for(int i=0; i<NODE; i++){
    nb_double = 0;
    newtodo.clear();
    compHash_gpu<T>(todo, workSpace, d_gen, hashed, size, NODE, nb_gen, memory);
    
    for(int j=0; j<todo.size()/NODE*nb_gen; j++){
        std::array<int8_t, NODE> newWord;
        for(int k=0; k<NODE; k++)
          newWord[k] = todo.host()[(j/nb_gen)*NODE + k];    
        newWord[i] = j%nb_gen;
        uint64_t hash1 = hashed[j * NB_HASH_FUNC];
        uint64_t hash2 = hashed[j * NB_HASH_FUNC + 1];
        Key new_key(hash1, hash2, newWord);

      tstart = high_resolution_clock::now();
        if(hash1 == UINT64_MAX && hash2 == UINT64_MAX)
          nb_double++;
        else if(elems.insert(new_key).second){
          newtodo.push_back(&(newWord[0]), NODE);
        }
      tfin = high_resolution_clock::now();
      tm = duration_cast<duration<double>>(tfin - tstart);
      timeIns += tm.count();
    }
  
    auto tfinGpu = high_resolution_clock::now();
    auto tmGpu = duration_cast<duration<double>>(tfinGpu - tstartGpu);
    timeTotal = tmGpu.count();
    cout << i << ", todo = " << newtodo.size()/NODE << ", elems = " << elems.size()
         << ", #Bucks = " << elems.bucket_count() << ", table size = " 
         << elems.bucket_count()*sizeof(Key)*1e-6
         << " Mo, doublons : " << nb_double << "/" << todo.size()/NODE*nb_gen
         << ", " << std::setprecision(2) << (double)nb_double/(todo.size()/NODE*nb_gen)*100 << " %"
         << endl
         << "     Timings : Total = " 
         //~ << (int)timeTotal/3600 << ":" << (int)timeTotal%3600/60 << ":" << ((int)timeTotal%3600)%60
         << std::setprecision(3) << timeTotal
         << endl << "      insert = " 
         //~ << (int)timeIns/3600 << ":" << (int)timeIns%3600/60 << ":" << ((int)timeIns%3600)%60
         << timeIns-timeEq
         << ", " << timeIns/timeTotal*100
         << "%      equal = " 
         //~ << (int)timeEq/3600 << ":" << (int)timeEq%3600/60 << ":" << ((int)timeEq%3600)%60
         << timeEq
         << ", " << timeEq/timeTotal*100
         << "%      comp = " 
         //~ << (int)timeC/3600 << ":" << (int)timeC%3600/60 << ":" << ((int)timeC%3600)%60
         << timeC
         << ", " << timeC/timeTotal*100
         << "%      hash = " 
         //~ << (int)timeH/3600 << ":" << (int)timeH%3600/60 << ":" << ((int)timeH%3600)%60
         << timeH
         << ", " << timeH/timeTotal*100
         << "%      pre-insert = " 
         //~ << (int)timeP/3600 << ":" << (int)timeP%3600/60 << ":" << ((int)timeP%3600)%60
         << timeP
         << ", " << timeP/timeTotal*100
         << "%" << endl;
    if(newtodo.size()/NODE == 0)
      break;
    todo.swap(&newtodo);
  }
  
  auto tfinGpu = high_resolution_clock::now();
  auto tmGpu = duration_cast<duration<double>>(tfinGpu - tstartGpu);
  timeTotal = tmGpu.count();
  printf("Total time : %.3f s\n", timeTotal);
  
  cout << "elems =  " << elems.size() << endl;
  free_gen<T>(d_gen);
  free_words(d_words);

}







































