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
#include "perm_generic.hpp"
#include <math.h>
#include <array>
#include <string>
#include <vector>
#include <sparsehash/dense_hash_set>
#include <chrono>
#include <iotools.hpp>
#include <iomanip>

using namespace std;
using namespace chrono;
using namespace HPCombi;

double timeEq=0;
double timeCH=0;

void printWord(const array<int8_t, NODE>& word){
  for(int i=0; i<NODE; i++){
    if(word[i]==-1)
      break;
    cout << static_cast<int>(word[i]) << "|";
  }
  cout << endl;
  }

template <typename T, size_t size>
uint128_t hash_cpu(const VectGeneric<size, T>& trans){
  uint64_t prime = 0x9e3779b97f4a7bb9;
  uint128_t result = trans[0]*prime;
  for(int i=1; i<size; i++){
    result += trans[i];
    result *= prime;
  }
  return result;
}

template <typename T, size_t size>
bool equal_cpu(const Key& key1, const Key& key2, const vector< VectGeneric<size, T> >& gens){
  auto tstartCpu = high_resolution_clock::now();
  
  VectGeneric<size, T> tmp1(0, size);
  VectGeneric<size, T> tmp2(0, size);
  for(int j=0; j<NODE; j++){
    if(key1[j]==-1)
      break;
    tmp1 = gens[ key1[j] ].permuted(tmp1);
  }
  for(int j=0; j<NODE; j++){
    if(key2[j]==-1)
      break;
    tmp2 = gens[ key2[j] ].permuted(tmp2);
  }
  
  auto tfinCpu = high_resolution_clock::now();
  auto tmCpu = duration_cast<duration<double>>(tfinCpu - tstartCpu);
  timeEq += tmCpu.count();
  return tmp1 == tmp2;
}


template <typename T, size_t size>
struct eqTransCPU
{
  private :
    vector< VectGeneric<size, T>> gens;
  public :
    eqTransCPU(const vector< VectGeneric<size, T> >& gens) : gens(gens){}
    inline bool operator()(const Key& key1, const Key& key2) const
    {
      return (key1.hashed() == key2.hashed()) && (equal_cpu<T>(key1, key2, gens));
      //~ return key1.hashed() == key2.hashed();
    }

};

class hash_cpu_class
{
  public :
    inline bool operator()(const Key& keyIn) const
    {
      return keyIn.hashed();
    }
};
template <typename T, size_t size>
void renner(int8_t nb_gen, uint64_t* gen);
template void renner<int8_t, 2>(int8_t nb_gen, uint64_t* gen);
template void renner<int8_t, 6>(int8_t nb_gen, uint64_t* gen);
template void renner<int8_t, 24>(int8_t nb_gen, uint64_t* gen);
template void renner<int8_t, 120>(int8_t nb_gen, uint64_t* gen);
template void renner<int16_t, 720>(int8_t nb_gen, uint64_t* gen);
template void renner<int16_t, 5040>(int8_t nb_gen, uint64_t* gen);

int main(int argc, char* argv[]){
  using namespace chrono;
  int size = 10000;
  int8_t nb_gen = 6;

	uint64_t* gen;
  string fileName;
  if(argc > 1)
    fileName = string(argv[1]);
  else
    fileName = "RenA2";
	fileName = fileName + ".txt";
  readRenner(fileName, &gen, &size, &nb_gen);
  printf("Size : %d, sizeof key : %lu bytes, max nodes : %d\n", size, sizeof(Key), NODE);

  if(size==2){
    printf("Using uint8_t for workSpace\n\n");
    renner<uint8_t, 2>(nb_gen, gen);
  }
  if(size==6){
    printf("Using uint8_t for workSpace\n\n");
    renner<uint8_t, 6>(nb_gen, gen);
  }
  if(size==24){
    printf("Using uint8_t for workSpace\n\n");
    renner<uint8_t, 24>(nb_gen, gen);
  }
  if(size==120){
    printf("Using uint8_t for workSpace\n\n");
    renner<uint8_t, 120>(nb_gen, gen);
  }
  else if(size==720){
    printf("Using uint16_t for workSpace\n\n");
    renner<uint16_t, 720>(nb_gen, gen);
  }
  else if(size==5040){
    printf("Using uint16_t for workSpace\n\n");
    renner<uint16_t, 5040>(nb_gen, gen);
  }
  free(gen);
}

template <typename T, size_t size>
void renner(int8_t nb_gen, uint64_t* gen){

	vector< VectGeneric<size, T> > gens;
  for(int i=0; i<nb_gen; i++){
    VectGeneric<size, T> tmp;
    for(int j=0; j<size; j++)
      tmp[j] = gen[j + i*size];
    gens.push_back(tmp);    
  }


  //~ vector< array<int8_t, NODE> > todo(pow(2, 12));
  //~ vector< array<int8_t, NODE> > newtodo(pow(2, 12));
  vector< array<int8_t, NODE> > todo, newtodo;

  array<int8_t, NODE> empty_word;
  empty_word.fill(-10);  
  Key empty_key(UINT64_MAX, UINT64_MAX, empty_word);

  hash_cpu_class hashG;
  eqTransCPU<T, size> equalG(gens);
  google::dense_hash_set< Key, hash_cpu_class, eqTransCPU<T, size> > elems(7000000, hashG, equalG);
  
  elems.set_empty_key(empty_key);

  VectGeneric<size, T> id(0, size);
  uint128_t hashed = hash_cpu(id);
  array<int8_t, NODE> id_word;
  id_word.fill(-1);
  todo.push_back(id_word);
    
  Key id_key(hashed, id_word);
  elems.insert(id_key);


  double timeTotal=0;
  double timeIns=0;
  double timeCon=0;  
  auto tstart = high_resolution_clock::now();
  auto tfin = high_resolution_clock::now();
  auto tm = duration_cast<duration<double>>(tfin - tstart);
  
  auto tstartGpu = high_resolution_clock::now();
  for(int i=0; i<NODE; i++){
    newtodo.clear();
    for(int i=0; i<todo.size(); i++){
      VectGeneric<size, T> tmp(0, size);
      auto word = todo[i];
      int j=0;
      for(j=0; j<NODE; j++){
        if(word[j]==-1)
          break;
        tmp = gens[ word[j] ].permuted(tmp);
      }
      
      for(int k=0; k<gens.size(); k++){
        auto newWord = word;
        newWord[j] = k;
        hashed = hash_cpu<T, size>(gens[k].permuted(tmp));
        Key new_key(hashed, newWord);
        
        tstart = high_resolution_clock::now();
        if(elems.insert(new_key).second){
          newtodo.push_back(newWord);
        }
        tfin = high_resolution_clock::now();
        tm = duration_cast<duration<double>>(tfin - tstart);
        timeIns += tm.count();
      }
    }

    swap(todo, newtodo); 
    auto tfinGpu = high_resolution_clock::now();
    auto tmGpu = duration_cast<duration<double>>(tfinGpu - tstartGpu);
    timeTotal = tmGpu.count();
    cout << i << ", todo = " << todo.size() << ", elems = " << elems.size()
         << ", #Bucks = " << elems.bucket_count() << ", table size = " 
         << elems.bucket_count()*sizeof(Key)*1e-6 
         << " Mo" << endl
         << "     Timings : Total = " 
         << (int)timeTotal/3600 << ":" << (int)timeTotal%3600/60 << ":" << ((int)timeTotal%3600)%60
         << endl << "      insert = " 
         << (int)timeIns/3600 << ":" << (int)timeIns%3600/60 << ":" << ((int)timeIns%3600)%60
         << ", " << setprecision(3) << timeIns/timeTotal*100
         << "%      equal = " 
         << (int)timeEq/3600 << ":" << (int)timeEq%3600/60 << ":" << ((int)timeEq%3600)%60
         << ", " << setprecision(3) << timeEq/timeTotal*100
         //~ << "%" << endl << "      constr = " 
         //~ << (int)timeCon/3600 << ":" << (int)timeCon%3600/60 << ":" << ((int)timeCon%3600)%60
         //~ << ", " << setprecision(3) << timeCon/timeTotal*100
         //~ << "%      compHash = " 
         //~ << (int)timeCH/3600 << ":" << (int)timeCH%3600/60 << ":" << ((int)timeCH%3600)%60
         //~ << ", " << setprecision(3) << timeCH/timeTotal*100
         << "%" << endl;
    if(todo.size() == 0)
      break;
  }
  
  auto tfinGpu = high_resolution_clock::now();
  auto tmGpu = duration_cast<duration<double>>(tfinGpu - tstartGpu);
  timeTotal = tmGpu.count();
  printf("Total time : %.3f s\n", timeTotal);
  
  cout << "elems =  " << elems.size() << endl;

}







































