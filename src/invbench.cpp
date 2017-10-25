#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <benchmark/benchmark.h>

#include "perm16.hpp"

using namespace IVMPG;
using namespace std;
const int rep = 100;

class Perm16Fixture : public benchmark::Fixture {
public:
  virtual void SetUp(const benchmark::State& state) {
    sample = Perm16::random();
  }
  virtual void TearDown(const benchmark::State& state) {}
  Perm16 sample;
};

BENCHMARK_DEFINE_F(Perm16Fixture, BM_Perm16invert_ref) (benchmark::State& state) {
  for (auto _ : state) {
    Perm16 p = sample;
    for (int i = 0; i < state.range(0); i++) p = p.inverse_ref();
    benchmark::DoNotOptimize(p);
  }
}
BENCHMARK_REGISTER_F(Perm16Fixture, BM_Perm16invert_ref)->RangeMultiplier(10)->Range(1, 1000);

BENCHMARK_DEFINE_F(Perm16Fixture, BM_Perm16invert_sort) (benchmark::State& state) {
  for (auto _ : state) {
    Perm16 p = sample;
    for (int i = 0; i < state.range(0); i++) p = p.inverse_sort();
    benchmark::DoNotOptimize(p);
  }
}
BENCHMARK_REGISTER_F(Perm16Fixture, BM_Perm16invert_sort)->RangeMultiplier(10)->Range(1, 1000);

BENCHMARK_DEFINE_F(Perm16Fixture, BM_Perm16invert_find) (benchmark::State& state) {
  for (auto _ : state) {
    Perm16 p = sample;
    for (int i = 0; i < state.range(0); i++) p = p.inverse_find();
    benchmark::DoNotOptimize(p);
  }
}
BENCHMARK_REGISTER_F(Perm16Fixture, BM_Perm16invert_find)->RangeMultiplier(10)->Range(1, 1000);

BENCHMARK_DEFINE_F(Perm16Fixture, BM_Perm16invert_pow) (benchmark::State& state) {
  for (auto _ : state) {
    Perm16 p = sample;
    for (int i = 0; i < state.range(0); i++) p = p.inverse_pow();
    benchmark::DoNotOptimize(p);
  }
}
BENCHMARK_REGISTER_F(Perm16Fixture, BM_Perm16invert_pow)->RangeMultiplier(10)->Range(1, 1000);



using Fun = std::function<Perm16(Perm16)>;
// Methods to be tested
const std::array<std::pair<string, Fun>, 4> invproc = {
  std::make_pair("ref", &Perm16::inverse_ref),
  std::make_pair("sort", &Perm16::inverse_sort),
  std::make_pair("find", &Perm16::inverse_find),
  std::make_pair("pow", &Perm16::inverse_pow)
};


int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);

  for (auto namefun : invproc) {
    auto bench = [namefun](benchmark::State& state, Perm16 sample) {
      for (auto _ : state) {
        Perm16 p = sample;
        for (int i = 0; i < state.range(0); i++) p = namefun.second(p);
        benchmark::DoNotOptimize(p);
      }
    };
    benchmark::RegisterBenchmark(namefun.first.c_str(), bench, Perm16::random()
                                 )->RangeMultiplier(10)->Range(1,1000);
  }

  benchmark::RunSpecifiedBenchmarks();
}
