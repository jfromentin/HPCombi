#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <fstream>

void write_hash(int block_size,  const size_t size, const size_t nb_vect, uint64_t* hashed, double time, std::string file_name);
void write_renner(const int size, const int8_t nb_gen, const int count, double time);
