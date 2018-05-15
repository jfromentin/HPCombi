#ifndef HPCOMBI_IOTOOLS_HPP_INCLUDED
#include <iotools.hpp>
#include <sstream>
#include <string>

void write_hash(int block_size,  const int size, const int nb_vect, uint64_t* hashed, double time, std::string file_name){
	file_name = "examples/outData/" + file_name + ".data";
	//~ file_name = "examples/outData/test.data";

	std::ofstream myfile;
	myfile.open (file_name, std::ios::app);
	myfile << hashed[0] << ";" << block_size << ";" << size << ";" << nb_vect << ";" << time << std::endl;
	myfile.close();	
}

void write_renner(const int size, const int8_t nb_gen, const int count, double time){
	std::string file_name = "examples/outData/" + std::to_string(nb_gen) + "_" + std::to_string(count) + ".data";

	std::ofstream myfile;
	myfile.open (file_name, std::ios::app);
	myfile << size << ";" << time << std::endl;
	myfile.close();	
}


void readRenner(std::string file_name, uint32_t** gen, int* size, int8_t* nb_gen){
	file_name = "../data/" + file_name;
	std::ifstream infile(file_name);
	std::string line;
	std::getline(infile, line);
	std::getline(infile, line);
	*nb_gen = std::stoi(line);
	if(*nb_gen <= 0){
		printf("nb_gen is not valid : %d", *nb_gen);
		exit(1);
	}
	std::getline(infile, line);
	*size = std::stoi(line);	
	if(*size <= 0){
		printf("Size is not valid : %d", *size);
		exit(1);
	}
	
	(*gen) = (uint32_t*)malloc((*size)*(*nb_gen) * sizeof(uint32_t));
	std::cout << "Reading file " << file_name << ", size : " << *size << ", nb_gen : " << (int)*nb_gen << std::endl;;
		
	for(int i=0; i<*nb_gen; i++){
		std::getline(infile, line);
	    std::istringstream iss(line);
	    for(int j=0; j<*size; j++){
			iss >> (*gen)[j + i*(*size)];
		}
	}
}

#endif  // HPCOMBI_IOTOOLS_HPP_INCLUDED
