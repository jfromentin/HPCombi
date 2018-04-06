#ifndef HPCOMBI_IOTOOLS_HPP_INCLUDED
#include <iotools.hpp>

void write_hash(int block_size,  const size_t size, const size_t nb_vect, uint64_t* hashed, double time, std::string file_name){
	file_name = "examples/outData/" + file_name + ".data";
	//~ file_name = "examples/outData/test.data";

	std::ofstream myfile;
	myfile.open (file_name, std::ios::app);
	myfile << hashed[0] << ";" << block_size << ";" << size << ";" << nb_vect << ";" << time << std::endl;
	myfile.close();	
}
	
#endif  // HPCOMBI_IOTOOLS_HPP_INCLUDED
