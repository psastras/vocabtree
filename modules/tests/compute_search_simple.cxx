#include <config.hpp>
#include "tests_config.hpp"

#include <utils/filesystem.hpp>
#include <utils/numerics.hpp>
#include <utils/dataset.hpp>
#include <utils/vision.hpp>
#include <utils/logger.hpp>
#include <search/bag_of_words/bag_of_words.hpp>
#include <search/inverted_index/inverted_index.hpp>

#include <iostream>

_INITIALIZE_EASYLOGGINGPP

int main(int argc, char *argv[]) {
#if ENABLE_MULTITHREADING && ENABLE_MPI
	MPI::Init(argc, argv);
	int rank = MPI::COMM_WORLD.Get_rank();
	if(rank == 0) {
#endif
	const uint32_t num_clusters = 512;

	SimpleDataset simple_dataset(s_oxfordmini_data_dir, s_oxfordmini_database_location);
	LINFO << simple_dataset;

	std::stringstream index_output_file;
	index_output_file << simple_dataset.location() << "/index/" << num_clusters << ".index";
	InvertedIndex ii(index_output_file.str());

	for(uint32_t i=0; i<3; i++) {
		PTR_LIB::shared_ptr<InvertedIndex::MatchResults> matches = 
		std::static_pointer_cast<InvertedIndex::MatchResults>(ii.search(simple_dataset, PTR_LIB::shared_ptr<SearchParamsBase>(), simple_dataset.image(i) ));	
		LINFO << "Query " << i << ": " << *matches;
	}
#if ENABLE_MULTITHREADING && ENABLE_MPI
	}
	MPI::Finalize();
#endif
	return 0;
}