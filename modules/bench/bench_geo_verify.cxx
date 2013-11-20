#include "bench_config.hpp"

#include <config.hpp>
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

	const uint32_t num_clusters = 1024;

	SimpleDataset simple_dataset(s_simple_data_dir, s_simple_database_location);
	LINFO << simple_dataset;

	std::stringstream index_output_file;
	index_output_file << simple_dataset.location() << "/index/" << num_clusters << ".index";
	InvertedIndex ii(index_output_file.str());

	for(uint32_t i=0; i<3; i++) {
		std::shared_ptr<InvertedIndex::MatchResults> matches = 
		std::static_pointer_cast<InvertedIndex::MatchResults>(ii.search(simple_dataset, nullptr, simple_dataset.image(i) ));	
		LINFO << "Query " << i << ": " << *matches;
	}

	return 0;
}