#include <config.hpp>

#include "bench_config.hpp"

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
#endif
	const uint32_t num_clusters = s_paul_num_clusters;
	SimpleDataset oxford_dataset(s_paul_data_dir, s_paul_database_location);
	LINFO << oxford_dataset;

	std::stringstream vocab_output_file;
	vocab_output_file << oxford_dataset.location() << "/vocabulary/" << num_clusters << ".vocab";

	PTR_LIB::shared_ptr<BagOfWords> bow = PTR_LIB::make_shared<BagOfWords>(vocab_output_file.str());

	InvertedIndex ii;
	PTR_LIB::shared_ptr<InvertedIndex::TrainParams> train_params = PTR_LIB::make_shared<InvertedIndex::TrainParams>();
	train_params->bag_of_words = bow;
	ii.train(oxford_dataset, train_params, oxford_dataset.all_images());

	std::stringstream index_output_file;
	index_output_file << oxford_dataset.location() << "/index/" << num_clusters << ".index";
	filesystem::create_file_directory(index_output_file.str());
	ii.save(index_output_file.str());

#if ENABLE_MULTITHREADING && ENABLE_MPI
	MPI::Finalize();
#endif
	return 0;
}