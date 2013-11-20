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

	const uint32_t num_clusters = 16000;

	SimpleDataset oxford_dataset(s_oxford_data_dir, s_oxford_database_location);
	LINFO << oxford_dataset;

	std::stringstream vocab_output_file;
	vocab_output_file << oxford_dataset.location() << "/vocabulary/" << num_clusters << ".vocab";

	std::shared_ptr<BagOfWords> bow = std::make_shared<BagOfWords>(vocab_output_file.str());

	InvertedIndex ii;
	std::shared_ptr<InvertedIndex::TrainParams> train_params = std::make_shared<InvertedIndex::TrainParams>();
	train_params->bag_of_words = bow;
	ii.train(oxford_dataset, train_params, oxford_dataset.random_images(1024));

	std::stringstream index_output_file;
	index_output_file << oxford_dataset.location() << "/index/" << num_clusters << ".index";
	filesystem::create_file_directory(index_output_file.str());
	ii.save(index_output_file.str());


	return 0;
}