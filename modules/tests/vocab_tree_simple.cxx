#include "tests_config.hpp"
#include <search/vocab_tree/vocab_tree.hpp>
#include <iostream>

#include <utils/filesystem.hpp>
#include <utils/numerics.hpp>
#include <utils/dataset.hpp>
#include <utils/vision.hpp>
#include <utils/logger.hpp>
#include <search/bag_of_words/bag_of_words.hpp>
#include <search/inverted_index/inverted_index.hpp>


int main(int argc, char *argv[]) {

	VocabTree vt;
	std::shared_ptr<VocabTree::TrainParams> train_params = std::make_shared<VocabTree::TrainParams>();
	train_params->depth = 6;
	train_params->split = 4;
	

  //const uint32_t depth = 6;
  //const uint32_t depth = 6;

  SimpleDataset simple_dataset(s_simple_data_dir, s_simple_database_location);
  //LINFO << simple_dataset;

  std::stringstream vocab_output_file;
  vocab_output_file << simple_dataset.location() << "/vocabulary/" << train_params->split << "." 
    << train_params->depth << ".vocab";

  std::shared_ptr<VocabTree> bow = std::make_shared<BagOfWords>(vocab_output_file.str());

  InvertedIndex ii;
  std::shared_ptr<InvertedIndex::TrainParams> train_params = std::make_shared<InvertedIndex::TrainParams>();
  train_params->bag_of_words = bow;
  ii.train(simple_dataset, train_params, simple_dataset.all_images());

  std::stringstream index_output_file;
  index_output_file << simple_dataset.location() << "/index/" << num_clusters << ".index";
  filesystem::create_file_directory(index_output_file.str());
  ii.save(index_output_file.str());


  return 0;
}