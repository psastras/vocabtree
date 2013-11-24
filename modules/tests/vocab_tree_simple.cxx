#include "tests_config.hpp"
#include <search/vocab_tree/vocab_tree.hpp>
#include <iostream>

#include <utils/filesystem.hpp>
#include <utils/numerics.hpp>
#include <utils/dataset.hpp>
#include <utils/vision.hpp>
#include <utils/logger.hpp>
#include <vis/matches_page.hpp>

int main(int argc, char *argv[]) {

  SimpleDataset simple_dataset(s_oxfordmini_data_dir, s_oxfordmini_database_location);
  //LINFO << simple_dataset;

  //std::stringstream vocab_output_file;
  //vocab_output_file << simple_dataset.location() << "/vocab/" << train_params->split << "-" 
    //<< train_params->depth << ".vocab";

  //std::shared_ptr<VocabTree> bow = std::make_shared<VocabTree>(vocab_output_file.str());

  VocabTree vt;
  std::shared_ptr<VocabTree::TrainParams> train_params = std::make_shared<VocabTree::TrainParams>();
  train_params->depth = 4;
  train_params->split = 4;
  vt.train(simple_dataset, train_params, simple_dataset.random_images(128));

  /*
  std::stringstream index_output_file;
  index_output_file << simple_dataset.location() << "/vocab/" << train_params->split << "-"
    << train_params->depth << ".vtree";
  filesystem::create_file_directory(index_output_file.str());
  vt.save(index_output_file.str());
  */

  MatchesPage html_output;
  for (uint32_t i = 0; i<256; i++) {
    std::shared_ptr<VocabTree::MatchResults> matches =
     std::static_pointer_cast<VocabTree::MatchResults>(vt.search(simple_dataset, nullptr, simple_dataset.image(i)));
    //LINFO << "Query " << i << ": " << *matches;
    printf("Matches for image %d: ", i);
    for (uint64_t id : matches->matches)
      printf("%d ", id);
    printf("\n");

     html_output.add_match(i, matches->matches, simple_dataset);
  }


    html_output.write(simple_dataset.location() + "/results/matches/");
  return 0;
}