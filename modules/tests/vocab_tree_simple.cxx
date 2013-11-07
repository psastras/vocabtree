#include "tests_config.hpp"
#include <search/vocab_tree/vocab_tree.hpp>
#include <iostream>


int main(int argc, char *argv[]) {

	VocabTree ii;
	std::shared_ptr<VocabTree::TrainParams> train_params = std::make_shared<VocabTree::TrainParams>();
	train_params->depth = 6;
	train_params->split = 4;
	

	return 0;
}