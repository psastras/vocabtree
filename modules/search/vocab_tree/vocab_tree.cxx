#include "vocab_tree.hpp"
#include <iostream>

VocabTree::VocabTree() : SearchBase() {


}

bool VocabTree::load (const std::string &file_path) {
	std::cout << "Reading vocab tree from " << file_path << "..." << std::endl;

	// code here

	std::cout << "Done reading vocab tree." << std::endl;
	
	return false;
}


bool VocabTree::save (const std::string &file_path) const {
	std::cout << "Writing vocab tree to " << file_path << "..." << std::endl;

	// code here

	std::cout << "Done writing vocab tree." << std::endl;

	return false;
}

bool VocabTree::train (const std::shared_ptr<const TrainParamsBase> &params, const std::vector< std::shared_ptr<const Image > > &examples) {
	const std::shared_ptr<const TrainParams> &vt_params = std::static_pointer_cast<const TrainParams>(params);
	uint32_t split = vt_params->split;
	uint32_t depth = vt_params->depth;

	return false;
}

std::shared_ptr<MatchResultsBase> VocabTree::search (const std::shared_ptr<const SearchParamsBase> &params, const std::shared_ptr<const Image > &example) {
	std::cout << "Searching for matching images..." << std::endl;
	const std::shared_ptr<const SearchParams> &ii_params = std::static_pointer_cast<const SearchParams>(params);
	
	std::shared_ptr<MatchResults> match_result = std::make_shared<MatchResults>();

	// returns zero as the only match 
	match_result->matches.push_back(0);

	return (std::shared_ptr<MatchResultsBase>)match_result;
}