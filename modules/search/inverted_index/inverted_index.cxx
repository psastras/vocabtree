#include "inverted_index.hpp"
#include <iostream>

InvertedIndex::InvertedIndex() : SearchBase() {


}

bool InvertedIndex::load (const std::string &file_path) {
	std::cout << "Reading inverted index from " << file_path << "..." << std::endl;

	// code here

	std::cout << "Done reading inverted index." << std::endl;
	
	return false;
}


bool InvertedIndex::save (const std::string &file_path) const {
	std::cout << "Writing inverted index to " << file_path << "..." << std::endl;

	// code here

	std::cout << "Done writing inverted index." << std::endl;

	return false;
}

bool InvertedIndex::train (const std::shared_ptr<const TrainParamsBase> &params, const std::vector< std::shared_ptr<const Image > > &examples) {
	const std::shared_ptr<const TrainParams> &ii_params = std::static_pointer_cast<const TrainParams>(params);
	uint32_t k = ii_params->numClusters;
	uint32_t n = ii_params->numFeatures;

	std::cout << "Training inverted index with " << k << " clusters on " << n << " features." << std::endl;
	std::cout << "Reading features from disk..." << std::endl;
	for(size_t i=0; i<examples.size(); i++) {
		const std::string sift_feat_path = examples[i]->feature_path("sift");
		
		//load sift

		// push back feature
	}
	std::cout << "Clustering..." << std::endl;
	// cluster all features
	std::cout << "Done training inverted index." << std::endl;
	return false;
}

std::shared_ptr<MatchResultsBase> InvertedIndex::search (const std::shared_ptr<const SearchParamsBase> &params, const std::shared_ptr<const Image > &example) {
	std::cout << "Searching for matching images..." << std::endl;
	const std::shared_ptr<const SearchParams> &ii_params = std::static_pointer_cast<const SearchParams>(params);
	
	std::shared_ptr<MatchResults> match_result = std::make_shared<MatchResults>();

	// returns zero as the only match with a score of zero
	match_result->tfidf_scores.push_back(0.f);
	match_result->matches.push_back(0);

	return (std::shared_ptr<MatchResultsBase>)match_result;
}