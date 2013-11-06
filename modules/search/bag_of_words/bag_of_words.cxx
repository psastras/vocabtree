#include "bag_of_words.hpp"
#include <iostream>

BagOfWords::BagOfWords() : SearchBase() {


}

bool BagOfWords::load (const std::string &file_path) {
	std::cout << "Reading bag of words from " << file_path << "..." << std::endl;

	// code here

	std::cout << "Done reading bag of words." << std::endl;
	
	return false;
}


bool BagOfWords::save (const std::string &file_path) const {
	std::cout << "Writing bag of words to " << file_path << "..." << std::endl;

	// code here

	std::cout << "Done writing bag of words." << std::endl;

	return false;
}

bool BagOfWords::train (const std::shared_ptr<const TrainParamsBase> &params, const std::vector< std::shared_ptr<const Image > > &examples) {
	const std::shared_ptr<const TrainParams> &ii_params = std::static_pointer_cast<const TrainParams>(params);
	uint32_t k = ii_params->numClusters;
	uint32_t n = ii_params->numFeatures;

	std::cout << "Training bag of words with " << k << " clusters on " << n << " features." << std::endl;
	std::cout << "Reading features from disk..." << std::endl;
	for(size_t i=0; i<examples.size(); i++) {
		
		const std::string sift_feat_path = examples[i]->feature_path("sift");
		
		//load sift

		// push back feature
	}
	std::cout << "Clustering..." << std::endl;
	// cluster all features
	std::cout << "Done training bag of words." << std::endl;
	return false;
}

std::shared_ptr<MatchResultsBase> BagOfWords::search (const std::shared_ptr<const SearchParamsBase> &params, const std::shared_ptr<const Image > &example) {
	assert(0);
}