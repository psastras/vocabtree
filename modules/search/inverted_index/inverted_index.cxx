#include "inverted_index.hpp"
#include <utils/filesystem.hpp>
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

bool InvertedIndex::train(Dataset &dataset, const std::shared_ptr<const TrainParamsBase> &params, const std::vector< std::shared_ptr<const Image > > &examples) {
	for (size_t i = 0; i < dataset.num_images(); i++) {
		const std::shared_ptr<const Image> &image = dataset.image(i);
		const std::string &bow_descriptors_location = dataset.location(image->feature_path("bow_descriptors"));

		if (!filesystem::file_exists(bow_descriptors_location)) continue;


	}

	return false;
}

std::shared_ptr<MatchResultsBase> InvertedIndex::search(Dataset &dataset, const std::shared_ptr<const SearchParamsBase> &params, const std::shared_ptr<const Image > &example) {
	std::cout << "Searching for matching images..." << std::endl;
	const std::shared_ptr<const SearchParams> &ii_params = std::static_pointer_cast<const SearchParams>(params);
	
	std::shared_ptr<MatchResults> match_result = std::make_shared<MatchResults>();

	// returns zero as the only match with a score of zero
	match_result->tfidf_scores.push_back(0.f);
	match_result->matches.push_back(0);

	return (std::shared_ptr<MatchResultsBase>)match_result;
}