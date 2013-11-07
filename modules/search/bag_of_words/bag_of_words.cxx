#include "bag_of_words.hpp"
#include <utils/filesystem.hpp>
#include <utils/vision.hpp>
#include <iostream>
#include <memory>
BagOfWords::BagOfWords() : SearchBase() {


}

bool BagOfWords::load (const std::string &file_path) {
	std::cout << "Reading bag of words from " << file_path << "..." << std::endl;

	if (!filesystem::load_cvmat(file_path, vocabulary_matrix)) {
		std::cerr << "Failed to read vocabulary from " << file_path << std::endl;
		return false;
	}

	std::cout << "Done reading bag of words." << std::endl;
	
	return false;
}


bool BagOfWords::save (const std::string &file_path) const {
	std::cout << "Writing bag of words to " << file_path << "..." << std::endl;

	filesystem::create_file_directory(file_path);
	if (!filesystem::write_cvmat(file_path, vocabulary_matrix)) {
		std::cerr << "Failed to write vocabulary to " << file_path << std::endl;
		return false;
	}

	std::cout << "Done writing bag of words." << std::endl;
	return true;
}

bool BagOfWords::train(Dataset &dataset, const std::shared_ptr<const TrainParamsBase> &params, const std::vector< std::shared_ptr<const Image > > &examples) {

	const std::shared_ptr<const TrainParams> &ii_params = std::static_pointer_cast<const TrainParams>(params);
	
	uint32_t k = ii_params->numClusters;
	uint32_t n = ii_params->numFeatures;

	std::vector<uint64_t> all_ids(examples.size());
	for (uint64_t i = 0; i < examples.size(); i++) {
		all_ids[i] = examples[i]->id;
	}
	std::random_shuffle(all_ids.begin(), all_ids.end());

	std::vector<cv::Mat> all_descriptors;
	uint64_t num_features = 0;
	for (size_t i = 0; i < all_ids.size(); i++) {
		std::shared_ptr<Image> image = std::static_pointer_cast<Image>(dataset.image(all_ids[i]));
		if (image == nullptr) continue;

		const std::string &descriptors_location = dataset.location(image->feature_path("descriptors"));
		if (!filesystem::file_exists(descriptors_location)) continue;

		cv::Mat descriptors;
		if (filesystem::load_cvmat(descriptors_location, descriptors)) {
			num_features += descriptors.rows;
			if (n > 0 && num_features > n) break;

			all_descriptors.push_back(descriptors);
		}
	}

	const cv::Mat merged_descriptor = vision::merge_descriptors(all_descriptors, true);
	cv::Mat labels;
	uint32_t attempts = 1;
	cv::TermCriteria tc(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 16, 0.0001);
	cv::kmeans(merged_descriptor, k, labels, tc, attempts, cv::KMEANS_PP_CENTERS, vocabulary_matrix);

	return true;
}

std::shared_ptr<MatchResultsBase> BagOfWords::search(Dataset &dataset, const std::shared_ptr<const SearchParamsBase> &params, const std::shared_ptr<const Image > &example) {
	assert(0);
	return nullptr;
}

cv::Mat BagOfWords::vocabulary() const {
	return vocabulary_matrix;
}