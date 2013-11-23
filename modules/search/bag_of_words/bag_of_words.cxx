#include <config.hpp>

#include "bag_of_words.hpp"

#include <utils/filesystem.hpp>
#include <utils/vision.hpp>
#include <iostream>
#include <memory>

#if ENABLE_FASTCLUSTER
#include <fastann/fastann.hpp>
#include <fastcluster/kmeans.hpp>
#endif
#if ENABLE_MPI
#include <mpi.h>
#endif

BagOfWords::BagOfWords() : SearchBase() {


}

BagOfWords::BagOfWords(const std::string &file_path) : SearchBase(file_path) {
	if(!filesystem::file_exists(file_path)) {
		std::cerr << "Error reading bag of words from " << file_path << std::endl;
		return;
	}
	if(!this->load(file_path)) {
		std::cerr << "Error reading bag of words from " << file_path << std::endl;
	}
}

bool BagOfWords::load (const std::string &file_path) {
	std::cout << "Reading bag of words from " << file_path << "..." << std::endl;

	if (!filesystem::load_cvmat(file_path, vocabulary_matrix)) {
		std::cerr << "Failed to read vocabulary from " << file_path << std::endl;
		return false;
	}

	std::cout << "Done reading bag of words." << std::endl;
	
	return true;
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
#if ENABLE_FASTCLUSTER && ENABLE_MPI
void load_rows_in_mem(void* p, unsigned l, unsigned r, float* out) {
	float *fp = (float *)p;
	memcpy(out, &fp[l * 128] , sizeof(float) * 128 * (r - l));
}

void load_rows_out_mem(void* p, unsigned l, unsigned r, float* out) {
	assert(0);
	// float *fp = (float *)p;
	// memcpy(out, &fp[l * 128] , sizeof(float) * 128 * (r - l));
}

fastann::nn_obj<float>* build_nnobj(void* p, float* clusters, unsigned K, unsigned D) {
	fastann::nn_obj<float>* kd_tree = fastann::nn_obj_build_kdtree(clusters, K, D, 8, 512);
	return kd_tree;
}
#endif

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
	
#if ENABLE_FASTCLUSTER && ENABLE_MPI
	
	int rank = MPI::COMM_WORLD.Get_rank();

	uint32_t D = merged_descriptor.cols;
	float *dataf = (float *)merged_descriptor.data;
	vocabulary_matrix = cv::Mat(k, D, CV_32FC1);
	float *clusters = (float *)vocabulary_matrix.data;

	// initialize the clusters (random at the moment)
	if(rank == 0) { // initial clusters get broadcast
		std::vector<uint64_t> indices(num_features);
		for(size_t i=0; i<indices.size(); i++) {
			indices[i] = i;
		}
		std::random_shuffle(indices.begin(), indices.end());
		for(size_t i=0; i<k; i++) {
			memcpy(&clusters[D * i], &dataf[D * indices[i]], sizeof(float) * D);
		}
	}

    fastcluster::kmeans<float>(load_rows_in_mem, (void *)merged_descriptor.data, 
                                    build_nnobj, (void *)merged_descriptor.data,
                                    (float *)clusters, num_features, D, k, 16, 0, (char *)0);
	return true;
#else
	cv::Mat labels;
	uint32_t attempts = 1;
	cv::TermCriteria tc(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 16, 0.0001);
	cv::kmeans(merged_descriptor, k, labels, tc, attempts, cv::KMEANS_PP_CENTERS, vocabulary_matrix);
#endif
	return true;
}

std::shared_ptr<MatchResultsBase> BagOfWords::search(Dataset &dataset, const std::shared_ptr<const SearchParamsBase> &params, const std::shared_ptr<const Image > &example) {
	assert(0);
	return nullptr;
}

cv::Mat BagOfWords::vocabulary() const {
	return vocabulary_matrix;
}

uint32_t BagOfWords::num_clusters() const {
	return vocabulary_matrix.rows;
}