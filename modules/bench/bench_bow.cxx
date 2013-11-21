#include "bench_config.hpp"

#include <config.hpp>

#include <utils/filesystem.hpp>
#include <utils/numerics.hpp>
#include <utils/dataset.hpp>
#include <utils/vision.hpp>
#include <utils/logger.hpp>
#include <search/bag_of_words/bag_of_words.hpp>

#include <iostream>
#include <sstream>

#if ENABLE_MULTITHREADING && ENABLE_OPENMP
#include <omp.h>
#endif

_INITIALIZE_EASYLOGGINGPP

int main(int argc, char *argv[]) {

	SimpleDataset oxford_dataset(s_oxford_data_dir, s_oxford_database_location);
	LINFO << oxford_dataset;
	
	std::shared_ptr<BagOfWords::TrainParams> train_params = std::make_shared<BagOfWords::TrainParams>();
	train_params->numClusters = s_oxford_num_clusters;

	// cluster sift features
	BagOfWords bow;
	std::stringstream vocab_output_file;
	vocab_output_file << oxford_dataset.location() << "/vocabulary/" << train_params->numClusters << ".vocab";
	if(filesystem::file_exists(vocab_output_file.str())) {
		bow.load(vocab_output_file.str());
	} else {
		
		const std::vector<  std::shared_ptr<const Image> > &random_images = oxford_dataset.random_images(1024);
		bow.train(oxford_dataset, train_params, random_images);		
		bow.save(vocab_output_file.str());
	}
	
	const std::vector<  std::shared_ptr<const Image> > &all_images = oxford_dataset.all_images();

#if ENABLE_MULTITHREADING && ENABLE_OPENMP
	uint32_t num_threads = omp_get_max_threads();
	std::vector< cv::Ptr<cv::DescriptorMatcher> > matchers;
	for(uint32_t i=0; i<num_threads; i++) {
		matchers.push_back(vision::construct_descriptor_matcher(bow.vocabulary()));
	}
#pragma omp parallel for schedule(dynamic)
#else
	const cv::Ptr<cv::DescriptorMatcher> &matcher = vision::construct_descriptor_matcher(bow.vocabulary());
#endif

	for (int64_t i = 0; i < (int64_t)all_images.size(); i++) {

#if ENABLE_MULTITHREADING && ENABLE_OPENMP
		const cv::Ptr<cv::DescriptorMatcher> &matcher = matchers[omp_get_thread_num()];
#endif
		const std::string &sift_descriptor_location = oxford_dataset.location(all_images[i]->feature_path("descriptors"));
		const std::string &bow_descriptor_location = oxford_dataset.location(all_images[i]->feature_path("bow_descriptors"));

		cv::Mat descriptors, bow_descriptors;
		if (!filesystem::file_exists(sift_descriptor_location)) continue;
		if (!filesystem::load_cvmat(sift_descriptor_location, descriptors)) continue;
		filesystem::create_file_directory(bow_descriptor_location);

		if (!vision::compute_bow_feature(descriptors, matcher, bow_descriptors, nullptr)) continue;
		const std::vector< std::pair<uint32_t, float> > &bow_descriptors_sparse = numerics::sparsify(bow_descriptors);
		filesystem::write_sparse_vector(bow_descriptor_location, bow_descriptors_sparse);
		
		LINFO << "Wrote " << bow_descriptor_location;
	}

	return 0;
}