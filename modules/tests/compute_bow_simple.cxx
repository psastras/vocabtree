#include "tests_config.hpp"

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
#if ENABLE_MPI
#include <mpi.h>
#endif
_INITIALIZE_EASYLOGGINGPP

int main(int argc, char *argv[]) {
#if ENABLE_MPI
	MPI::Init(argc, argv);
	int rank = MPI::COMM_WORLD.Get_rank();
	int size = MPI::COMM_WORLD.Get_size();

	std::cout << "rank: " << rank << ", size: " << size << std::endl;
#endif

	SimpleDataset simple_dataset(s_simple_data_dir, s_simple_database_location);
	LINFO << simple_dataset;

	BagOfWords bow;
	std::shared_ptr<BagOfWords::TrainParams> train_params = std::make_shared<BagOfWords::TrainParams>();
	const std::vector<  std::shared_ptr<const Image> > &all_images = simple_dataset.all_images();
	std::cout << "Start train" << std::endl;
	bow.train(simple_dataset, train_params, all_images);
	std::cout << "Done train" << std::endl;
#if ENABLE_MPI
	MPI::Finalize();
	return 0;
#endif

	std::stringstream vocab_output_file;
	vocab_output_file << simple_dataset.location() << "/vocabulary/" << train_params->numClusters << ".vocab";
	bow.save(vocab_output_file.str());

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
		const std::string &sift_descriptor_location = simple_dataset.location(all_images[i]->feature_path("descriptors"));
		const std::string &bow_descriptor_location = simple_dataset.location(all_images[i]->feature_path("bow_descriptors"));

		cv::Mat descriptors, bow_descriptors;
		if (!filesystem::file_exists(sift_descriptor_location)) continue;
		if (!filesystem::load_cvmat(sift_descriptor_location, descriptors)) continue;
		filesystem::create_file_directory(bow_descriptor_location);

		if (!vision::compute_bow_feature(descriptors, matcher, bow_descriptors, nullptr)) continue;
		const std::vector< std::pair<uint32_t, float> > &bow_descriptors_sparse = numerics::sparsify(bow_descriptors);
		filesystem::write_sparse_vector(bow_descriptor_location, bow_descriptors_sparse);
		
		LINFO << "Wrote " << bow_descriptor_location;
	}

#if ENABLE_MPI
	MPI::Finalize();
#endif

	return 0;
}