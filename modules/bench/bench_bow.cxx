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
#if ENABLE_MULTITHREADING && ENABLE_MPI
#include <mpi.h>
#endif

_INITIALIZE_EASYLOGGINGPP

void compute_bow(SimpleDataset &dataset, int num_clusters, int num_images, int num_features) {
#if ENABLE_FASTCLUSTER && ENABLE_MPI
	int rank = MPI::COMM_WORLD.Get_rank();
#else
	int rank = 0;
#endif
	if(rank == 0) {
		LINFO << dataset;
	}

	BagOfWords bow;
	PTR_LIB::shared_ptr<BagOfWords::TrainParams> train_params = PTR_LIB::make_shared<BagOfWords::TrainParams>();
	train_params->numClusters = num_clusters;
	train_params->numFeatures = num_features;

	std::stringstream vocab_output_file;
	vocab_output_file << dataset.location() << "/vocabulary/" << train_params->numClusters << ".vocab";

	if(filesystem::file_exists(vocab_output_file.str())) {
        bow.load(vocab_output_file.str());
    } else {
        const std::vector< PTR_LIB::shared_ptr<const Image> > &random_images = dataset.random_images(num_images);
        bow.train(dataset, train_params, random_images);                
#if ENABLE_FASTCLUSTER && ENABLE_MPI
        if(rank == 0) {
#endif
         	bow.save(vocab_output_file.str());
#if ENABLE_FASTCLUSTER && ENABLE_MPI        	
        }
#endif
    }

#if ENABLE_MULTITHREADING && ENABLE_MPI
	if(rank == 0) {
#endif
	
	const std::vector<  PTR_LIB::shared_ptr<const Image> > &all_images = dataset.all_images();

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
		const std::string &sift_descriptor_location = dataset.location(all_images[i]->feature_path("descriptors"));
		const std::string &bow_descriptor_location = dataset.location(all_images[i]->feature_path("bow_descriptors"));

		cv::Mat descriptors, bow_descriptors, descriptorsf;
		if (!filesystem::file_exists(sift_descriptor_location)) continue;
		if (!filesystem::load_cvmat(sift_descriptor_location, descriptors)) continue;
		descriptors.convertTo(descriptorsf, CV_32FC1);
		filesystem::create_file_directory(bow_descriptor_location);

		if (!vision::compute_bow_feature(descriptorsf, matcher, bow_descriptors, PTR_LIB::shared_ptr< std::vector<std::vector<uint32_t> > >())) continue;
		const std::vector< std::pair<uint32_t, float> > &bow_descriptors_sparse = numerics::sparsify(bow_descriptors);
		filesystem::write_sparse_vector(bow_descriptor_location, bow_descriptors_sparse);
		LINFO << "Wrote " << bow_descriptor_location;
	}
#if ENABLE_MULTITHREADING && ENABLE_MPI
	}
#endif
}

int main(int argc, char *argv[]) {

#if ENABLE_MULTITHREADING && ENABLE_MPI
	MPI::Init(argc, argv);
#endif

	// {
	// 	SimpleDataset oxford_dataset(s_oxford_data_dir, s_oxford_database_location);
	// 	compute_bow(oxford_dataset, s_oxford_num_clusters, 1024, 0);
	// }

	{
		SimpleDataset paul_dataset(s_paul_data_dir, s_paul_database_location);
		compute_bow(paul_dataset, s_paul_num_clusters, 2048, 0);
	}
	
#if ENABLE_MULTITHREADING && ENABLE_MPI
	MPI::Finalize();
#endif
	return 0;
}