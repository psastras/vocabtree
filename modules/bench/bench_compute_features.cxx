#include <config.hpp>

#include "bench_config.hpp"
#include <utils/filesystem.hpp>
#include <utils/dataset.hpp>
#include <utils/vision.hpp>
#include <utils/logger.hpp>
#include <utils/misc.hpp>
#include <search/bag_of_words/bag_of_words.hpp>
#include <iostream>

_INITIALIZE_EASYLOGGINGPP

void compute_features(const SimpleDataset &dataset, BagOfWords &bow) {
#if ENABLE_MULTITHREADING && ENABLE_MPI
	int rank, procs;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &procs);
	uint64_t images_per_node = (dataset.num_images() / procs) + 1;
	uint64_t begin = rank * images_per_node;
	uint64_t end = MIN((rank+1) * images_per_node, dataset.num_images());
#else
	uint64_t begin = 0;
	uint64_t end = dataset.num_images();
#endif

#if ENABLE_MULTITHREADING && ENABLE_OPENMP
	uint32_t num_threads = omp_get_max_threads();
	std::vector< cv::Ptr<cv::DescriptorMatcher> > matchers;
	for(uint32_t i=0; i<num_threads; i++) {
		matchers.push_back(vision::construct_descriptor_matcher(bow.vocabulary()));
	}
#else
	const cv::Ptr<cv::DescriptorMatcher> &matcher = vision::construct_descriptor_matcher(bow.vocabulary());
#endif
#if ENABLE_MULTITHREADING && ENABLE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
	for (int64_t i = begin; i < end; i++) {
		PTR_LIB::shared_ptr<SimpleDataset::SimpleImage> image = std::static_pointer_cast<SimpleDataset::SimpleImage>(dataset.image(i));
		if (image == 0) continue;

		const std::string &keypoints_location = dataset.location(image->feature_path("keypoints"));
		const std::string &descriptors_location = dataset.location(image->feature_path("descriptors"));
		
		const std::string &image_location = dataset.location(image->location());
		if (!filesystem::file_exists(image_location)) continue;
		cv::Mat keypoints, descriptors, descriptorsf, bow_descriptors;
		if (filesystem::file_exists(keypoints_location) && filesystem::file_exists(descriptors_location)) {
			filesystem::load_cvmat(descriptors_location, descriptors);
		} else {
			cv::Mat im = cv::imread(image_location, cv::IMREAD_GRAYSCALE);
			
			if (!vision::compute_sparse_sift_feature(im, PTR_LIB::shared_ptr<const vision::SIFTParams>(), keypoints, descriptors)) continue;
			filesystem::create_file_directory(keypoints_location);
			filesystem::create_file_directory(descriptors_location);

			filesystem::write_cvmat(keypoints_location, keypoints);
			filesystem::write_cvmat(descriptors_location, descriptors);
		}

#if ENABLE_MULTITHREADING && ENABLE_OPENMP
		const cv::Ptr<cv::DescriptorMatcher> &matcher = matchers[omp_get_thread_num()];
#endif
		descriptors.convertTo(descriptorsf, CV_32FC1);
		const std::string &bow_descriptor_location = dataset.location(image->feature_path("bow_descriptors"));
		filesystem::create_file_directory(bow_descriptor_location);
		if (!vision::compute_bow_feature(descriptorsf, matcher, bow_descriptors, PTR_LIB::shared_ptr< std::vector<std::vector<uint32_t> > >())) continue;
		const std::vector< std::pair<uint32_t, float> > &bow_descriptors_sparse = numerics::sparsify(bow_descriptors);
		filesystem::write_sparse_vector(bow_descriptor_location, bow_descriptors_sparse);
	}
}

int main(int argc, char *argv[]) {
#if ENABLE_MULTITHREADING && ENABLE_MPI
	MPI_Init(&argc, &argv);
#endif
	{
		SimpleDataset dataset(s_oxford100k_data_dir, s_oxford100k_database_location);
		BagOfWords bow;
		std::stringstream vocab_output_file;
		vocab_output_file << dataset.location() << "/vocabulary/" << s_oxford100k_num_clusters << ".vocab";
		if(filesystem::file_exists(vocab_output_file.str())) {
			bow.load(vocab_output_file.str());	
			compute_features(dataset, bow);
		} 
		else { 
			std::cerr << "No vocabulary found.";
		}	
	}

	{
		SimpleDataset dataset(s_oxfordmini_data_dir, s_oxfordmini_database_location);
		BagOfWords bow;
		std::stringstream vocab_output_file;
		vocab_output_file << dataset.location() << "/vocabulary/" << s_oxfordmini_num_clusters << ".vocab";
		if(filesystem::file_exists(vocab_output_file.str())) {
			bow.load(vocab_output_file.str());	
			compute_features(dataset, bow);
		} 
		else { 
			std::cerr << "No vocabulary found.";
		}
	}

	{
		SimpleDataset dataset(s_paul_data_dir, s_paul_database_location);
		BagOfWords bow;
		std::stringstream vocab_output_file;
		vocab_output_file << dataset.location() << "/vocabulary/" << s_paul_num_clusters << ".vocab";
		if(filesystem::file_exists(vocab_output_file.str())) {
			bow.load(vocab_output_file.str());	
			compute_features(dataset, bow);
		} 
		else { 
			std::cerr << "No vocabulary found.";
		}
	}


	// {
	// 	SimpleDataset dataset(s_holidays_data_dir, s_holidays_database_location);
	// 	BagOfWords bow;
	// 	std::stringstream vocab_output_file;
	// 	vocab_output_file << dataset.location() << "/vocabulary/" << s_oxford100k_num_clusters << ".vocab";
	// 	if(filesystem::file_exists(vocab_output_file.str())) {
	// 		bow.load(vocab_output_file.str());	
	// 		compute_features(dataset, bow);
	// 	} 
	// 	else { 
	// 		std::cerr << "No vocabulary found.";
	// 	}
	// }

#if ENABLE_MULTITHREADING && ENABLE_MPI
	MPI_Finalize();
#endif
	return 0;
}