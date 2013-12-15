#include "bench_config.hpp"

#include <config.hpp>

#include <utils/filesystem.hpp>
#include <utils/dataset.hpp>
#include <utils/vision.hpp>
#include <utils/logger.hpp>
#include <utils/misc.hpp>
#include <iostream>

#if ENABLE_MULTITHREADING && ENABLE_OPENMP
#include <omp.h>
#endif
#if ENABLE_MULTITHREADING && ENABLE_MPI
#include <mpi.h>
#endif

_INITIALIZE_EASYLOGGINGPP


void compute_features(const SimpleDataset &dataset) {
	LINFO << dataset;
#if ENABLE_MULTITHREADING && ENABLE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
	for (int64_t i = 0; i < (int64_t)dataset.num_images(); i++) {
		PTR_LIB::shared_ptr<SimpleDataset::SimpleImage> image = std::static_pointer_cast<SimpleDataset::SimpleImage>(dataset.image(i));
		if (image == 0) continue;

		const std::string &keypoints_location = dataset.location(image->feature_path("keypoints"));
		const std::string &descriptors_location = dataset.location(image->feature_path("descriptors"));
		
		const std::string &image_location = dataset.location(image->location());
		if (!filesystem::file_exists(image_location)) continue;
		
	
		cv::Mat im = cv::imread(image_location, cv::IMREAD_GRAYSCALE);

		cv::Mat keypoints, descriptors;
		if (!vision::compute_sparse_sift_feature(im, 0, keypoints, descriptors)) continue;

		filesystem::create_file_directory(keypoints_location);
		filesystem::create_file_directory(descriptors_location);

		filesystem::write_cvmat(keypoints_location, keypoints);
		filesystem::write_cvmat(descriptors_location, descriptors);
	}
}

int main(int argc, char *argv[]) {
#if ENABLE_MULTITHREADING && ENABLE_MPI
	MPI::Init(argc, argv);
#endif
	// {
	// 	SimpleDataset oxford_dataset(s_oxford_data_dir, s_oxford_database_location);
	// 	compute_features(oxford_dataset);
	// }

	{
		SimpleDataset oxford100k_dataset(s_oxford100k_data_dir, s_oxford100k_database_location);
		compute_features(oxford100k_dataset);
	}
#if ENABLE_MULTITHREADING && ENABLE_MPI
	MPI::Finalize();
#endif
	return 0;
}