#include "tests_config.hpp"

#include <config.hpp>

#include <utils/filesystem.hpp>
#include <utils/dataset.hpp>
#include <utils/vision.hpp>
#include <utils/logger.hpp>
#include <iostream>

#if ENABLE_MULTITHREADING && ENABLE_OPENMP
#include <omp.h>
#endif
#if ENABLE_MULTITHREADING && ENABLE_MPI
#include <mpi.h>
#endif

_INITIALIZE_EASYLOGGINGPP

int main(int argc, char *argv[]) {
	LINFO << "HJ";
#if ENABLE_MULTITHREADING && ENABLE_MPI
	MPI::Init(argc, argv);
	int rank = MPI::COMM_WORLD.Get_rank();
	if(rank == 0) {
#endif
	SimpleDataset simple_dataset(s_oxfordmini_data_dir, s_oxfordmini_database_location);
	  //SimpleDataset simple_dataset(s_simple_data_dir, s_simple_database_location);
	LINFO << simple_dataset;
#if ENABLE_MULTITHREADING && ENABLE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
	for (int64_t i = 0; i < simple_dataset.num_images(); i++) {

		std::shared_ptr<SimpleDataset::SimpleImage> image = std::static_pointer_cast<SimpleDataset::SimpleImage>(simple_dataset.image(i));
		if (image == nullptr) continue;

		const std::string &keypoints_location = simple_dataset.location(image->feature_path("keypoints"));
		const std::string &descriptors_location = simple_dataset.location(image->feature_path("descriptors"));
		// if (filesystem::file_exists(keypoints_location) && filesystem::file_exists(descriptors_location)) continue;
		
		const std::string &image_location = simple_dataset.location(image->location());

		if (!filesystem::file_exists(image_location)) continue;
		
		cv::Mat im = cv::imread(image_location, cv::IMREAD_GRAYSCALE);

		cv::Mat keypoints, descriptors;
		if (!vision::compute_sparse_sift_feature(im, nullptr, keypoints, descriptors)) continue;

		filesystem::create_file_directory(keypoints_location);
		filesystem::create_file_directory(descriptors_location);

		filesystem::write_cvmat(keypoints_location, keypoints);
		filesystem::write_cvmat(descriptors_location, descriptors);
	}
#if ENABLE_MULTITHREADING && ENABLE_MPI
	}
	MPI::Finalize();
#endif
	return 0;
}
