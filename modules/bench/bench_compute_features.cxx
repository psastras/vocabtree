#include "bench_config.hpp"

#include <config.hpp>

#include <utils/filesystem.hpp>
#include <utils/dataset.hpp>
#include <utils/vision.hpp>
#include <utils/logger.hpp>
#include <iostream>

_INITIALIZE_EASYLOGGINGPP

int main(int argc, char *argv[]) {

	SimpleDataset oxford_dataset(s_oxford_data_dir, s_oxford_database_location);
	LINFO << oxford_dataset;
#if ENABLE_MULTITHREADING && ENABLE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
	for (int64_t i = 0; i < oxford_dataset.num_images(); i++) {

		std::shared_ptr<SimpleDataset::SimpleImage> image = std::static_pointer_cast<SimpleDataset::SimpleImage>(oxford_dataset.image(i));
		if (image == nullptr) continue;

		const std::string &keypoints_location = oxford_dataset.location(image->feature_path("keypoints"));
		const std::string &descriptors_location = oxford_dataset.location(image->feature_path("descriptors"));
		
		const std::string &image_location = oxford_dataset.location(image->location());

		if (!filesystem::file_exists(image_location)) continue;
		
		cv::Mat im = cv::imread(image_location, cv::IMREAD_GRAYSCALE);

		cv::Mat keypoints, descriptors;
		if (!vision::compute_sparse_sift_feature(im, nullptr, keypoints, descriptors)) continue;

		filesystem::create_file_directory(keypoints_location);
		filesystem::create_file_directory(descriptors_location);

		filesystem::write_cvmat(keypoints_location, keypoints);
		filesystem::write_cvmat(descriptors_location, descriptors);
	}

	return 0;
}