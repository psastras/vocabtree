#include "image.hpp"

#include <utils/filesystem.hpp>

Image::Image(uint64_t imageid) { 
	id = imageid;
}

Image::~Image() { }

bool Image::load_sift_feature(cv::Mat &data) const {
	std::string path = feature_path("sift");
	return filesystem::load_cvmat(path, data);
}

bool Image::load_bow_feature(std::vector<std::pair<uint32_t, float > > &data) const {
	std::string path = feature_path("bow");
	return filesystem::load_bow(path, data);
}