#pragma once

#include <string>
#include <stdint.h>
#include <opencv2/opencv.hpp>

// Abstract class representing an image.  Implementing classes must provide a way to 
// load images and construct image paths for loading features. 
class Image {
public:
	Image(uint64_t image_id);
	
	uint64_t id; // All images are assigned a unique id in the dataset.

	virtual ~Image();

	virtual std::string feature_path(const std::string &feat_name) const = 0;

	// Loads sift feature data as a cv::Mat into passed in class, returns true if success, false otherwise.
	bool load_sift_feature(cv::Mat &data) const;
	
	// Loads bag of words feature into the vector in a sparse format (index, value)
	bool load_bow_feature(std::vector<std::pair<uint32_t, float > > &data) const;

private:
	
};
