#pragma once

#include <string>
#include <stdint.h>
#include <opencv2/opencv.hpp>

#include "numerics.hpp"

/// Abstract class representing an image.  Implementing classes must provide a way to 
/// load images and construct image paths for loading features.  See tests and benchmarks
/// for example implementations of Image.
class Image {
public:
	Image(uint64_t image_id);
	
	uint64_t id; /// All images are assigned a unique id in the dataset.

	virtual ~Image();

	/// Returns the corresponding feature path given a feature name (ex. "sift").
	virtual std::string feature_path(const std::string &feat_name) const = 0;

	/// Returns the image location relative to the database data directory.
	virtual std::string location() const = 0;
protected:
	// std::function<std::vector<char>(const std::string &feat_name)> load_function;
private:
	
};

