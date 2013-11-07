#pragma once

#include <stdint.h>
#include <opencv2/opencv.hpp>

/// Provides useful wrappers around many filesystem related functionality, including reading writing
/// certain common data structures as well as common operations (ex. file_exists).
namespace filesystem {
	/// Returns the basename of the input filepath, including or not including the extension.
	std::string basename(const std::string &path, bool include_extension = false);
	/// Returns true if file exists at location, else returns false.
	bool file_exists(const std::string& name);
	/// Recursively creates all directories if needed up to the specified file.
	void create_file_directory(const std::string &absfilepath);
	/// Writes a cv::Mat structure to the specified location.
	bool write_cvmat(const std::string &fname, const cv::Mat &data);
	/// Loads a cv::Mat structure from the specified location.  Returns true if file exists,
	/// false otherwise.
	bool load_cvmat(const std::string &fname, cv::Mat &data);
	/// Writes the BoW feature to the specified location.  First dimension of data is cluster index,
	/// second dimension is TF score.
	bool write_sparse_vector(const std::string &fname, const std::vector<std::pair<uint32_t, float > > &data);
	/// Loads the BoW feature from the specified location.  First dimension of data is cluster index,
	/// second dimension is TF score.
	bool load_sparse_vector(const std::string &fname, std::vector<std::pair<uint32_t, float > > &data);
	/// Lists all files in the given directory with an optional extension.  The extension must include
	/// the dot (ie. ext=".txt").
	std::vector<std::string> list_files(const std::string &path, const std::string &ext = "") ;
};