#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

/// Provides useful wrappers around many numerical functionality, such as dealing with sparse
/// and dense matrix / vector data.
namespace numerics {
	typedef std::vector< std::pair<uint32_t, float > > sparse_vector_t;

	/// Converts the input 1D cv::Mat to a sparse format, where each pair in the vector
	/// is index, value.  This is useful for BoW features which are usually zero.
	std::vector< std::pair<uint32_t, float> > sparsify(const cv::Mat &dense);

	/// Converts the cosine similarity between two sparse weight vectors, which are premultiplied
	/// by the relevant entries in idfw.  Sample usage would be weights0 and weights1 to represent
	/// two BoW vectors, and idfw to represent a vector of inverse document frequencies.
	float cos_sim(const std::vector<std::pair<uint32_t, float> > &weights0, 
		const std::vector<std::pair<uint32_t, float> > &weights1,
		const std::vector<float> &idfw);

}