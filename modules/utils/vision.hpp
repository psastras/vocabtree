#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>
#include <memory>

#include "config.hpp"

/// Provides useful wrappers around many OpenCV functions as well
/// as some simple vision - based routines.
namespace vision {

	typedef std::set<std::pair<int,int> > MatchesSet;

	/// Describes SIFT extraction parameters when calling OpenCV's
	/// SIFT implementation.  See the OpenCV documentation for
	/// more details.
	struct SIFTParams {
		SIFTParams(int max_features = 0, int num_octave_layers = 3, 
			double contrast_threshold = 0.04, double edge_threshold = 11, 
			double sigma = 1.6) : max_features(max_features), num_octave_layers(num_octave_layers),
			contrast_threshold(contrast_threshold), edge_threshold(edge_threshold), sigma(sigma) {

			}

		int max_features ;	/// Max number of features to retrieve (0 keeps all features).
		int num_octave_layers;	/// Number of octave layers in the pyramid to compute.
		double contrast_threshold; /// Contrast threshold, lower -> more features, but less stable.
		double edge_threshold;   /// Edge threshold, higher -> more features, but less stable.
		double sigma;  /// Smoothing kernel size, higher -> more smoothing, fewer features.
	};

	/// Given a grayscale image, img, and SIFT extraction parameters computes sparse sift features.  If 
	/// params is a 0, we use the default settings for SIFTParams.  Returns true if successful,
	/// false otherwise.
	bool compute_sparse_sift_feature(const cv::Mat &img, const PTR_LIB::shared_ptr<const SIFTParams> &params ,
		cv::Mat &keypoints, cv::Mat &descriptors);

	/// Given a set of image descriptors, a descriptor matcher computes the the cluster match based on the matcher
	/// of each descriptor and returns the histogram of clusters.  If cluster_indices is not null, it will also
	/// return the cluster assignment of each descriptor vector.  Returns true if successful, false otherwise.
	bool compute_bow_feature(const cv::Mat& descriptors, const cv::Ptr<cv::DescriptorMatcher> &matcher,
			cv::Mat& bow_descriptors, PTR_LIB::shared_ptr< std::vector<std::vector<uint32_t> > > cluster_indices);

	/// Given a vocabulary constructs a FLANN based matcher needed to compute Bag of Words (BoW) features.  Expects
	/// the vocabulary to be in the same format as computed in the search module.
	cv::Ptr<cv::DescriptorMatcher> construct_descriptor_matcher(const cv::Mat &vocabulary);

	/// Merges the descriptors into a single matrix.  This is useful for clustering, which requires a single
	/// matrix.
	cv::Mat merge_descriptors(std::vector<cv::Mat> &descriptors, bool release_original = true);

	/// Computes a homography between the input pairs of points and descriptors and returns the result
	/// in MatchesInfo (including homography, confidence score, etc.)  If output inlier vectors are 
	/// provided, will insert a list of feature indices belonging to the homography inliers.
	void geo_verify_h(const cv::Mat &descriptors0, const cv::Mat &points0,
		const cv::Mat &descriptors1, const cv::Mat &points1, cv::detail::MatchesInfo &matches_info,
		std::vector<uint32_t> *inliers0 = 0, std::vector<uint32_t> *inliers1 = 0);

	/// Computes a fundamental matrix between the input pairs of points and descriptors and returns the result
	/// in MatchesInfo (including fundamental, confidence score, etc.)  If output inlier vectors are 
	/// provided, will insert a list of feature indices belonging to the fundamental inliers.
	void geo_verify_f(const cv::Mat &descriptors0, const cv::Mat &points0,
		const cv::Mat &descriptors1, const cv::Mat &points1, cv::detail::MatchesInfo &matches_info,
		std::vector<uint32_t> *inliers0 = 0, std::vector<uint32_t> *inliers1 = 0);

	/// Returns true if matches_info represents a good match, false if otherwise.  The heuristic is based
	/// on the total number of inliers and the ratio of inliers to outliers.
	bool is_good_match(const cv::detail::MatchesInfo &matches_info);
};