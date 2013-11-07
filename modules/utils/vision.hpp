#pragma once

#include <opencv2/opencv.hpp>
#include <memory>

/// Provides useful wrappers around many OpenCV functions as well
/// as some simple vision - based routines.
namespace vision {

	/// Describes SIFT extraction parameters when calling OpenCV's
	/// SIFT implementation.  See the OpenCV documentation for
	/// more details.
	struct SIFTParams {
		int max_features 			= 0;	/// Max number of features to retrieve (0 keeps all features).
		int num_octave_layers 		= 3;	/// Number of octave layers in the pyramid to compute.
		double contrast_threshold 	= 0.04; /// Contrast threshold, lower -> more features, but less stable.
		double edge_threshold 		= 11;   /// Edge threshold, higher -> more features, but less stable.
		double sigma 				= 1.6;  /// Smoothing kernel size, higher -> more smoothing, fewer features.
	};

	/// Given a grayscale image, img, and SIFT extraction parameters computes sparse sift features.  If 
	/// params is a nullptr, we use the default settings for SIFTParams.  Returns true if successful,
	/// false otherwise.
	bool compute_sparse_sift_feature(const cv::Mat &img, const std::shared_ptr<const SIFTParams> &params ,
		cv::Mat &keypoints, cv::Mat &descriptors);

	/// Given a set of image descriptors, a descriptor matcher computes the the cluster match based on the matcher
	/// of each descriptor and returns the histogram of clusters.  If cluster_indices is not null, it will also
	/// return the cluster assignment of each descriptor vector.  Returns true if successful, false otherwise.
	bool compute_bow_feature(const cv::Mat& descriptors, const cv::Ptr<cv::DescriptorMatcher> &matcher,
			cv::Mat& bow_descriptors, std::shared_ptr< std::vector<std::vector<uint32_t> > > cluster_indices);

	/// Given a vocabulary constructs a FLANN based matcher needed to compute Bag of Words (BoW) features.  Expects
	/// the vocabulary to be in the same format as computed in the search module.
	cv::Ptr<cv::DescriptorMatcher> construct_descriptor_matcher(const cv::Mat &vocabulary);

	/// Merges the descriptors into a single matrix.  This is useful for clustering, which requires a single
	/// matrix.
	cv::Mat merge_descriptors(std::vector<cv::Mat> &descriptors, bool release_original = true);

	/// Given a pair of keypoints and corresponding SIFT descriptors, attempts to compute a RANSAC
	/// homography between them.  Results are returned as MatchInfo.  This function is useful in conjuction
	/// with is_good_homography_match which reads the MatchInfo from this function and determines if
	/// a good registration has been found.
	// cv::MatchInfo find_homography(const cv::Mat &keypoints_0, const cv::Mat &descriptors_0, 
	// 	const cv::Mat &keypoints_1, const cv::Mat &descriptors_1);
};