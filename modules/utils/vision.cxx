#include "vision.hpp"

#include <opencv2/nonfree/nonfree.hpp>

namespace vision {
	bool compute_sparse_sift_feature(const cv::Mat &img, const std::shared_ptr<const SIFTParams> &params ,
		cv::Mat &keypoints, cv::Mat &descriptors) {
		std::shared_ptr<const SIFTParams> sift_parameters;
		
		if(params == nullptr) {
			sift_parameters = std::make_shared<const SIFTParams>();
		}

		cv::SIFT sift_extractor(sift_parameters->max_features, sift_parameters->num_octave_layers,
				 sift_parameters->contrast_threshold, sift_parameters->edge_threshold, sift_parameters->sigma);

		if(img.size().area() > 0) {
			
			std::vector<cv::KeyPoint> keypoint_vec;
			sift_extractor.detect(img, keypoint_vec);
			sift_extractor.compute(img, keypoint_vec, descriptors);
			keypoints = cv::Mat(keypoint_vec.size(), 2, CV_32FC1);
        
			for(int i=0; i<(int)keypoint_vec.size(); i++) {
			    keypoints.at<float>(i, 0) = keypoint_vec[i].pt.x;
			    keypoints.at<float>(i, 1) = keypoint_vec[i].pt.y;
			}

			return true;
		}

		return false;
	}

	bool compute_bow_feature(const cv::Mat& descriptors, const cv::Ptr<cv::DescriptorMatcher> &matcher,
			cv::Mat& bow_descriptors, std::shared_ptr< std::vector<std::vector<uint32_t> > > cluster_indices) {

		int clusterCount = matcher->getTrainDescriptors().size();

	    std::vector<cv::DMatch> matches;
	    matcher->match(descriptors, matches);

	    if(cluster_indices != nullptr) {
			cluster_indices->clear();
			cluster_indices->resize(clusterCount);
	    }

	    bow_descriptors.release();
	    bow_descriptors = cv::Mat( 1, clusterCount, CV_32FC1, cv::Scalar::all(0.0) );
	    float *dptr = (float*)bow_descriptors.data;
	    for(size_t i=0; i < matches.size(); i++) {
	        int queryIdx = matches[i].queryIdx;
	        int trainIdx = matches[i].trainIdx; 

	        dptr[trainIdx] = dptr[trainIdx] + 1.f;
	        if(cluster_indices != nullptr) {
	            (*cluster_indices)[trainIdx].push_back( queryIdx );
	        }
	    }

	    return true;
	}

	cv::Ptr<cv::DescriptorMatcher> construct_descriptor_matcher(const cv::Mat &vocabulary) {
		cv::Ptr<cv::DescriptorMatcher> matcher = cv::makePtr<cv::FlannBasedMatcher>();
		matcher->add(std::vector<cv::Mat>(1, vocabulary));
		return matcher;
	}
}