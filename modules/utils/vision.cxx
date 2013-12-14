#include "vision.hpp"

#include <opencv2/nonfree/nonfree.hpp>

namespace vision {

	void geo_verify_f(const cv::Mat &descriptors0, const cv::Mat &points0,
		const cv::Mat &descriptors1, const cv::Mat &points1, cv::detail::MatchesInfo &matches_info,
		std::vector<uint32_t> *inliers0, std::vector<uint32_t> *inliers1) {

		cv::Ptr<cv::flann::IndexParams> indexParams = cv::makePtr<cv::flann::KDTreeIndexParams>();
	    cv::Ptr<cv::flann::SearchParams> searchParams = cv::makePtr<cv::flann::SearchParams>();
	    matches_info.confidence = 0;
	    matches_info.num_inliers = 0;
	    matches_info.H = cv::Mat::zeros(cv::Size(3, 3), CV_32FC1);
	    if(descriptors1.rows < 9 || descriptors0.rows < 9 || descriptors0.cols < 128 || descriptors1.cols < 128) return;
	    // if (descriptors0.depth() == CV_8U) {
	    //     indexParams->setAlgorithm(cvflann::FLANN_INDEX_LSH);
	    //     searchParams->setAlgorithm(cvflann::FLANN_INDEX_LSH);
	    // }
	    try {
		    cv::FlannBasedMatcher matcher(indexParams, searchParams);
		    std::vector< std::vector<cv::DMatch> > pair_matches;
		    // std::cout << "knn match 1" << std::endl;
		    MatchesSet matches;
		    // Find 1->2 matches
		    cv::Mat descriptors0f, descriptors1f;
		    descriptors0.convertTo(descriptors0f, CV_32FC1);
		    descriptors1.convertTo(descriptors1f, CV_32FC1);
		    matcher.knnMatch(descriptors0f, descriptors1f, pair_matches, 2);
		    float match_conf_ = 0.2f;
		    for (size_t i = 0; i < pair_matches.size(); ++i) {
		        if (pair_matches[i].size() < 2) continue;
		        const cv::DMatch& m0 = pair_matches[i][0];
		        const cv::DMatch& m1 = pair_matches[i][1];
		        if (m0.distance < (1.f - match_conf_) * m1.distance) {
		            matches_info.matches.push_back(m0);
		            matches.insert(std::make_pair(m0.queryIdx, m0.trainIdx));
		        }
		    }

		    pair_matches.clear();
		    matcher.knnMatch(descriptors1f, descriptors0f, pair_matches, 2);
		    for (size_t i = 0; i < pair_matches.size(); ++i)
		    {
		        if (pair_matches[i].size() < 2)
		            continue;
		        const cv::DMatch& m0 = pair_matches[i][0];
		        const cv::DMatch& m1 = pair_matches[i][1];
		        if (m0.distance < (1.f - match_conf_) * m1.distance)
		            if (matches.find(std::make_pair(m0.trainIdx, m0.queryIdx)) == matches.end()) // if we haven't already added this pair
		                matches_info.matches.push_back(cv::DMatch(m0.trainIdx, m0.queryIdx, m0.distance));
		    }

		    int num_matches_thresh1_ = 9;
		    // int num_matches_thresh2_ = 9;

		    if (matches_info.matches.size() < static_cast<size_t>(num_matches_thresh1_))
		        return;

		    cv::Mat src_points(1, static_cast<int>(matches_info.matches.size()), CV_32FC2);
		    cv::Mat dst_points(1, static_cast<int>(matches_info.matches.size()), CV_32FC2);
		    std::vector<uint32_t> src_idx, dst_idx;

		    for (size_t i = 0; i < matches_info.matches.size(); ++i)   {
		        const cv::DMatch& m = matches_info.matches[i];

		        cv::Point2f p(points0.at<float>(m.queryIdx, 0), points0.at<float>(m.queryIdx, 1));
		        src_points.at<cv::Point2f>(0, static_cast<int>(i)) = p;


		        p = cv::Point2f(points1.at<float>(m.trainIdx, 0), points1.at<float>(m.trainIdx, 1));
		        dst_points.at<cv::Point2f>(0, static_cast<int>(i)) = p;
		    }
		    matches_info.H = cv::findFundamentalMat(src_points, dst_points, cv::RANSAC, 3.0, 0.99, matches_info.inliers_mask);
		    if (matches_info.H.empty())// || std::abs(determinant(matches_info.H)) < std::numeric_limits<double>::epsilon())
		        return;
		

		    assert(matches_info.matches.size() == matches_info.inliers_mask.size());

		    for (size_t i = 0; i < matches_info.inliers_mask.size(); ++i) {
		        if (matches_info.inliers_mask[i]) {
		        	if(inliers0) inliers0->push_back(matches_info.matches[i].queryIdx);
		        	if(inliers1) inliers1->push_back(matches_info.matches[i].trainIdx);

		            matches_info.num_inliers++;
		        }
		    }

		    matches_info.confidence = matches_info.num_inliers / (8 + 0.3 * matches_info.matches.size());
		    return;	
	    }	catch(cv::Exception e) {
	    	return;
	    }
	}

	void geo_verify_h(const cv::Mat &descriptors0, const cv::Mat &points0,
		const cv::Mat &descriptors1, const cv::Mat &points1, cv::detail::MatchesInfo &matches_info,
		std::vector<uint32_t> *inliers0, std::vector<uint32_t> *inliers1) {

	    cv::Ptr<cv::flann::IndexParams> indexParams = cv::makePtr<cv::flann::KDTreeIndexParams>();
	    cv::Ptr<cv::flann::SearchParams> searchParams = cv::makePtr<cv::flann::SearchParams>();
	    matches_info.confidence = 0;
	    matches_info.num_inliers = 0;
	    matches_info.H = cv::Mat::zeros(cv::Size(3, 3), CV_32FC1);
	    if(descriptors1.rows < 9 || descriptors0.rows < 9 || descriptors0.cols < 128 || descriptors1.cols < 128) return;
	    if (descriptors0.depth() == CV_8U) {
	        indexParams->setAlgorithm(cvflann::FLANN_INDEX_LSH);
	        searchParams->setAlgorithm(cvflann::FLANN_INDEX_LSH);
	    }
	    try {
		    cv::FlannBasedMatcher matcher(indexParams, searchParams);
		    std::vector< std::vector<cv::DMatch> > pair_matches;
		    // std::cout << "knn match 1" << std::endl;
		    MatchesSet matches;
		    // Find 1->2 matches
		    matcher.knnMatch(descriptors0, descriptors1, pair_matches, 2);
		    float match_conf_ = 0.2f;
		    for (size_t i = 0; i < pair_matches.size(); ++i) {
		        if (pair_matches[i].size() < 2) continue;
		        const cv::DMatch& m0 = pair_matches[i][0];
		        const cv::DMatch& m1 = pair_matches[i][1];
		        if (m0.distance < (1.f - match_conf_) * m1.distance) {
		            matches_info.matches.push_back(m0);
		            matches.insert(std::make_pair(m0.queryIdx, m0.trainIdx));
		        }
		    }

		    pair_matches.clear();
		    matcher.knnMatch(descriptors1, descriptors0, pair_matches, 2);
		    for (size_t i = 0; i < pair_matches.size(); ++i)
		    {
		        if (pair_matches[i].size() < 2)
		            continue;
		        const cv::DMatch& m0 = pair_matches[i][0];
		        const cv::DMatch& m1 = pair_matches[i][1];
		        if (m0.distance < (1.f - match_conf_) * m1.distance)
		            if (matches.find(std::make_pair(m0.trainIdx, m0.queryIdx)) == matches.end()) // if we haven't already added this pair
		                matches_info.matches.push_back(cv::DMatch(m0.trainIdx, m0.queryIdx, m0.distance));
		    }

		    int num_matches_thresh1_ = 9;
		    // int num_matches_thresh2_ = 9;

		    if (matches_info.matches.size() < static_cast<size_t>(num_matches_thresh1_))
		        return;

		    cv::Mat src_points(1, static_cast<int>(matches_info.matches.size()), CV_32FC2);
		    cv::Mat dst_points(1, static_cast<int>(matches_info.matches.size()), CV_32FC2);
		    std::vector<uint32_t> src_idx, dst_idx;

		    for (size_t i = 0; i < matches_info.matches.size(); ++i)   {
		        const cv::DMatch& m = matches_info.matches[i];

		        cv::Point2f p(points0.at<float>(m.queryIdx, 0), points0.at<float>(m.queryIdx, 1));
		        src_points.at<cv::Point2f>(0, static_cast<int>(i)) = p;


		        p = cv::Point2f(points1.at<float>(m.trainIdx, 0), points1.at<float>(m.trainIdx, 1));
		        dst_points.at<cv::Point2f>(0, static_cast<int>(i)) = p;
		    }

		    matches_info.H = cv::findHomography(src_points, dst_points, matches_info.inliers_mask, cv::RANSAC);
		    if (matches_info.H.empty() || std::abs(determinant(matches_info.H)) < std::numeric_limits<double>::epsilon())
		        return;
		

		    assert(matches_info.matches.size() == matches_info.inliers_mask.size());

		    for (size_t i = 0; i < matches_info.inliers_mask.size(); ++i) {
		        if (matches_info.inliers_mask[i]) {
		        	if(inliers0) inliers0->push_back(matches_info.matches[i].queryIdx);
		        	if(inliers1) inliers1->push_back(matches_info.matches[i].trainIdx);

		            matches_info.num_inliers++;
		        }
		    }

		    matches_info.confidence = matches_info.num_inliers / (8 + 0.3 * matches_info.matches.size());
		    return;	
	    }	catch(cv::Exception e) {
	    	return;
	    }
	}
	
	bool compute_sparse_sift_feature(const cv::Mat &img, const PTR_LIB::shared_ptr<const SIFTParams> &params ,
		cv::Mat &keypoints, cv::Mat &descriptors) {
		PTR_LIB::shared_ptr<const SIFTParams> sift_parameters;
		
		if(params == nullptr) {
			sift_parameters = PTR_LIB::make_shared<const SIFTParams>();
		}

		cv::SIFT sift_extractor(sift_parameters->max_features, sift_parameters->num_octave_layers,
				 sift_parameters->contrast_threshold, sift_parameters->edge_threshold, sift_parameters->sigma);

		cv::Mat descriptors_f;

		if(img.size().area() > 0) {
			
			std::vector<cv::KeyPoint> keypoint_vec;
			sift_extractor.detect(img, keypoint_vec);
			sift_extractor.compute(img, keypoint_vec, descriptors_f);
			descriptors_f.convertTo(descriptors, CV_8UC1);
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
			cv::Mat& bow_descriptors, PTR_LIB::shared_ptr< std::vector<std::vector<uint32_t> > > cluster_indices) {

		int clusterCount = matcher->getTrainDescriptors()[0].rows;

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

	cv::Mat merge_descriptors(std::vector<cv::Mat> &descriptors, bool release_original) {
		uint64_t descriptor_count = 0;
		for (size_t i = 0; i < descriptors.size(); i++) {
			descriptor_count += descriptors[i].rows;
		}

		cv::Mat merged(descriptor_count, descriptors[0].cols, descriptors[0].type());
		for (size_t i = 0, start = 0; i < descriptors.size(); i++) {
			cv::Mat submut = merged.rowRange((int)start, (int)(start + descriptors[i].rows));
			descriptors[i].copyTo(submut);
			if (release_original) descriptors[i].release();
			start += descriptors[i].rows;
		}

		return merged;
	}

	bool is_good_match(const cv::detail::MatchesInfo &matches_info) {
		return matches_info.num_inliers >= 16 && matches_info.confidence > 0.7;
	}
}