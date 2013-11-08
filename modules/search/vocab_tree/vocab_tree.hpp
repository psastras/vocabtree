#pragma once

#include <search/search_base/search_base.hpp>

class VocabTree : public SearchBase {
public:

	// Subclass of train params base which specifies inverted index training parameters.
	struct TrainParams : public TrainParamsBase {
		uint32_t depth; // tree depth
		uint32_t split; // number of children per node
	};

	// Subclass of train params base which specifies inverted index training parameters.
	struct SearchParams : public SearchParamsBase {
		
	};

	// Subclass of match results base which also returns scores
	struct MatchResults : public MatchResultsBase {
		std::vector<float> tfidf_scores;
	};

	VocabTree();

	// Given a set of training parameters, list of images, trains.  Returns true if successful, false
	// if not successful.

	bool train(Dataset &dataset, const std::shared_ptr<const TrainParamsBase> &params,
		 		const std::vector< std::shared_ptr<const Image > > &examples);

	// Loads a trained search structure from the input filepath
	bool load (const std::string &file_path);

	// Saves a trained search structure to the input filepath
	bool save (const std::string &file_path) const;

	// Given a set of search parameters, a query image, searches for matching images and returns the match
	std::shared_ptr<MatchResultsBase> search(Dataset &dataset, const std::shared_ptr<const SearchParamsBase> &params, const std::shared_ptr<const Image > &example);

protected:

  struct TreeNode {
    uint32_t invertedFileLength;
    cv::Mat mean;
    std::vector<TreeNode> children;
  };

  TreeNode root;

  // Recursively builds a tree, starting with 0 and ending with currLevel = maxLevel-1
  void buildTreeRecursive(TreeNode t, cv::Mat descriptors, int split, cv::TermCriteria tc, int attempts, int flags, 
    int currLevel, int maxLevel);
	
};