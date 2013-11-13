#pragma once

#include <search/search_base/search_base.hpp>
#include <unordered_map>
#include <unordered_set>

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

    uint32_t level; // range from 0..maxLevel-1
    // index for this level, ranging from 0..split^level-1
    // For example: the first level children will simply have indexes from 0..split-1
    //   for the second level the children of the first child will have 0..split-1
    //   while the children of the second child we have split..2*split-1
    // This will be used to identify the node and used to index into the vectors for images
    uint32_t levelIndex; 

    // index in a level order traversal of the tree
    uint32_t index;
    cv::Mat mean;
    // index into the array of nodes of the first child, all children are next to eachother
    // if this is < 0 then it is a leaf
    uint32_t firstChildIndex;
  };

  // stores the amount of splits used to generate tree
  uint32_t split;
  // Stores the max level of the tree
  // Right now just set it here, but should have it as an input
  uint32_t maxLevel = 6;
  // number of nodes the tree will have, saved in variable so don't have to recompute
  uint32_t numberOfNodes;

  std::vector<float> weights;

  std::vector<TreeNode> tree;
  std::vector<std::unordered_map<uint64_t, uint32_t>> invertedFiles;

  // Stores the database vectors for all images in the database - d_i in the paper
  // Indexes by the image id
  std::unordered_map<uint64_t, std::vector<float>> databaseVectors;

  // Recursively builds a tree, starting with 0 and ending with currLevel = maxLevel-1
  void buildTreeRecursive(uint32_t t, cv::Mat descriptors, cv::TermCriteria tc, int attempts, int flags, int currLevel);

  // helper function, inserts a dummy possibleMatches
  std::vector<float> generateVector(cv::Mat descriptors, bool shouldWeight, uint64_t id = -1);

  // To call with an id call without possibleMatches and it will go to the helper function
  // Takes descriptors for an image and for each descriptor finds the path down the tree generating a vector (describing the path)
  // Adds up all vectors (one from each descriptor) to return the vector of counts for each node
  // If  shouldWeight is true will weight each by the weight of the node, should be true for general query and false for construction
  // If id is set then will insert that id into the invertedFile of each leaf visited, if negative or not set then won't do anything
  // When id is not set will use insert images into possibleMatches, possibleMatches will not be used if id is set
  std::vector<float> generateVector(cv::Mat descriptors, bool shouldWeight, std::unordered_set<uint64_t> & possibleMatches, uint64_t id = -1);

  // Recursive function that recursively goes down the tree from t to find where the single descriptor belongs (stopping at leaf)
  // On each node increments cound in the counts vector
  // If id is set (>=0) then adds the image with that id to the leaf
  // Picks the child to traverse down based on the max dot product
  void generateVectorHelper(uint32_t nodeIndex, cv::Mat descriptor, std::vector<float> & counts,
    std::unordered_set<uint64_t> & possibleMatches, uint64_t id = -1);
	
};