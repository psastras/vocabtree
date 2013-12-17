#pragma once

#include <search/search_base/search_base.hpp>

/// Implements a Bag of Words based (BoW) image search.  Note that search here is not implemented 
/// and would throw an error should you try to call it.  A naive implementation would have to compute
/// tf-idf with all possible image.  Instead, you should train a BoW model and
/// use this model in conjuction with a InvertedIndex search model to perform a query.
class BagOfWords : public SearchBase {
public:

	/// Subclass of train params base which specifies inverted index training parameters.
	struct TrainParams : public TrainParamsBase {
		TrainParams(uint32_t numClusters = 512, uint32_t numFeatures = 0) :
			numClusters(numClusters), numFeatures(numFeatures) { }
		uint32_t numClusters; // k number of clusters
		uint32_t numFeatures; // number of features to cluster
	};

	/// Subclass of train params base which specifies inverted index training parameters.
	struct SearchParams : public SearchParamsBase {
		
	};

	/// Subclass of match results base which also returns scores
	struct MatchResults : public MatchResultsBase {
		std::vector<float> tfidf_scores;
	};

	BagOfWords();
	BagOfWords(const std::string &file_path);

	/// Given a set of training parameters, list of images, trains.  Returns true if successful, false
	/// if not successful.
	bool train(Dataset &dataset, const PTR_LIB::shared_ptr<const TrainParamsBase> &params,
		 		const std::vector< PTR_LIB::shared_ptr<const Image > > &examples);

	/// Loads a trained search structure from the input filepath
	bool load (const std::string &file_path);

	/// Saves a trained search structure to the input filepath
	bool save (const std::string &file_path) const;

	/// Given a set of search parameters, a query image, searches for matching images and returns the match.
	/// Search is not valid for bag of words - this would require computing tf-idf on all possible images in the dataset, 
	/// and this function will assert(0) should you try to run it.  Instead, you should train a Bag of Words (BoW) model
	/// and use it with one of the other search mechanisms, such as the inverted index.
	PTR_LIB::shared_ptr<MatchResultsBase> search(Dataset &dataset, const PTR_LIB::shared_ptr<const SearchParamsBase> &params, const PTR_LIB::shared_ptr<const Image > &example);

  	std::vector< PTR_LIB::shared_ptr<MatchResultsBase> > search(Dataset &dataset, const PTR_LIB::shared_ptr<SearchParamsBase> &params,
    const std::vector< PTR_LIB::shared_ptr<const Image > > &examples);

	/// Returns the vocabulary matrix.
	cv::Mat vocabulary() const;

	/// Returns the number of clusters in the vocabulary.
	uint32_t num_clusters() const;

protected:
	
	cv::Mat vocabulary_matrix;
};