#pragma once

#include <search/search_base/search_base.hpp>

class InvertedIndex : public SearchBase {
public:

	// Subclass of train params base which specifies inverted index training parameters.
	struct TrainParams : public TrainParamsBase {
		uint32_t numClusters; // k number of clusters
		uint32_t numFeatures; // number of features to cluster
	};

	// Subclass of train params base which specifies inverted index training parameters.
	struct SearchParams : public SearchParamsBase {
		
	};

	// Subclass of match results base which also returns scores
	struct MatchResults : public MatchResultsBase {
		std::vector<float> tfidf_scores;
	};

	InvertedIndex();

	// Given a set of training parameters, list of images, trains.  Returns true if successful, false
	// if not successful.

	bool train (const std::shared_ptr<const TrainParamsBase> &params,
		 		const std::vector< std::shared_ptr<const Image > > &examples);

	// Loads a trained search structure from the input filepath
	bool load (const std::string &file_path);

	// Saves a trained search structure to the input filepath
	bool save (const std::string &file_path) const;

	// Given a set of search parameters, a query image, searches for matching images and returns the match
	std::shared_ptr<MatchResultsBase> search (const std::shared_ptr<const SearchParamsBase> &params, const std::shared_ptr<const Image > &example);

protected:
	
};