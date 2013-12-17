#pragma once

#include <search/search_base/search_base.hpp>
#include <search/bag_of_words/bag_of_words.hpp>

/// Implements a Bag of Words based (BoW) image search using an inverted index.  The inverted
/// index keeps track of a list of images associated with each visual word.  The images are
/// represented as an unsigned long long which must then be translated back to an actual image
/// using the appropriate Dataset class implementation.
class InvertedIndex : public SearchBase {
public:

	/// Subclass of train params base which specifies inverted index training parameters.
	struct TrainParams : public TrainParamsBase {
		PTR_LIB::shared_ptr<BagOfWords> bag_of_words;  /// bag of words to index on
	};

	/// Subclass of train params base which specifies inverted index training parameters.
	struct SearchParams : public SearchParamsBase {
		SearchParams(uint64_t cutoff_idx = 4096) : cutoff_idx(cutoff_idx) { }

		uint64_t cutoff_idx; /// number of top matches to consider
	};

	/// Subclass of match results base which also returns scores
	struct MatchResults : public MatchResultsBase {
		std::vector<float> tfidf_scores;
	};

	InvertedIndex();
	InvertedIndex(const std::string &file_name);

	/// Given a set of training parameters, list of images, trains.  Returns true if successful, false
	/// if not successful.
	bool train(Dataset &dataset, const PTR_LIB::shared_ptr<const TrainParamsBase> &params,
		 		const std::vector< PTR_LIB::shared_ptr<const Image > > &examples);

	/// Loads a trained search structure from the input filepath
	bool load (const std::string &file_path);

	/// Saves a trained search structure to the input filepath
	bool save (const std::string &file_path) const;

	/// Returns the number of clusters used in the inverted index descriptors
	uint32_t num_clusters() const;

	/// Given a set of search parameters, a query image, searches for matching images and returns the match.  If the match is 0, then the search failed 
	/// (it will fail if the example image has missing features).
	PTR_LIB::shared_ptr<MatchResultsBase> search(Dataset &dataset, const PTR_LIB::shared_ptr<const SearchParamsBase> &params, const PTR_LIB::shared_ptr<const Image > &example);
	
	/// Given a set of search parameters, a query image, searches for matching images and returns the match.  If the match is 0, then the search failed 
	/// (it will fail if the example image has missing features).
	std::vector< PTR_LIB::shared_ptr<MatchResultsBase> > search(Dataset &dataset, const PTR_LIB::shared_ptr<SearchParamsBase> &params,
															 const std::vector< PTR_LIB::shared_ptr<const Image > > &examples);

protected:
	
	std::vector< std::vector<uint64_t> > inverted_index; /// Stores the inverted index, dimension one is the cluster index, dimension two holds a list of ids containing that word.
	std::vector<float> idf_weights; /// Stores the idf weights, one element per cluster

};

/// Prints out information about the match results.
std::ostream& operator<< (std::ostream &out, const InvertedIndex::MatchResults &match_results);
