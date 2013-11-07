#include "search_base.hpp"

SearchBase::SearchBase() { }
SearchBase::SearchBase(const std::string &file_path) { }

SearchBase::~SearchBase() { }

std::vector< std::shared_ptr<MatchResultsBase> > SearchBase::search(Dataset &dataset, const std::shared_ptr<SearchParamsBase> &params,
															 const std::vector< std::shared_ptr<const Image > > &examples) {
	std::vector< std::shared_ptr<MatchResultsBase> > all_matches;
	for(size_t i=0; i<examples.size(); i++) {
		all_matches.push_back(search(dataset, params, examples[i]));
	}
	return all_matches;
}