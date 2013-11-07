#include "inverted_index.hpp"

#include <config.hpp>

#include <utils/filesystem.hpp>
#include <utils/numerics.hpp>

#include <iostream>
#include <fstream>

InvertedIndex::InvertedIndex() : SearchBase() {


}

InvertedIndex::InvertedIndex(const std::string &file_name) : SearchBase(file_name) {
	if(!filesystem::file_exists(file_name)) {
		std::cerr << "Error reading index from " << file_name << std::endl;
		return;
	}
	if(!this->load(file_name)) {
		std::cerr << "Error reading index from " << file_name << std::endl;
	}
}

bool InvertedIndex::load (const std::string &file_path) {
	std::cout << "Reading inverted index from " << file_path << "..." << std::endl;

	std::ifstream ifs(file_path, std::ios::binary);
	uint32_t num_clusters;
	ifs.read((char *)&num_clusters, sizeof(uint32_t));
	inverted_index.resize(num_clusters);
	for(uint32_t i=0; i<num_clusters; i++) {
		uint64_t num_entries;
		ifs.read((char *)&num_entries, sizeof(uint64_t));
		inverted_index[i].resize(num_entries);
		ifs.read((char *)&inverted_index[i][0], sizeof(uint64_t) * num_entries);
	}

	std::cout << "Done reading inverted index." << std::endl;
	
	return (ifs.rdstate() & std::ifstream::failbit) == 0;
}


bool InvertedIndex::save (const std::string &file_path) const {
	std::cout << "Writing inverted index to " << file_path << "..." << std::endl;

	std::ofstream ofs(file_path, std::ios::binary | std::ios::trunc);

	uint32_t num_clusters = inverted_index.size();
	ofs.write((const char *)&num_clusters, sizeof(uint32_t));
	for(uint32_t i=0; i<num_clusters; i++) {
		uint64_t num_entries = inverted_index[i].size();
		ofs.write((const char *)&num_entries, sizeof(uint64_t));
		ofs.write((const char *)&inverted_index[i][0], sizeof(uint64_t) * num_entries);
	}

	std::cout << "Done writing inverted index." << std::endl;

	return (ofs.rdstate() & std::ofstream::failbit) == 0;
}

bool InvertedIndex::train(Dataset &dataset, const std::shared_ptr<const TrainParamsBase> &params, const std::vector< std::shared_ptr<const Image > > &examples) {
	const std::shared_ptr<const TrainParams> &ii_params = std::static_pointer_cast<const TrainParams>(params);
	
	const std::shared_ptr<BagOfWords> &bag_of_words = ii_params->bag_of_words;
	
	if(bag_of_words == nullptr) return false;

	inverted_index.resize(bag_of_words->num_clusters());

	for (size_t i = 0; i < examples.size(); i++) {
		const std::shared_ptr<const Image> &image = examples[i];
		const std::string &bow_descriptors_location = dataset.location(image->feature_path("bow_descriptors"));

		if (!filesystem::file_exists(bow_descriptors_location)) continue;

		numerics::sparse_vector_t bow_descriptors;
		if(!filesystem::load_sparse_vector(bow_descriptors_location, bow_descriptors)) continue;

		for(size_t j=0; j<bow_descriptors.size(); j++) {
			inverted_index[bow_descriptors[j].first].push_back(image->id);
		}
	}

	return true;
}

std::shared_ptr<MatchResultsBase> InvertedIndex::search(Dataset &dataset, const std::shared_ptr<const SearchParamsBase> &params, const std::shared_ptr<const Image > &example) {
	std::cout << "Searching for matching images..." << std::endl;

	// const std::shared_ptr<const SearchParams> &ii_params = std::static_pointer_cast<const SearchParams>(params);
	
	std::shared_ptr<MatchResults> match_result = std::make_shared<MatchResults>();

	const std::string &example_bow_descriptors_location = dataset.location(example->feature_path("bow_descriptors"));
	if (!filesystem::file_exists(example_bow_descriptors_location)) return nullptr;
	numerics::sparse_vector_t example_bow_descriptors;
	if(!filesystem::load_sparse_vector(example_bow_descriptors_location, example_bow_descriptors)) return nullptr;

	std::vector<uint64_t> candidates(dataset.num_images(), 0);
	uint64_t num_candidates = 0;
	for(size_t i=0; i<example_bow_descriptors.size(); i++) {
		uint32_t cluster = example_bow_descriptors[i].first;
		for(size_t j=0; j<inverted_index[cluster].size(); j++) {
			if(!candidates[inverted_index[cluster][j]])
				candidates[inverted_index[cluster][j]] = ++num_candidates;
		}
	}

	std::vector<float> fake_idfw(inverted_index.size(), 1.f); // sets idf weights to one.

	std::vector< std::pair<float, uint64_t> > candidate_scores(num_candidates);

#if ENABLE_MULTITHREADING && ENABLE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
	for(int64_t i=0; i<candidates.size(); i++) {
		if(!candidates[i]) continue;

		const std::string &bow_descriptors_location = dataset.location(dataset.image(i)->feature_path("bow_descriptors"));
		if (!filesystem::file_exists(bow_descriptors_location)) continue;
		numerics::sparse_vector_t bow_descriptors;
		if(!filesystem::load_sparse_vector(bow_descriptors_location, bow_descriptors)) continue;

		float sim = numerics::cos_sim(example_bow_descriptors, bow_descriptors, fake_idfw);
		candidate_scores[candidates[i]-1] = std::pair<float, uint64_t>(sim, i);
	}

	std::sort(candidate_scores.begin(), candidate_scores.end(), 
          boost::bind(&std::pair<float, uint64_t>::first, _1) >
          boost::bind(&std::pair<float, uint64_t>::first, _2));

	match_result->tfidf_scores.resize(candidate_scores.size());
	match_result->matches.resize(candidate_scores.size());
	
	for(int64_t i=0; i<candidate_scores.size(); i++) {
		match_result->tfidf_scores[i] = candidate_scores[i].first;
		match_result->matches[i] = candidate_scores[i].second;
	}

	return std::static_pointer_cast<MatchResultsBase>(match_result);
}

std::ostream& operator<< (std::ostream &out, const InvertedIndex::MatchResults &match_results) {
	out << "[ ";
	for(uint32_t i=0; i<MIN(8, match_results.matches.size()); i++) {
		out << "[ " << match_results.matches[i] << ", " <<  match_results.tfidf_scores[i] << " ] ";
	}
	out << "]";
	return out;
}