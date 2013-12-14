#include "inverted_index.hpp"

#include <config.hpp>

#include <utils/filesystem.hpp>
#include <utils/numerics.hpp>

#include <iostream>
#include <fstream>

#if ENABLE_MULTITHREADING && ENABLE_MPI
#include <mpi.h>
#endif

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
	idf_weights.resize(num_clusters);
	ifs.read((char *)&idf_weights[0], sizeof(float) * num_clusters);
	for(uint32_t i=0; i<num_clusters; i++) {
		uint64_t num_entries;
		ifs.read((char *)&num_entries, sizeof(uint64_t));
		inverted_index[i].resize(num_entries);
    if (num_entries != 0)
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
	ofs.write((const char *)&idf_weights[0], sizeof(float) * num_clusters);
	for(uint32_t i=0; i<num_clusters; i++) {
		uint64_t num_entries = inverted_index[i].size();
		ofs.write((const char *)&num_entries, sizeof(uint64_t));
    if (num_entries != 0)
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
	idf_weights.resize(bag_of_words->num_clusters(), 0.f);

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

	for(size_t i=0; i<idf_weights.size(); i++) {
		idf_weights[i] = logf(
				(float)examples.size() /
				(float)inverted_index[i].size());
	}

	return true;
}

std::vector< std::shared_ptr<MatchResultsBase> > InvertedIndex::search(Dataset &dataset, const std::shared_ptr<SearchParamsBase> &params,
															 const std::vector< std::shared_ptr<const Image > > &examples) {
std::vector< std::shared_ptr<MatchResultsBase> > match_results(examples.size());
	#pragma omp parallel for schedule(dynamic)
	for(int64_t i=0; i<(int64_t)examples.size(); i++) {
		match_results[i] = this->search(dataset, params, examples[i]);
	}
	return match_results;
}

std::shared_ptr<MatchResultsBase> InvertedIndex::search(Dataset &dataset, const std::shared_ptr<const SearchParamsBase> &params, 
	const std::shared_ptr<const Image > &example) {

	const std::shared_ptr<const SearchParams> &ii_params = params == nullptr ?
		std::make_shared<const SearchParams>()
		: std::static_pointer_cast<const SearchParams>(params);
	
	std::shared_ptr<MatchResults> match_result = std::make_shared<MatchResults>();

	const numerics::sparse_vector_t &example_bow_descriptors = dataset.load_bow_feature(
		example->id
	);

	std::vector<std::pair<uint64_t, uint64_t> > candidates(dataset.num_images(), std::pair<uint64_t, uint64_t>(0, 0));
	uint64_t num_candidates = 0; // number of matches > 0
	for(size_t i=0; i<example_bow_descriptors.size(); i++) {
		uint32_t cluster = example_bow_descriptors[i].first;
		for(size_t j=0; j<inverted_index[cluster].size(); j++) {
			uint64_t id = inverted_index[cluster][j];
			if(!candidates[id].first) {
				candidates[id].second = id;
				++num_candidates;
			}
			candidates[id].first++;
			
		}
	}

	std::sort(candidates.begin(), candidates.end());
	std::reverse(candidates.begin(), candidates.end());
	
	num_candidates = MIN(num_candidates, ii_params->cutoff_idx);

  if (num_candidates == 0)
    return match_result;

  std::vector< std::pair<float, uint64_t> > candidate_scores(num_candidates);

#if ENABLE_MULTITHREADING && ENABLE_MPI
  int rank, procs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &procs);

  /// number of candidates each node has
  int candidatesPerProc = (int)(((float)num_candidates) / ((float)procs)) + 1;
  int leftover = num_candidates%candidatesPerProc;
  /// actual number of candidates a node has, will only be different for last node if num_candidates % procs !=0
  int myCandidates = (rank==procs-1)?leftover : candidatesPerProc;
#endif


#if ENABLE_MULTITHREADING && ENABLE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
#if ENABLE_MULTITHREADING && ENABLE_MPI
  for (int64_t i = rank*candidatesPerProc; i<(rank*candidatesPerProc + myCandidates); i++) {
#else
	for(int64_t i=0; i<num_candidates; i++) {
#endif

		const numerics::sparse_vector_t &bow_descriptors = dataset.load_bow_feature(
				candidates[i].second
			);

		float sim = numerics::min_hist(example_bow_descriptors, bow_descriptors, idf_weights);
		candidate_scores[i] = std::pair<float, uint64_t>(sim, candidates[i].second);
	}

	// aggregate all results into node zero.
#if ENABLE_MULTITHREADING && ENABLE_MPI
  if (rank == 0) {
    // recieve all candidates from other nodes
    std::vector<MPI_Request> requests(procs - 1);
    for (int p = 1; p < procs; p++) {
      int num = (p == procs - 1) ? leftover : candidatesPerProc;
      //if (p == procs - 1 && num_candidates%procs != 0)
        //num = num_candidates - ((procs - 1)*candidatesPerProc);
      MPI_Irecv(&candidate_scores[p*candidatesPerProc], (sizeof(float)+sizeof(uint64_t))*num, MPI_BYTE, p, p, MPI_COMM_WORLD, &requests[p - 1]);
    }
    MPI_Waitall(requests.size(), &requests[0], MPI_STATUSES_IGNORE);
  }
  else {
    // send candidate information to root, return empty
    MPI_Send(&candidate_scores[rank*candidatesPerProc], (sizeof(float)+sizeof(uint64_t))*myCandidates,
      MPI_BYTE, 0, rank, MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank != 0)
    return std::static_pointer_cast<MatchResultsBase>(match_result); // will be empty
#endif

	std::sort(candidate_scores.begin(), candidate_scores.end(), 
          boost::bind(&std::pair<float, uint64_t>::first, _1) >
          boost::bind(&std::pair<float, uint64_t>::first, _2));

	match_result->tfidf_scores.resize(candidate_scores.size());
	match_result->matches.resize(candidate_scores.size());
	
	for(int64_t i=0; i<(int64_t)candidate_scores.size(); i++) {
		match_result->tfidf_scores[i] = candidate_scores[i].first;
		match_result->matches[i] = candidate_scores[i].second;
	}

	return std::static_pointer_cast<MatchResultsBase>(match_result);
}

uint32_t InvertedIndex::num_clusters() const {
	return idf_weights.size();
}

std::ostream& operator<< (std::ostream &out, const InvertedIndex::MatchResults &match_results) {
	out << "[ ";
	for(uint32_t i=0; i<MIN(8, match_results.matches.size()); i++) {
		out << "[ " << match_results.matches[i] << ", " <<  match_results.tfidf_scores[i] << " ] ";
	}
	out << "]";
	return out;
}