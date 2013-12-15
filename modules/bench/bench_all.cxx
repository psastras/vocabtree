#include "bench_config.hpp"

#include <fstream>

#include <config.hpp>
#include <utils/filesystem.hpp>
#include <utils/numerics.hpp>
#include <utils/dataset.hpp>
#include <utils/misc.hpp>
#include <utils/vision.hpp>
#include <utils/logger.hpp>
#include <utils/image.hpp>
#include <utils/cycletimer.hpp>
#include <search/bag_of_words/bag_of_words.hpp>
#include <search/inverted_index/inverted_index.hpp>
#include <search/vocab_tree/vocab_tree.hpp>
#include <vis/matches_page.hpp>

#if ENABLE_MULTITHREADING && ENABLE_OPENMP
#include <omp.h>
#endif
#if ENABLE_MULTITHREADING && ENABLE_MPI
#include <mpi.h>
#endif

_INITIALIZE_EASYLOGGINGPP

void compute_features(Dataset &dataset) {
#if ENABLE_MULTITHREADING && ENABLE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
	for (int64_t i = 0; i < dataset.num_images(); i++) {

		PTR_LIB::shared_ptr<SimpleDataset::SimpleImage> image = std::static_pointer_cast<SimpleDataset::SimpleImage>(dataset.image(i));
		if (image == 0) continue;

		const std::string &keypoints_location = dataset.location(image->feature_path("keypoints"));
		const std::string &descriptors_location = dataset.location(image->feature_path("descriptors"));
		if (filesystem::file_exists(keypoints_location) && filesystem::file_exists(descriptors_location)) continue;

		const std::string &image_location = dataset.location(image->location());

		if (!filesystem::file_exists(image_location)) continue;

		cv::Mat im = cv::imread(image_location, cv::IMREAD_GRAYSCALE);

		cv::Mat keypoints, descriptors;
		if (!vision::compute_sparse_sift_feature(im, 0, keypoints, descriptors)) continue;

		filesystem::create_file_directory(keypoints_location);
		filesystem::create_file_directory(descriptors_location);

		filesystem::write_cvmat(keypoints_location, keypoints);
		filesystem::write_cvmat(descriptors_location, descriptors);
	}
}

void compute_bow_features(Dataset &dataset, PTR_LIB::shared_ptr<BagOfWords> bow, uint32_t num_clusters) {
	const std::vector<  PTR_LIB::shared_ptr<const Image> > &all_images = dataset.all_images();
#if ENABLE_MULTITHREADING && ENABLE_OPENMP
	uint32_t num_threads = omp_get_max_threads();
	std::vector< cv::Ptr<cv::DescriptorMatcher> > matchers;
	for (uint32_t i = 0; i < num_threads; i++) {
		matchers.push_back(vision::construct_descriptor_matcher(bow->vocabulary()));
	}
#pragma omp parallel for schedule(dynamic)
#else
	const cv::Ptr<cv::DescriptorMatcher> &matcher = vision::construct_descriptor_matcher(bow->vocabulary());
#endif
	for (int64_t i = 0; i < (int64_t)all_images.size(); i++) {
#if ENABLE_MULTITHREADING && ENABLE_OPENMP
		const cv::Ptr<cv::DescriptorMatcher> &matcher = matchers[omp_get_thread_num()];
#endif
		const std::string &sift_descriptor_location = dataset.location(all_images[i]->feature_path("descriptors"));
		const std::string &bow_descriptor_location = dataset.location(all_images[i]->feature_path("bow_descriptors"));

		cv::Mat descriptors, bow_descriptors, descriptorsf;
		if (!filesystem::file_exists(sift_descriptor_location)) continue;
		if (!filesystem::load_cvmat(sift_descriptor_location, descriptors)) continue;
		descriptors.convertTo(descriptorsf, CV_32FC1);
		filesystem::create_file_directory(bow_descriptor_location);

		if (!vision::compute_bow_feature(descriptorsf, matcher, bow_descriptors, 0)) continue;
		const std::vector< std::pair<uint32_t, float> > &bow_descriptors_sparse = numerics::sparsify(bow_descriptors);
		filesystem::write_sparse_vector(bow_descriptor_location, bow_descriptors_sparse);

		LINFO << "Wrote " << bow_descriptor_location;
	}
}

PTR_LIB::shared_ptr<VocabTree> train_tree(Dataset &dataset, uint32_t num_images, uint32_t split, uint32_t depth) {
	VocabTree vt;
	PTR_LIB::shared_ptr<VocabTree::TrainParams> train_params = PTR_LIB::make_shared<VocabTree::TrainParams>();
	train_params->depth = depth;
	train_params->split = split;
	vt.train(dataset, train_params, dataset.random_images(num_images));

	std::stringstream vocab_output_file;
	vocab_output_file << dataset.location() << "/tree/" << split << "." << depth << ".vocab";
	vt.save(vocab_output_file.str());
	return PTR_LIB::make_shared<VocabTree>(vt);
}

PTR_LIB::shared_ptr<BagOfWords> train_bow(Dataset &dataset, uint32_t num_images, uint32_t num_clusters) {
	BagOfWords bow;
	PTR_LIB::shared_ptr<BagOfWords::TrainParams> train_params = PTR_LIB::make_shared<BagOfWords::TrainParams>();
	train_params->numClusters = num_clusters;
	const std::vector<  PTR_LIB::shared_ptr<const Image> > &random_images = dataset.random_images(num_images);
	bow.train(dataset, train_params, random_images);
	std::stringstream vocab_output_file;
	vocab_output_file << dataset.location() << "/vocabulary/" << train_params->numClusters << ".vocab";
	bow.save(vocab_output_file.str());
	return  PTR_LIB::make_shared<BagOfWords>(bow);
}

PTR_LIB::shared_ptr<InvertedIndex> train_index(Dataset &dataset, PTR_LIB::shared_ptr<BagOfWords> bow) {
	InvertedIndex ii;
	PTR_LIB::shared_ptr<InvertedIndex::TrainParams> train_params = PTR_LIB::make_shared<InvertedIndex::TrainParams>();
	train_params->bag_of_words = bow;
	ii.train(dataset, train_params, dataset.all_images());

	return PTR_LIB::make_shared<InvertedIndex>(ii);
}

void benchmark_dataset(Dataset &dataset) {
	compute_features(dataset); // compute sift features

	// parameters
	uint32_t num_images = 128;

	uint32_t bow_clusters[] = { 256, 3125, 46656 };
	std::pair<uint32_t, uint32_t> tree_branches[] = { 
		std::pair<uint32_t, uint32_t>(4, 4), 
		std::pair<uint32_t, uint32_t>(5, 5), 
		std::pair<uint32_t, uint32_t>(6, 6)};

	std::stringstream timings_file_name;
	timings_file_name << dataset.location() + "/results/times.index.json";
	filesystem::create_file_directory(timings_file_name.str());
	std::ofstream ofs(timings_file_name.str(), std::ios::app);
	// o letsdoit
	for(size_t i=0; i<3; i++) {
		LINFO << "Training bag of words";
		double start_time_bow = CycleTimer::currentSeconds();
		PTR_LIB::shared_ptr<BagOfWords> bow = train_bow(dataset, num_images, bow_clusters[i]);
		double end_time_bow = CycleTimer::currentSeconds();
		{
			std::stringstream timing;
			timing << "{ " <<
				"\"machine\" : \"" << misc::get_machine_name() << "\", " <<
				"\"operation\" : \"" << "bow_train" << "\", " <<
				"\"bow_numclusters\" : " << bow->num_clusters() << ", " <<
				"\"db_size\" : " << dataset.num_images() << ", " <<
				"\"time\" : " << end_time_bow - start_time_bow << ", " <<
				"\"multithreading\" : " << ENABLE_MULTITHREADING << ", " <<
				"\"openmp\" : " << ENABLE_OPENMP << ", " <<
				"\"mpi\" : " << ENABLE_MPI << ", " <<
				"}" << std::endl;
			ofs.write(timing.str().c_str(), timing.str().size());
			ofs.flush();
		}

		LINFO << "Computing bag of words features";
		double start_time_bowfeatures = CycleTimer::currentSeconds();
		compute_bow_features(dataset, bow, bow_clusters[i]);
		double end_time_bowfeatures = CycleTimer::currentSeconds();
		{
			std::stringstream timing;
			timing << "{ " <<
				"\"machine\" : \"" << misc::get_machine_name() << "\", " <<
				"\"operation\" : \"" << "bow_features" << "\", " <<
				"\"bow_numclusters\" : " << bow->num_clusters() << ", " <<
				"\"db_size\" : " << dataset.num_images() << ", " <<
				"\"time\" : " << end_time_bowfeatures - start_time_bowfeatures << ", " <<
				"\"multithreading\" : " << ENABLE_MULTITHREADING << ", " <<
				"\"openmp\" : " << ENABLE_OPENMP << ", " <<
				"\"mpi\" : " << ENABLE_MPI << ", " <<
				"}" << std::endl;
			ofs.write(timing.str().c_str(), timing.str().size());
			ofs.flush();
		}

		LINFO << "Computing index";
		double start_time_index = CycleTimer::currentSeconds();
		PTR_LIB::shared_ptr<InvertedIndex> ii = train_index(dataset, bow);
		double end_time_index = CycleTimer::currentSeconds();
		{
			std::stringstream timing;
			timing << "{ " <<
				"\"machine\" : \"" << misc::get_machine_name() << "\", " <<
				"\"operation\" : \"" << "index_train" << "\", " <<
				"\"index_numclusters\" : " << ii->num_clusters() << ", " <<
				"\"db_size\" : " << dataset.num_images() << ", " <<
				"\"time\" : " << end_time_index - start_time_index << ", " <<
				"\"multithreading\" : " << ENABLE_MULTITHREADING << ", " <<
				"\"openmp\" : " << ENABLE_OPENMP << ", " <<
				"\"mpi\" : " << ENABLE_MPI << ", " <<
				"}" << std::endl;
			ofs.write(timing.str().c_str(), timing.str().size());
			ofs.flush();
		}

		LINFO << "Training tree";
		double start_time_tree = CycleTimer::currentSeconds();
		PTR_LIB::shared_ptr<VocabTree> vt = train_tree(dataset, num_images, tree_branches[i].first, tree_branches[i].second);
		double end_time_tree = CycleTimer::currentSeconds();
		{
			std::stringstream timing;
			timing << "{ " <<
				"\"machine\" : \"" << misc::get_machine_name() << "\", " <<
				"\"operation\" : \"" << "tree_train" << "\", " <<
				"\"tree_depth\" : " << vt->tree_depth() << ", " <<
				"\"tree_split\" : " << vt->tree_splits() << ", " <<
				"\"db_size\" : " << dataset.num_images() << ", " <<
				"\"time\" : " << end_time_tree - start_time_tree << ", " <<
				"\"multithreading\" : " << ENABLE_MULTITHREADING << ", " <<
				"\"openmp\" : " << ENABLE_OPENMP << ", " <<
				"\"mpi\" : " << ENABLE_MPI << ", " <<
				"}" << std::endl;
			ofs.write(timing.str().c_str(), timing.str().size());
			ofs.flush();
		}

		uint32_t num_validate = 10;
		uint32_t total_iterations = MIN(dataset.num_images(), 128);

		LINFO << "Running index search";
		// search index
		{
			MatchesPage html_output_index;
			double total_time = 0.0;
			uint32_t total_correct = 0, total_tested = 0;
			for (uint32_t i = 0; i < total_iterations; i++) {
				double start_time = CycleTimer::currentSeconds();

				PTR_LIB::shared_ptr<SimpleDataset::SimpleImage> query_image = std::static_pointer_cast<SimpleDataset::SimpleImage>(dataset.image(i));
				PTR_LIB::shared_ptr<InvertedIndex::MatchResults> matches_index =
					std::static_pointer_cast<InvertedIndex::MatchResults>(ii->search(dataset, 0, query_image));
				if (matches_index == 0) {
					LERROR << "Error while running search.";
					continue;
				}
				double end_time = CycleTimer::currentSeconds();
				total_time += (end_time - start_time);

				// validate matches
				cv::Mat keypoints_0, descriptors_0;
				const std::string &query_keypoints_location = dataset.location(query_image->feature_path("keypoints"));
				const std::string &query_descriptors_location = dataset.location(query_image->feature_path("descriptors"));
				filesystem::load_cvmat(query_keypoints_location, keypoints_0);
				filesystem::load_cvmat(query_descriptors_location, descriptors_0);
				std::vector<int> validated(MIN(num_validate, matches_index->matches.size()), 0);
				total_tested += validated.size();
				uint32_t total_correct_tmp = 0;
#if ENABLE_MULTITHREADING && ENABLE_OPENMP
#pragma omp parallel for schedule(dynamic) reduction(+:total_correct_tmp)
#endif
				for (int32_t j = 0; j < validated.size(); j++) {
					cv::Mat keypoints_1, descriptors_1;
					PTR_LIB::shared_ptr<SimpleDataset::SimpleImage> match_image = std::static_pointer_cast<SimpleDataset::SimpleImage>(dataset.image(matches_index->matches[j]));
					const std::string &match_keypoints_location = dataset.location(match_image->feature_path("keypoints"));
					const std::string &match_descriptors_location = dataset.location(match_image->feature_path("descriptors"));
					filesystem::load_cvmat(match_keypoints_location, keypoints_1);
					filesystem::load_cvmat(match_descriptors_location, descriptors_1);

					cv::detail::MatchesInfo match_info;
					vision::geo_verify_f(descriptors_0, keypoints_0, descriptors_1, keypoints_1, match_info);

					validated[j] = vision::is_good_match(match_info) ? 1 : -1;
					if (validated[j] > 0) total_correct_tmp++;
				}
				total_correct += total_correct_tmp;
				html_output_index.add_match(i, matches_index->matches, dataset, PTR_LIB::make_shared< std::vector<int> >(validated));

				std::stringstream outfilestr;
				outfilestr << dataset.location() << "/results/matches/index." << bow->num_clusters();
				html_output_index.write(outfilestr.str());
			}

			// Write out the timings
			std::stringstream timing;
			timing << "{ " <<
				"\"machine\" : \"" << misc::get_machine_name() << "\", " <<
				"\"operation\" : \"" << "index_search" << "\", " <<
				"\"index_numclusters\" : " << ii->num_clusters() << ", " <<
				"\"db_size\" : " << dataset.num_images() << ", " <<
				"\"time\" : " << total_time << ", " <<
				"\"iterations\" : " << total_iterations << ", " <<
				"\"correct\" : " << total_correct << ", " <<
				"\"tested\" : " << total_tested << ", " <<
				"\"multithreading\" : " << ENABLE_MULTITHREADING << ", " <<
				"\"openmp\" : " << ENABLE_OPENMP << ", " <<
				"\"mpi\" : " << ENABLE_MPI << ", " <<
				"}" << std::endl;
			ofs.write(timing.str().c_str(), timing.str().size());
			ofs.flush();
		}

		LINFO << "Running tree search";
		// search tree
		{
			MatchesPage html_output_tree;
			double total_time = 0.0;
			uint32_t total_correct = 0, total_tested = 0;
			for (uint32_t i = 0; i < total_iterations; i++) {
				double start_time = CycleTimer::currentSeconds();

				PTR_LIB::shared_ptr<SimpleDataset::SimpleImage> query_image = std::static_pointer_cast<SimpleDataset::SimpleImage>(dataset.image(i));
				PTR_LIB::shared_ptr<InvertedIndex::MatchResults> matches_index =
					std::static_pointer_cast<InvertedIndex::MatchResults>(vt->search(dataset, 0, query_image));
				if (matches_index == 0) {
					LERROR << "Error while running search.";
					continue;
				}
				double end_time = CycleTimer::currentSeconds();
				total_time += (end_time - start_time);

				// validate matches
				cv::Mat keypoints_0, descriptors_0;
				const std::string &query_keypoints_location = dataset.location(query_image->feature_path("keypoints"));
				const std::string &query_descriptors_location = dataset.location(query_image->feature_path("descriptors"));
				filesystem::load_cvmat(query_keypoints_location, keypoints_0);
				filesystem::load_cvmat(query_descriptors_location, descriptors_0);
				std::vector<int> validated(MIN(num_validate, matches_index->matches.size()), 0);
				total_tested += validated.size();
				uint32_t total_correct_tmp = 0;
#if ENABLE_MULTITHREADING && ENABLE_OPENMP
#pragma omp parallel for schedule(dynamic) reduction(+:total_correct_tmp)
#endif
				for (int32_t j = 0; j < validated.size(); j++) {
					cv::Mat keypoints_1, descriptors_1;
					PTR_LIB::shared_ptr<SimpleDataset::SimpleImage> match_image = std::static_pointer_cast<SimpleDataset::SimpleImage>(dataset.image(matches_index->matches[j]));
					const std::string &match_keypoints_location = dataset.location(match_image->feature_path("keypoints"));
					const std::string &match_descriptors_location = dataset.location(match_image->feature_path("descriptors"));
					filesystem::load_cvmat(match_keypoints_location, keypoints_1);
					filesystem::load_cvmat(match_descriptors_location, descriptors_1);

					cv::detail::MatchesInfo match_info;
					vision::geo_verify_f(descriptors_0, keypoints_0, descriptors_1, keypoints_1, match_info);

					validated[j] = vision::is_good_match(match_info) ? 1 : -1;
					if (validated[j] > 0) total_correct_tmp++;
				}
				total_correct += total_correct_tmp;
				html_output_tree.add_match(i, matches_index->matches, dataset, PTR_LIB::make_shared< std::vector<int> >(validated));
				
				std::stringstream outfilestr;
				outfilestr << dataset.location() << "/results/matches/tree." << vt->tree_depth() << "." << vt->tree_splits();
				html_output_tree.write(outfilestr.str());
			}

			// Write out the timings
			std::stringstream timing;
			timing << "{ " <<
				"\"machine\" : \"" << misc::get_machine_name() << "\", " <<
				"\"operation\" : \"" << "tree_search" << "\", " <<
				"\"tree_depth\" : " << vt->tree_depth() << ", " <<
				"\"tree_split\" : " << vt->tree_splits() << ", " <<
				"\"db_size\" : " << dataset.num_images() << ", " <<
				"\"time\" : " << total_time << ", " <<
				"\"iterations\" : " << total_iterations << ", " <<
				"\"correct\" : " << total_correct << ", " <<
				"\"tested\" : " << total_tested << ", " <<
				"\"multithreading\" : " << ENABLE_MULTITHREADING << ", " <<
				"\"openmp\" : " << ENABLE_OPENMP << ", " <<
				"\"mpi\" : " << ENABLE_MPI << ", " <<
				"}" << std::endl;
			ofs.write(timing.str().c_str(), timing.str().size());
			ofs.flush();
		}
	
	}
	ofs.close();
}	

int main(int argc, char *argv[]) {
#if ENABLE_MULTITHREADING && ENABLE_MPI
	MPI::Init(argc, argv);
	int rank = MPI::COMM_WORLD.Get_rank();
	if(rank == 0) {
#endif

	SimpleDataset datasets[] = {
		
		SimpleDataset(s_oxfordmini_data_dir, s_oxfordmini_database_location),
		SimpleDataset(s_holidays_data_dir, s_holidays_database_location),
		SimpleDataset(s_oxford_data_dir, s_oxford_database_location)
	};

	for (size_t i = 0; i < 3; i++) {
		LINFO << datasets[i];
		benchmark_dataset(datasets[i]);
	}


#if ENABLE_MULTITHREADING && ENABLE_MPI
	}
	MPI::Finalize();
#endif
	return 0;
}