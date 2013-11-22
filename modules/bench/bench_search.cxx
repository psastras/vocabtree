#include "bench_config.hpp"

#include <fstream>

#include <config.hpp>
#include <utils/filesystem.hpp>
#include <utils/numerics.hpp>
#include <utils/dataset.hpp>
#include <utils/misc.hpp>
#include <utils/vision.hpp>
#include <utils/logger.hpp>
#include <utils/cycletimer.hpp>
#include <search/bag_of_words/bag_of_words.hpp>
#include <search/inverted_index/inverted_index.hpp>
#include <vis/matches_page.hpp>

#if ENABLE_MULTITHREADING && ENABLE_OPENMP
#include <omp.h>
#endif
#if ENABLE_MULTITHREADING && ENABLE_MPI
#include <mpi.h>
#endif

_INITIALIZE_EASYLOGGINGPP

void bench_oxford() {

	MatchesPage html_output;
	
	const uint32_t num_clusters = s_oxford_num_clusters;

	SimpleDataset oxford_dataset(s_oxford_data_dir, s_oxford_database_location);
	LINFO << oxford_dataset;

	std::stringstream index_output_file;
	index_output_file << oxford_dataset.location() << "/index/" << num_clusters << ".index";
	InvertedIndex ii(index_output_file.str());

	double total_time = 0.0;
	uint32_t total_iterations = 256;
	for(uint32_t i=0; i<total_iterations; i++) {
		double start_time = CycleTimer::currentSeconds();
		std::shared_ptr<InvertedIndex::MatchResults> matches = 
		std::static_pointer_cast<InvertedIndex::MatchResults>(ii.search(oxford_dataset, nullptr, oxford_dataset.image(i)));	
		if(matches == nullptr) {
			LERROR << "Error while running search.";
			continue;
		}
		double end_time = CycleTimer::currentSeconds();
		total_time += (end_time - start_time);
		html_output.add_match(i, matches->matches, oxford_dataset);
	}

	html_output.write(oxford_dataset.location() + "/results/matches/");


	// Write out the timings
	std::stringstream timings_file_name;
	timings_file_name << oxford_dataset.location() + "/results/times.index." <<  ii.num_clusters() << ".csv";
	std::ofstream ofs(timings_file_name.str(), std::ios::app);
	if(ofs.tellp() == 0) {
		std::stringstream header;
		header << "machine\ttime(s)\titerations\tmultithreading\topenmp\tmpi\t" << std::endl;
		ofs.write(header.str().c_str(), header.str().size());
	}
	std::stringstream timing;
	timing << misc::get_machine_name() << "\t" << total_time << "\t" << total_iterations << "\t" << 
			  ENABLE_MULTITHREADING << "\t" << ENABLE_OPENMP << 
			  "\t" << ENABLE_MPI << "\t" << std::endl;
	ofs.write(timing.str().c_str(), timing.str().size());
	ofs.close();

}

int main(int argc, char *argv[]) {
#if ENABLE_MULTITHREADING && ENABLE_MPI
	MPI::Init(argc, argv);
#endif

	bench_oxford();

#if ENABLE_MULTITHREADING && ENABLE_MPI
	MPI::Finalize();
#endif
	return 0;
}