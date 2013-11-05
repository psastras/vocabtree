#include "tests_config.hpp"
#include <search/inverted_index/inverted_index.hpp>
#include <iostream>


class SimpleImage : public Image {
	public:
		SimpleImage(const std::string &path, uint64_t imageid) : Image(imageid) { 
			image_path = path;
		}	

		std::string feature_path(const std::string &feat_name) const {
			return s_test_data_dir + "/simple/feats/" + feat_name + "/"; 
		}

	protected:

		std::string image_path;
};


int main(int argc, char *argv[]) {

	InvertedIndex ii;
	std::shared_ptr<InvertedIndex::TrainParams> train_params = std::make_shared<InvertedIndex::TrainParams>();
	train_params->numClusters = 100;
	train_params->numFeatures = 10000;
	
	std::vector<std::shared_ptr<const Image> > examples(1);
	examples[0] = std::make_shared<const SimpleImage>(s_test_data_dir + "/simple/images/0000.jpg", 0);

	ii.train(train_params, examples);

	std::shared_ptr<InvertedIndex::SearchParams> search_params = std::make_shared<InvertedIndex::SearchParams>();
	std::shared_ptr<InvertedIndex::MatchResults> match_results = std::static_pointer_cast<InvertedIndex::MatchResults>(ii.search(search_params, examples[0]));


	return 0;
}