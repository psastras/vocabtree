#include "tests_config.hpp"
#include <search/vocab_tree/vocab_tree.hpp>
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

	VocabTree ii;
	std::shared_ptr<VocabTree::TrainParams> train_params = std::make_shared<VocabTree::TrainParams>();
	train_params->depth = 6;
	train_params->split = 4;
	
	std::vector<std::shared_ptr<const Image> > examples(1);
	examples[0] = std::make_shared<const SimpleImage>(s_test_data_dir + "/simple/images/0000.jpg", 0);

	ii.train(train_params, examples);

	std::shared_ptr<VocabTree::SearchParams> search_params = std::make_shared<VocabTree::SearchParams>();
	std::shared_ptr<VocabTree::MatchResults> match_results = std::static_pointer_cast<VocabTree::MatchResults>(ii.search(search_params, examples[0]));


	return 0;
}