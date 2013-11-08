#include "vocab_tree.hpp"
#include <utils/filesystem.hpp>
#include <utils/vision.hpp>
#include <iostream>
#include <memory>

VocabTree::VocabTree() : SearchBase() {


}

bool VocabTree::load (const std::string &file_path) {
	std::cout << "Reading vocab tree from " << file_path << "..." << std::endl;

	// code here

	std::cout << "Done reading vocab tree." << std::endl;
	
	return false;
}


bool VocabTree::save (const std::string &file_path) const {
	std::cout << "Writing vocab tree to " << file_path << "..." << std::endl;

	// code here

	std::cout << "Done writing vocab tree." << std::endl;

	return false;
}

bool VocabTree::train(Dataset &dataset, const std::shared_ptr<const TrainParamsBase> &params, const std::vector< std::shared_ptr<const Image > > &examples) {
	const std::shared_ptr<const TrainParams> &vt_params = std::static_pointer_cast<const TrainParams>(params);
	uint32_t split = vt_params->split;
	uint32_t depth = vt_params->depth;

  // took the following from bag_of_words
  std::vector<uint64_t> all_ids(examples.size());
  for (uint64_t i = 0; i < examples.size(); i++) {
    all_ids[i] = examples[i]->id;
  }
  std::random_shuffle(all_ids.begin(), all_ids.end());

  std::vector<cv::Mat> all_descriptors;
  uint64_t num_features = 0;
  for (size_t i = 0; i < all_ids.size(); i++) {
    std::shared_ptr<Image> image = std::static_pointer_cast<Image>(dataset.image(all_ids[i]));
    if (image == nullptr) continue;

    const std::string &descriptors_location = dataset.location(image->feature_path("descriptors"));
    if (!filesystem::file_exists(descriptors_location)) continue;

    cv::Mat descriptors;
    if (filesystem::load_cvmat(descriptors_location, descriptors)) {
      num_features += descriptors.rows;
      //if (n > 0 && num_features > n) break;

      all_descriptors.push_back(descriptors);
    }
  }

  const cv::Mat merged_descriptor = vision::merge_descriptors(all_descriptors, true);
  cv::Mat labels;
  uint32_t attempts = 1;
  cv::TermCriteria tc(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 16, 0.0001);
  // end of stuff from bag of words

  //cv::Mat centers;
  //cv::kmeans(merged_descriptor, split, labels, tc, attempts, cv::KMEANS_PP_CENTERS, centers);

  buildTreeRecursive(root, merged_descriptor, split, tc, attempts, cv::KMEANS_PP_CENTERS, 6, 0);

	return false;
}

void VocabTree::buildTreeRecursive(TreeNode t, cv::Mat descriptors, int split, cv::TermCriteria tc, 
  int attempts, int flags, int currLevel, int maxLevel) {

  t.invertedFileLength = descriptors.rows;

  // handles the leaves
  if (currLevel == maxLevel - 1) {

    return;
  }

  cv::Mat labels;
  cv::Mat centers;

  cv::kmeans(descriptors, split, labels, tc, attempts, flags, centers);

  std::vector<cv::Mat> groups(split);
  for (int i = 0; i < split; i++)
    groups[i] = cv::Mat();

  for (int i = 0; i < labels.rows; i++) {
    int index = labels.at<int>(i);
    groups[index].push_back(descriptors.row(i));
  }

  for (int i = 0; i < split; i++) {
    TreeNode child;
    child.mean = centers.row(i);
    t.children.push_back(child);
    buildTreeRecursive(child, groups[i], split, tc, attempts, flags, currLevel + 1, maxLevel);
  }
}

std::shared_ptr<MatchResultsBase> VocabTree::search(Dataset &dataset, const std::shared_ptr<const SearchParamsBase> &params, const std::shared_ptr<const Image > &example) {
	std::cout << "Searching for matching images..." << std::endl;
	const std::shared_ptr<const SearchParams> &ii_params = std::static_pointer_cast<const SearchParams>(params);
	
	std::shared_ptr<MatchResults> match_result = std::make_shared<MatchResults>();

	// returns zero as the only match 
	match_result->matches.push_back(0);

	return (std::shared_ptr<MatchResultsBase>)match_result;
}