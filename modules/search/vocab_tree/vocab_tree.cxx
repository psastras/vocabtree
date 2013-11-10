#include "vocab_tree.hpp"
#include <utils/filesystem.hpp>
#include <utils/vision.hpp>
#include <iostream>
#include <memory>
#include <math.h> // for pow
#include <utility> // std::pair

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

bool VocabTree::train(Dataset &dataset, const std::shared_ptr<const TrainParamsBase> &params, 
  const std::vector< std::shared_ptr<const Image > > &examples) {

	const std::shared_ptr<const TrainParams> &vt_params = std::static_pointer_cast<const TrainParams>(params);
	split = vt_params->split;
	uint32_t depth = vt_params->depth;
  numberOfNodes = (uint32_t)pow(split, maxLevel) / (split - 1);
  weights.resize(numberOfNodes);

  // took the following from bag_of_words
  std::vector<uint64_t> all_ids(examples.size());
  for (uint32_t i = 0; i < examples.size(); i++) {
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


  root.levelIndex = 0;
  root.index = 0;
  buildTreeRecursive(root, merged_descriptor, tc, attempts, cv::KMEANS_PP_CENTERS, 0);

  databaseVectors.reserve(all_ids.size());

  // now generate data on the reference images - descriptors go down tree, add images to inverted lists at leaves, 
  //   and generate di vector for image
  // Also stores counts for how many images pass through each node to calculate weights
  std::vector<uint32_t> counts(numberOfNodes);
  for (size_t i = 0; i < numberOfNodes; i++)
    counts[i] = 0;

  for (size_t i = 0; i < all_ids.size(); i++) {
    std::shared_ptr<Image> image = std::static_pointer_cast<Image>(dataset.image(all_ids[i]));
    if (image == nullptr) continue;

    const std::string &descriptors_location = dataset.location(image->feature_path("descriptors"));
    if (!filesystem::file_exists(descriptors_location)) continue;

    cv::Mat descriptors;
    if (filesystem::load_cvmat(descriptors_location, descriptors)) {
      std::vector<float> result = generateVector(descriptors, false, all_ids[i]);
      // accumulate counts
      for (size_t j = 0; j < numberOfNodes; j++)
      if (result[j] > 0)
        counts[j]++;

      //databaseVectors.insert(std::make_pair<uint64_t, std::vector<float>>(all_ids[i], result));
      databaseVectors[all_ids[i]] = result;
    }
  }
  for (size_t i = 0; i < numberOfNodes; i++)
    weights[i] = log(((float)counts[i]) / ((float)all_ids.size()));

  // now that we have the weights we iterate over all images and adjust the vector by weights, 
  //  then normalizes the vector
  typedef std::unordered_map<uint64_t, std::vector<float>>::iterator it_type;
  for (it_type iterator = databaseVectors.begin(); iterator != databaseVectors.end(); iterator++) {
    float length = 0; // hopefully shouldn't overflow
    for (size_t i = 0; i < numberOfNodes; i++) {
      (iterator->second)[i] *= weights[i];
      length += (float)pow((iterator->second)[i], 2.0);
    }
    length = sqrt(length);
    // normalizing
    for (size_t i = 0; i < numberOfNodes; i++) 
      (iterator->second)[i] /= length;
  }

	return true;
}

void VocabTree::buildTreeRecursive(TreeNode t, cv::Mat descriptors, cv::TermCriteria tc, 
  int attempts, int flags, int currLevel) {

  t.level = currLevel;

  // handles the leaves
  if (currLevel == maxLevel - 1) {
    //create inverted file
    return;
  }

  cv::Mat labels;
  cv::Mat centers;

  cv::kmeans(descriptors, split, labels, tc, attempts, flags, centers);

  std::vector<cv::Mat> groups(split);
  for (uint32_t i = 0; i < split; i++)
    groups[i] = cv::Mat();

  for (int i = 0; i < labels.rows; i++) {
    int index = labels.at<int>(i);
    groups[index].push_back(descriptors.row(i));
  }

  for (uint32_t i = 0; i < split; i++) {
    TreeNode child;
    child.mean = centers.row(i);
    child.levelIndex = t.levelIndex*split + i;
    child.index = (uint32_t)(pow(split, t.level) / (split - 1)) + child.levelIndex;
    t.children.push_back(child);

    buildTreeRecursive(child, groups[i], tc, attempts, flags, currLevel + 1);
  }
}

std::vector<float> VocabTree::generateVector(cv::Mat descriptors, bool shouldWeight, uint64_t id) {
  std::unordered_set<uint64_t> dummy;
  return generateVector(descriptors, shouldWeight, dummy, id);
}

std::vector<float> VocabTree::generateVector(cv::Mat descriptors, bool shouldWeight, 
  std::unordered_set<uint64_t> & possibleMatches,  uint64_t id) {

  std::vector<float> vec(numberOfNodes);
  for (uint32_t i = 0; i < numberOfNodes; i++)
    vec[i] = 0;

  for (int r = 0; r < descriptors.rows; r++) {
    generateVectorHelper(root, descriptors.row(r), vec, possibleMatches, id);
  }

  // if shouldWeight is true then weight all values in the vector and normalize
  if (shouldWeight) {
    float length = 0; // for normalizing
    for (uint32_t i = 0; i < numberOfNodes; i++) {
      vec[i] *= weights[i];
      length += vec[i] * vec[i];
    }
    length = sqrt(length);
    for (uint32_t i = 0; i < numberOfNodes; i++)
      vec[i] /= length;
  }

  return vec;
}

void VocabTree::generateVectorHelper(TreeNode t, cv::Mat descriptor, std::vector<float> & counts, 
  std::unordered_set<uint64_t> & possibleMatches, uint64_t id) {

  counts[t.index]++;
  // if leaf
  if (t.children.size() == 0) {
    if (id >= 0) {
      if (t.invertedFile.find(id) == t.invertedFile.end())
        t.invertedFile[id] = 1;
      else
        t.invertedFile[id]++;
    }
    else {
      // i don't like doing this serial, should find a better method
      typedef std::unordered_map<uint64_t, uint32_t>::iterator it_type;
      for (it_type iterator = t.invertedFile.begin(); iterator != t.invertedFile.end(); iterator++)
        possibleMatches.insert(iterator->first);
    }
  }
  // if inner node
  else {
    int maxChild = 0;
    double max = descriptor.dot(t.children[0].mean);
    
    for (uint32_t i = 1; i < split; i++) {
      double dot = descriptor.dot(t.children[i].mean);
      if (dot>max) {
        max = dot;
        maxChild = 1;
      }
    }
    generateVectorHelper(t.children[maxChild], descriptor, counts, possibleMatches, id);
  }
}


std::shared_ptr<MatchResultsBase> VocabTree::search(Dataset &dataset, const std::shared_ptr<const SearchParamsBase> &params, 
  const std::shared_ptr<const Image > &example) {

	std::cout << "Searching for matching images..." << std::endl;
	const std::shared_ptr<const SearchParams> &ii_params = std::static_pointer_cast<const SearchParams>(params);
	
	std::shared_ptr<MatchResults> match_result = std::make_shared<MatchResults>();

  // get descriptors for example
  if (example == nullptr) return nullptr;
  const std::string &descriptors_location = dataset.location(example->feature_path("descriptors"));
  if (!filesystem::file_exists(descriptors_location)) return nullptr;

  cv::Mat descriptors;
  if (!filesystem::load_cvmat(descriptors_location, descriptors)) return nullptr;

  std::unordered_set<uint64_t> possibleMatches;

  std::vector<float> vec = generateVector(descriptors, true, possibleMatches);

  typedef std::pair<uint64_t, float> matchPair;
  std::vector<matchPair> values(possibleMatches.size());
  //for (int i = 0; i < vec.size(); i++) {
  for (uint64_t elem : possibleMatches) {
    // compute L1 norm (based on paper eq 5)
    float l1norm = 0;
    for (uint32_t i = 0; i < numberOfNodes; i++)
      l1norm += abs(vec[i] * (databaseVectors[elem])[i]);
    values.push_back(std::make_pair(elem, l1norm));
  }

  struct myComparer {
    bool operator() (matchPair a, matchPair b) { return a.second < b.second; };
  } comparer;

  std::sort(values.begin(), values.end(), comparer);

	// add in matches, right now just return the top 10%
  for (int i = 0; i < possibleMatches.size() / 10.0; i++)
    match_result->matches.push_back(values[i].first);
	//match_result->matches.push_back(0);

	return (std::shared_ptr<MatchResultsBase>)match_result;
}