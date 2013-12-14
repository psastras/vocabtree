#include "vocab_tree.hpp"
#include <config.hpp>

#include <utils/filesystem.hpp>
#include <utils/vision.hpp>
#include <iostream>
#include <fstream>
#include <memory>
#include <math.h> // for pow
#include <utility> // std::pair

#define ENABLE_MULTINODE_TRAIN 0

#if ENABLE_MULTITHREADING && ENABLE_OPENMP
#include <omp.h>
#endif
#if ENABLE_MULTITHREADING && ENABLE_MPI
#include <mpi.h>
// define enums for message tags: 
enum { index_tag, levelIndex_tag, meanHeader_tag, mean_tag, indicesCount_tag, indices_tag, level_tag, maxNode_tag };
#endif

VocabTree::VocabTree() : SearchBase() {


}

// struct used for writing and reading cv::mat's
struct cvmat_header {
  uint64_t elem_size;
  int32_t elem_type;
  uint32_t rows, cols;
};

bool VocabTree::load (const std::string &file_path) {
  std::cout << "Reading vocab tree from " << file_path << "..." << std::endl;

  std::ifstream ifs(file_path, std::ios::binary);
  ifs.read((char *)&split, sizeof(uint32_t));
  ifs.read((char *)&maxLevel, sizeof(uint32_t));
  ifs.read((char *)&numberOfNodes, sizeof(uint32_t));

  weights.resize(numberOfNodes);
  ifs.read((char *)&weights[0], sizeof(float)*numberOfNodes);

  // load image data
  uint32_t imageCount;
  ifs.read((char *)&imageCount, sizeof(uint32_t));
  for (uint32_t i = 0; i < imageCount; i++) {
    uint64_t imageId;
    std::vector<float> vec(numberOfNodes);
    ifs.read((char *)&imageId, sizeof(uint64_t));
    ifs.read((char *)&vec[0], sizeof(float)*numberOfNodes);
    databaseVectors[imageId] = vec;
  }

  // load inveted files
  uint32_t invertedFileCount;
  ifs.read((char *)&invertedFileCount, sizeof(uint32_t));
  invertedFiles.resize(invertedFileCount);

  for (uint32_t i = 0; i < invertedFileCount; i++) {
    uint32_t size;
    ifs.read((char *)&size, sizeof(uint32_t));
    for (uint32_t j = 0; j < size; j++) {
      uint64_t imageId;
      uint32_t imageCount;
      ifs.read((char *)&imageId, sizeof(uint64_t));
      ifs.read((char *)&imageCount, sizeof(uint32_t));
      invertedFiles[i][imageId] = imageCount;
    }
  }

  // read in tree
  tree.resize(numberOfNodes);
  for (uint32_t i = 0; i < numberOfNodes; i++) {
    ifs.read((char *)&tree[i].firstChildIndex, sizeof(uint32_t));
    ifs.read((char *)&tree[i].index, sizeof(uint32_t));
    ifs.read((char *)&tree[i].invertedFileLength, sizeof(uint32_t));
    ifs.read((char *)&tree[i].level, sizeof(uint32_t));
    ifs.read((char *)&tree[i].levelIndex, sizeof(uint32_t));

    // read cv::mat, copied from filesystem.cxx
    cvmat_header h;
    ifs.read((char *)&h, sizeof(cvmat_header));
    tree[i].mean.create(h.rows, h.cols, h.elem_type);
    if (h.rows == 0 || h.cols == 0) continue;
    ifs.read((char *)tree[i].mean.ptr(), h.rows * h.cols * h.elem_size);
  }

  std::cout << "Done reading vocab tree." << std::endl;
  
  return (ifs.rdstate() & std::ifstream::failbit) == 0;
}

bool VocabTree::save (const std::string &file_path) const {
  std::cout << "Writing vocab tree to " << file_path << "..." << std::endl;

  std::ofstream ofs(file_path, std::ios::binary | std::ios::trunc);

  //uint32_t num_clusters = inverted_index.size();
  ofs.write((const char *)&split, sizeof(uint32_t));
  ofs.write((const char *)&maxLevel, sizeof(uint32_t));
  ofs.write((const char *)&numberOfNodes, sizeof(uint32_t));
  ofs.write((const char *)&weights[0], sizeof(float)*numberOfNodes); // weights

  // write out databaseVectors
  uint32_t imageCount = databaseVectors.size();
  ofs.write((const char *)&imageCount, sizeof(uint32_t));
  for (auto& pair : databaseVectors) {
    ofs.write((const char *)&pair.first, sizeof(uint64_t));
    ofs.write((const char *)&(pair.second)[0], sizeof(float)*numberOfNodes); 
  }

  // write out inverted files
  uint32_t numInvertedFiles = invertedFiles.size();
  ofs.write((const char *)&numInvertedFiles, sizeof(uint32_t));
  for (std::unordered_map<uint64_t, uint32_t> invFile : invertedFiles) {
    uint32_t size = invFile.size();
    ofs.write((const char *)&size, sizeof(uint32_t));
    for (std::pair<uint64_t, uint32_t> pair : invFile) {
      ofs.write((const char *)&pair.first, sizeof(uint64_t));
      ofs.write((const char *)&pair.second, sizeof(uint32_t));
    }
  }

  // write out tree
  for (uint32_t i = 0; i < numberOfNodes; i++) {
    TreeNode t = tree[i];
    ofs.write((const char *)&t.firstChildIndex, sizeof(uint32_t));
    ofs.write((const char *)&t.index, sizeof(uint32_t));
    ofs.write((const char *)&t.invertedFileLength, sizeof(uint32_t));
    ofs.write((const char *)&t.level, sizeof(uint32_t));
    ofs.write((const char *)&t.levelIndex, sizeof(uint32_t));

    // write cv::mat, copied from filesystem.cxx
    cvmat_header h;
    h.elem_size = t.mean.elemSize();
    h.elem_type = t.mean.type();
    h.rows = t.mean.rows;
    h.cols = t.mean.cols;
    ofs.write((char *)&h, sizeof(cvmat_header));
    ofs.write((char *)t.mean.ptr(), h.rows * h.cols * h.elem_size);
  }

  std::cout << "Done writing vocab tree." << std::endl;

  return (ofs.rdstate() & std::ofstream::failbit) == 0;
}

bool VocabTree::train(Dataset &dataset, const std::shared_ptr<const TrainParamsBase> &params,
  const std::vector< std::shared_ptr<const Image > > &examples) {
  //printf("Starting to build tree...\n");

  int rank = 0;
#if ENABLE_MULTITHREADING && ENABLE_MPI
  int procs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &procs);
  
  const std::string &tree_root_location = dataset.location("tree/");
  filesystem::create_file_directory(tree_root_location);
  std::stringstream ss;
  ss << tree_root_location << "tree." << split << "." << maxLevel << ".bin";
  const std::string &tree_location = ss.str();

  int t;
  if (rank != 0) {
    MPI_Recv(&t, 1, MPI_INT, 0, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    bool success = VocabTree::load(tree_location);
    printf("Node %d read from file path %d\n\n", rank, success);
    return success;
  }
#endif


  const std::shared_ptr<const TrainParams> &vt_params = std::static_pointer_cast<const TrainParams>(params);
  split = vt_params->split;
  //uint32_t depth = vt_params->depth;
  maxLevel = vt_params->depth;
  numberOfNodes = (uint32_t)(pow(split, maxLevel) - 1) / (split - 1);
  weights.resize(numberOfNodes);
  tree.resize(numberOfNodes);
  invertedFiles.resize((uint32_t)pow(split, maxLevel - 1));

  // took the following from bag_of_words
  std::vector<uint64_t> all_ids(examples.size());
  for (uint32_t i = 0; i < examples.size(); i++) {
    all_ids[i] = examples[i]->id;
  }

  // don't shuffle if using mpi because need to pass along image id's in the same order on all nodes
  std::random_shuffle(all_ids.begin(), all_ids.end());

  std::vector<cv::Mat> all_descriptors;
  uint64_t num_features = 0;
  for (size_t i = 0; i < all_ids.size(); i++) {
    std::shared_ptr<Image> image = std::static_pointer_cast<Image>(dataset.image(all_ids[i]));
    if (image == nullptr) continue;

    const std::string &descriptors_location = dataset.location(image->feature_path("descriptors"));
    if (!filesystem::file_exists(descriptors_location)) continue;

    cv::Mat descriptors, descriptorsf;
    if (filesystem::load_cvmat(descriptors_location, descriptors)) {
      descriptors.convertTo(descriptorsf, CV_32FC1);
      num_features += descriptors.rows;

      all_descriptors.push_back(descriptorsf);
    }
  }

  const cv::Mat merged_descriptor = vision::merge_descriptors(all_descriptors, true);
  cv::Mat labels;
  uint32_t attempts = 1;
  cv::TermCriteria tc(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 18, 0.000001);
  // end of stuff from bag of words

  uint32_t startNode = 0;
  uint32_t startLevel = 0;
  uint32_t levelIndex = 0;

  tree[startNode].levelIndex = levelIndex;
  tree[startNode].index = startNode;
  buildTreeRecursive(startNode, merged_descriptor, tc, attempts, cv::KMEANS_PP_CENTERS, startLevel);
  //printf("%d Built tree structure...\n", rank);

  databaseVectors.reserve(all_ids.size());

  // for mpi: synchronize all trees
  // even tags are for the node structs, odds are for mean data
#if ENABLE_MULTITHREADING && ENABLE_MPI && ENABLE_MULTINODE_TRAIN
  MPI_Barrier(MPI_COMM_WORLD); // all nodes should build their part of the tree

  int cols = merged_descriptor.cols;
  uint64_t elemSize = merged_descriptor.elemSize();
  int32_t elemType = merged_descriptor.type();
  printf("[%d] cols: %d, size: %d, type: %d\n", rank, cols, elemSize, elemType);

  printf(" %d here\n", rank);
  if (rank == 0)
    printf("master\n");
  else
    printf("slave");

  if (rank == 0) {
    printf("Sending...\n"); 
    std::vector<MPI_Request> sentRequests(procs - 1);
    int c = 0;
    for (int p = 1; p < procs; p++){
      MPI_Send(&tree[1], sizeof(TreeNode)*(numberOfNodes - 1), MPI_BYTE, p, 1874239473, MPI_COMM_WORLD);
      //MPI_Isend(&tree[1], sizeof(TreeNode)*(numberOfNodes - 1), MPI_BYTE, p, 1874239473, MPI_COMM_WORLD, &sentRequests[p - 1]);
      //MPI_Isend((char *)(tree[i].mean.ptr()), cols*elemSize, MPI_BYTE, p, 2 * i + 1, MPI_COMM_WORLD, &sentRequests[c++]);
    }
    //MPI_Waitall(procs-1, &sentRequests[0], MPI_STATUSES_IGNORE);
    /*std::vector<MPI_Request> sentRequests((procs-1)*(numberOfNodes-1));
    int c = 0;
    for (int i = 1; i < numberOfNodes; i++)
    for (int p = 1; p < procs; p++){
      MPI_Isend(&tree[i], sizeof(TreeNode), MPI_BYTE, p, i, MPI_COMM_WORLD, &sentRequests[c++]);
      //MPI_Isend((char *)(tree[i].mean.ptr()), cols*elemSize, MPI_BYTE, p, 2 * i + 1, MPI_COMM_WORLD, &sentRequests[c++]);
    }
    MPI_Waitall(c, &sentRequests[0], MPI_STATUSES_IGNORE); */


    /*cv::Mat tempMat;
    for (int i = 1; i < numberOfNodes; i++)
      tempMat.push_back(tree[i].mean);
    printf("ready to send %dx%d", tempMat.rows, tempMat.cols);
    std::vector<MPI_Request> requests((procs - 1));
    for (int p = 1; p < procs; p++)
      MPI_Isend((char *)tempMat.ptr(), cols*elemSize*(numberOfNodes), MPI_BYTE, p, 0, MPI_COMM_WORLD, &requests[p-1]);
    MPI_Waitall(requests.size(), &requests[0], MPI_STATUSES_IGNORE);*/
    /*c = 0;
    for (int i = 1; i < numberOfNodes; i++)
    for (int p = 1; p < procs; p++){
      MPI_Isend((char *)tree[i].mean.ptr(), cols*elemSize, MPI_BYTE, p, i, MPI_COMM_WORLD, &sentRequests[c++]);
    }
    MPI_Waitall(c, &sentRequests[0], MPI_STATUSES_IGNORE);*/

    int t = 42;
    MPI_Recv(&t, 1, MPI_INT, 1, 100000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("confirmed %d\n", t);
  }
  else {
    printf("Recieving %d nodes %d byte each; %d total...\n", numberOfNodes - 1, sizeof(TreeNode), sizeof(TreeNode)*(numberOfNodes - 1));
    std::vector<TreeNode> temp(numberOfNodes);
    MPI_Recv(&temp[1], sizeof(TreeNode)*(numberOfNodes - 1), MPI_BYTE, 0, 1874239473, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    /*std::vector<MPI_Request> requests(numberOfNodes - 1);
    for (int i = 1; i < numberOfNodes; i++) {
      MPI_Irecv(&tree[i], sizeof(TreeNode), MPI_BYTE, 0, i, MPI_COMM_WORLD, &requests[i-1]);
    }
    MPI_Waitall(requests.size(), &requests[0], MPI_STATUSES_IGNORE);*/

    /*cv::Mat tempMat(numberOfNodes - 1, cols, elemType);
    MPI_Recv((char *)tempMat.ptr(), cols*elemSize*(numberOfNodes), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (int i = 1; i < numberOfNodes; i++)
      tree[i].mean = tempMat.row(i-1);*/
    /*int index;
    MPI_Status status;
    for (int i = 0; i < 2*(numberOfNodes - 1); i++) {
      MPI_Waitany(requests.size(), &requests[0], &index, &status);
      if (status.MPI_TAG % 2 == 0) {
        tree[status.MPI_TAG / 2].mean.create(1, cols, elemType);
        MPI_Irecv((char *)tree[status.MPI_TAG / 2].mean.ptr(), cols*elemSize, MPI_BYTE, 0, status.MPI_TAG + 1, MPI_COMM_WORLD, &requests[index]);
      }
      else
        requests.erase(requests.begin() + index);
    }*/
    
    /*for (int i = 1; i < numberOfNodes; i++) {
      tree[i].mean.create(1, cols, elemType);
      MPI_Irecv((char *)(tree[i].mean.ptr()), cols*elemSize, MPI_BYTE, 0, i, MPI_COMM_WORLD, &requests[i-1]);
    }
    MPI_Waitall(requests.size(), &requests[0], MPI_STATUSES_IGNORE);*/
    int t = 42;
    MPI_Send(&t, 1, MPI_INT, 0, 100000, MPI_COMM_WORLD);
  }

  printf("  %d HERE!!\n", rank);

  if (rank == 0) {
    for (uint32_t i = 0; i < (uint32_t)pow(split, maxLevel - 1); i++) {
      //printf("Size of inv file %d: %d\n", i, invertedFiles[i].sizbe());
    }
    printf("\n\n");
    uint32_t l = 0, inL = 0;
    for (uint32_t i = 0; i < numberOfNodes; i++) {
      printf("Node %d, len %d, w %f | ", i, tree[i].invertedFileLength, weights[i]);
      inL++;
      if (inL >= (uint32_t)pow(split, l)) {
        l++;
        inL = 0;
        printf("\n");
      }
    }
  }
  return true;

  //MPI_Barrier(MPI_COMM_WORLD); 
#endif
  
  // generate data on the reference images - descriptors go down tree, add images to inverted lists at leaves, 
  //   and generate di vector for image
  // Also stores counts for how many images pass through each node to calculate weights
  std::vector<uint32_t> counts(numberOfNodes);
  for (size_t i = 0; i < numberOfNodes; i++)
    counts[i] = 0;

  
#if ENABLE_MULTITHREADING && ENABLE_MPI && ENABLE_MULTINODE_TRAIN
  int imagesPerProc = ceil(((float)all_ids.size()) / procs);
#else
  int imagesPerProc = all_ids.size();
#endif

#if ENABLE_MULTITHREADING && ENABLE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int i = rank*imagesPerProc; i < std::min((int)all_ids.size(), (rank + 1)*imagesPerProc); i++) {
    std::shared_ptr<Image> image = std::static_pointer_cast<Image>(dataset.image(all_ids[i]));
    if (image == nullptr) continue;

    const std::string &descriptors_location = dataset.location(image->feature_path("descriptors"));
    if (!filesystem::file_exists(descriptors_location)) continue;

    cv::Mat descriptors, descriptorsf;
    if (filesystem::load_cvmat(descriptors_location, descriptors)) {
      descriptors.convertTo(descriptorsf, CV_32FC1);
      std::vector<float> result = generateVector(descriptorsf, false, all_ids[i]);

      // accumulate counts
      for (size_t j = 0; j < numberOfNodes; j++)
      if (result[j] > 0)
#pragma omp critical
      {
        counts[j]++;
      }

      //databaseVectors.insert(std::make_pair<uint64_t, std::vector<float>>(all_ids[i], result));
#pragma omp critical
      {
        databaseVectors[all_ids[i]] = result;
      }
    }
  }

  
  // mpi synchronize counts
#if ENABLE_MULTITHREADING && ENABLE_MPI && ENABLE_MULTINODE_TRAIN
  int c = 0;
  std::vector<MPI_Request> requests(procs - 1);
  //MPI_Request requests[procs-1];

  for (int i = 0; i < procs; i++) {
    if (rank == i) continue;
    MPI_Isend(&counts[0], counts.size(), MPI_INT, i, 0, MPI_COMM_WORLD, &requests[c]);
    c++;
  }

  // recieving
  std::vector<uint32_t> addOthers(numberOfNodes);
  for(int i=0; i<numberOfNodes; i++)
    addOthers[i]=0;
  std::vector<uint32_t> otherCounts(numberOfNodes);

  for (int i = 0; i < procs - 1; i++) {
    MPI_Recv(&otherCounts[0], counts.size(), MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (int j = 0; j < counts.size(); j++)
      addOthers[j] += otherCounts[j];
  }
  MPI_Waitall(c, &requests[0], MPI_STATUSES_IGNORE);

  for (int j = 0; j < counts.size(); j++)
    counts[j] += addOthers[j];
#endif
    
  // create weights according to equation 4: w_i = ln(N / N_i)
#if ENABLE_MULTITHREADING && ENABLE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int i = 0; i < numberOfNodes; i++) {
    if (counts[i] == 0)
      weights[i] = 0;
    else
      weights[i] = log(((float)all_ids.size()) / ((float)counts[i]));
    // printf("Node %d, count %d, total %d, size %d, weight %f \n", i, counts[i], all_ids.size(), tree[i].invertedFileLength, weights[i]);
  }

  // now that we have the weights we iterate over all images and adjust the vector by weights, 
  //  then normalizes the vector
  typedef std::unordered_map<uint64_t, std::vector<float>>::iterator it_type;
  for (it_type iterator = databaseVectors.begin(); iterator != databaseVectors.end(); iterator++) {
    float length = 0; // hopefully shouldn't overflow from adding doubles
    //std::vector<float> datavec = (iterator->second);
    for (size_t i = 0; i < numberOfNodes; i++) {
      (iterator->second)[i] *= weights[i];
      length += (float)pow((iterator->second)[i], 2.0);
    }
    // normalizing
    length = sqrt(length);
    for (size_t i = 0; i < numberOfNodes; i++) 
      (iterator->second)[i] /= length;

    // write out vector to database
    std::shared_ptr<Image> image = std::static_pointer_cast<Image>(dataset.image(iterator->first));
    const std::string &datavec_location = dataset.location(image->feature_path("datavec"));

    filesystem::create_file_directory(datavec_location);

    std::ofstream ofs(datavec_location.c_str(), std::ios::binary | std::ios::trunc);
    ofs.write((char *)&(iterator->second)[0], numberOfNodes*sizeof(float));
    if ((ofs.rdstate() & std::ofstream::failbit) != 0)
      std::cout << "Failed to write data for " << iterator->first << " to " << datavec_location << std::endl;

    /*if (!filesystem::file_exists(datavec_location)) { printf("COULDN'T FIND FILE\n\n"); continue; };
    std::vector<float> dbVec(numberOfNodes);
    std::ifstream ifs(datavec_location.c_str(), std::ios::binary);
    ifs.read((char *)&dbVec[0], numberOfNodes*sizeof(float));
    if ((ifs.rdstate() & std::ifstream::failbit) != 0) { printf("FAILLLLLLLLLL\n\n"); continue; }

    printf("Original: ");
    for (int i = 0; i < numberOfNodes; i++)
      printf("%f ", datavec[i]);
    printf("\nSaved: ");
    for (int i = 0; i < numberOfNodes; i++)
      printf("%f ", dbVec[i]);
    printf("\n\n");*/
  }


  /*uint32_t l = 0, inL = 0;
  for (uint32_t i = 0; i < numberOfNodes; i++) {
    printf("Node %d, ifl %d, count %d, weight %f Desc (%d):\n ", i, tree[i].invertedFileLength, counts[i], weights[i], tree[i].mean.cols);
    for (int j = 0; j < tree[i].mean.cols && j<8; j++)
      printf("%f ", tree[i].mean.at<float>(0,j));
    printf("\n\n");
    inL++;
    if (inL >= (uint32_t)pow(split, l)) {
      l++;
      inL = 0;
      printf("-----------------------------------------\n\n");
    }
  }*/


  /*const std::string &tree_root_location = dataset.location("tree/");
  std::stringstream ss;
  //ss << "C:/Users/Conrad/Documents/15-869_Visual_Computing_Systems/Final_Project/vocabtree/data/oxfordmini/tree." << split << "." << maxLevel << ".bin";
  ss << tree_root_location << "tree." << split << "." << maxLevel << ".bin";
  const std::string &tree_location = ss.str();
  filesystem::create_file_directory(tree_root_location);
  if (VocabTree::save(tree_location))
    printf("Wrote successfully\n");
  else
    printf("Failed write\n");*/

  // synchronize leaf and vector information, everything send to node 0
  
#if ENABLE_MULTITHREADING && ENABLE_MPI && ENABLE_MULTINODE_TRAIN
  struct Temp_pair {
    uint64_t key;
    uint32_t val;
  };
  MPI_Barrier(MPI_COMM_WORLD); 
  // get leaf sets
  int leaves = invertedFiles.size();
  if (rank == 0) {
    int totalCount = 0;
    int t;
    for(int i=0; i<procs-1; i++) {
      MPI_Recv(&t, 1, MPI_INT, MPI_ANY_SOURCE, leaves, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      totalCount += t;
    }

    Temp_pair tmpPair;
    MPI_Status status;
    for(int i=0; i<totalCount; i++) {
      MPI_Recv(&tmpPair, sizeof(Temp_pair), MPI_BYTE, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
      invertedFiles[status.MPI_TAG][tmpPair.key] = tmpPair.val;
    }
  }
  else {
    //Request sizeRequests[leaves];
    std::vector<MPI_Request> sizeRequests(leaves);
    Temp_pair tmpPair;
    for (int i = 0; i < leaves; i++) {
      int fileSize = invertedFiles[i].size();
      MPI_Isend(&fileSize, 1, MPI_INT, 0, leaves, MPI_COMM_WORLD, &sizeRequests[i]);

      for (auto & p : invertedFiles[i]) {
        tmpPair.key = p.first;
        tmpPair.val = p.second;
        MPI_Send(&tmpPair, sizeof(Temp_pair), MPI_BYTE, 0, 1, MPI_COMM_WORLD);
      }
    }

    MPI_Waitall(leaves, &sizeRequests[0], MPI_STATUSES_IGNORE);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  //get data vectors
  if (rank == 0) {
    int totalCount = 0;
    int t;
    for(int i=0; i<procs-1; i++) {
      MPI_Recv(&t, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      totalCount += t;
    }
    MPI_Status status;
    for(int i=0; i<totalCount; i++) {
      std::vector<float> newVec(numberOfNodes);
      MPI_Recv(&newVec[0], numberOfNodes, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      databaseVectors[status.MPI_TAG-1] = newVec;
    }
  }
  else {
    int count = databaseVectors.size();
    MPI_Send(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    for (auto & p : databaseVectors)
      MPI_Send(&(p.second[0]), numberOfNodes, MPI_FLOAT, 0, p.first + 1, MPI_COMM_WORLD);
  }
#endif

#if ENABLE_MULTITHREADING && ENABLE_MPI
  // will only be here if node 0
  // save file and tell other nodes they can read the file
  if(!VocabTree::save(tree_location))
    return false;
  printf("Node 0 wrote to file path\n\n");

  std::vector<MPI_Request> requests(procs - 1);
  int asdf = 42;
  for (int p = 1; p < procs; p++)
    MPI_Isend(&asdf, 1, MPI_INT, p, 42, MPI_COMM_WORLD, &requests[p - 1]);
  MPI_Waitall(procs - 1, &requests[0], MPI_STATUSES_IGNORE);
#endif
  
  return true;
}


void VocabTree::buildTreeRecursive(uint32_t t, const cv::Mat &descriptors, cv::TermCriteria &tc,
  int attempts, int flags, int currLevel) {
  
  tree[t].invertedFileLength = descriptors.rows;
  tree[t].level = currLevel;

#if ENABLE_MULTITHREADING && ENABLE_MPI && ENABLE_MULTINODE_TRAIN
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) {
    tree[0].firstChildIndex = 1;
    return;
  }
#endif

  // handles the leaves
  if (currLevel == maxLevel - 1) {
    tree[t].firstChildIndex = 0;
    return;
  }
  cv::Mat labels;
  cv::Mat centers;
  
  std::vector<cv::Mat> groups(split);
  std::vector< std::vector<cv::Mat> > unjoinedGroups(split);
  for (uint32_t i = 0; i < split; i++)
    groups[i] = cv::Mat();//cv::Mat(0, descriptors.cols, descriptors.type);


  // printf("t: %d  rows: %d, counts: ", t, descriptors.rows);

  bool enoughToFill = true;
  if (descriptors.rows >= split) {
    // gather desired descriptors
    cv::kmeans(descriptors, split, labels, tc, attempts, flags, centers);

    for (int i = 0; i < labels.rows; i++) {
      int index = labels.at<int>(i);

      //groups[index].push_back(descriptors.row(i));
      unjoinedGroups[index].push_back(descriptors.row(i));

    }
  }
  else {
    // *** THIS SHOULDN'T BE THE CASE, why is kmeans splitting poorly? ****
    enoughToFill = false;
    for (int i = 0; i < descriptors.rows; i++) {
      //groups[i].push_back(descriptors.row(i));
      unjoinedGroups[i].push_back(descriptors.row(i));
    }

  }

 for (int i = 0; i<split; i++) {
   if (unjoinedGroups[i].size() > 0) {
     /***** This will give memory problems and crash for large inputs *****/
     //printf("joining %d ... ", unjoinedGroups[i].size());
     groups[i] = vision::merge_descriptors(unjoinedGroups[i], false);
     //printf("joined\n");
   }
  }


/*#if ENABLE_MULTITHREADING && ENABLE_OPENMP
  uint32_t totalChildren = pow(split, currLevel);
  uint32_t maxThreads = omp_get_num_threads();
#endif*/
  
  // only do omp parallel if not splitting the work to other children
#if ENABLE_MULTITHREADING && ENABLE_OPENMP /*&& totalChildren<maxThreads*/ // && !(ENABLE_MULTITHREADING && ENABLE_MPI)
#pragma omp parallel for schedule(dynamic)
#endif
  for (int32_t i = 0; i < split; i++) {
    uint32_t childLevelIndex = tree[t].levelIndex*split + i;
    uint32_t childIndex = (uint32_t)((pow(split, tree[t].level + 1) - 1) / (split - 1)) + childLevelIndex;
    if (i == 0)
      tree[t].firstChildIndex = childIndex;

    /*cv::Mat desc;
    if (unjoinedGroups[i].size()>0) {
      printf("joining %d ... ", unjoinedGroups[i].size());
      desc = vision::merge_descriptors(unjoinedGroups[i], false);
      printf("joined\n");
    }*/
    if (enoughToFill)
      cv::normalize(centers.row(i), tree[childIndex].mean);
      //tree[childIndex].mean = centers.row(i);
    tree[childIndex].levelIndex = childLevelIndex;
    tree[childIndex].index = childIndex;

    buildTreeRecursive(childIndex, groups[i], tc, attempts, flags, currLevel + 1);
    //buildTreeRecursive(childIndex, desc, tc, attempts, flags, currLevel + 1);
  }
}

std::vector<float> VocabTree::generateVector(const cv::Mat &descriptors, bool shouldWeight, int64_t id) {
  std::unordered_set<uint32_t> dummy;
  return generateVector(descriptors, shouldWeight, dummy, id);
}

std::vector<float> VocabTree::generateVector(const cv::Mat &descriptors, bool shouldWeight,
  std::unordered_set<uint32_t> & possibleMatches, int64_t id) {

  std::vector<float> vec(numberOfNodes);
  for (uint32_t i = 0; i < numberOfNodes; i++)
    vec[i] = 0;

#if ENABLE_MULTITHREADING && ENABLE_MPI
  int rank, procs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &procs);
#endif

#if ENABLE_MULTITHREADING && ENABLE_MPI
#if ENABLE_MULTITHREADING && ENABLE_OPENMP
#pragma omp parallel for
#endif
  // run over multiple nodes only if called by search, not train
  for (int r = (id>=0? 0:rank); r < descriptors.rows; r+=(id>0?1:procs)) {
#else
#if ENABLE_MULTITHREADING && ENABLE_OPENMP
#pragma omp parallel for
#endif
  for (int r = 0; r < descriptors.rows; r++) {
#endif
    //printf("%d ", r);
    generateVectorHelper(0, descriptors.row(r), vec, possibleMatches, id);
  }

  /*printf("my vector: ");
  for (int i = 0; i < numberOfNodes; i++)
    printf("%f ", vec[i]);
  printf("\n");*/
  
#if ENABLE_MULTITHREADING && ENABLE_MPI
  if(id<0) {
    /*printf("[%d] my vector: ", rank);
    for (int i = 0; i < numberOfNodes; i++)
      printf("%f ", vec[i]);
    printf("\n");*/

    std::vector<MPI_Request> requests(procs - 1);
    int c = 0;
    for(int i=0; i<procs; i++)
    if (i != rank)
      MPI_Isend(&vec[0], numberOfNodes, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &requests[c++]);

    std::vector<float> otherVec(numberOfNodes);
    std::vector<float> sumOthers(numberOfNodes);
    for (int i = 0; i < numberOfNodes; i++)
      sumOthers[i] = 0;

    for (int i = 0; i<procs; i++)
    if (i != rank) {
      MPI_Recv(&otherVec[0], numberOfNodes, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      for (int j = 0; j < numberOfNodes; j++)
        sumOthers[j] += otherVec[j];
    }

    MPI_Waitall(procs - 1, &requests[0], MPI_STATUSES_IGNORE);
    for (int i = 0; i < numberOfNodes; i++)
      vec[i] += sumOthers[i];

    /*printf("[%d] new vector: ", rank);
    for (int i = 0; i < numberOfNodes; i++)
      printf("%f ", vec[i]);
    printf("\n");*/
  }
#endif
  
  // if shouldWeight is true then weight all values in the vector and normalize
  if (shouldWeight) {
    /*printf("WEIGHTS: ");
    for (int i = 0; i < numberOfNodes; i++)
      printf("%f ", weights[i]);
    printf("\n");*/

    float length = 0; // for normalizing
    for (int32_t i = 0; i < numberOfNodes; i++) {
      vec[i] *= weights[i];
      length += vec[i] * vec[i];
    }
    length = sqrt(length);
#if ENABLE_MULTITHREADING && ENABLE_OPENMP
#pragma omp parallel for
#endif
    for (int32_t i = 0; i < numberOfNodes; i++) {
      if(length == 0)
        vec[i] = 0;
      else
        vec[i] /= length;
    }

  }

  return vec;
}

void VocabTree::generateVectorHelper(uint32_t nodeIndex, const cv::Mat &descriptor, std::vector<float> & counts,
  std::unordered_set<uint32_t> & possibleMatches, int64_t id) {

#pragma omp critical
    {
      counts[nodeIndex]++;
    }

  // if leaf
  if (tree[nodeIndex].firstChildIndex <= 0) {
    std::unordered_map<uint64_t, uint32_t> & invFile = invertedFiles[tree[nodeIndex].levelIndex];
    //printf("|%d|=%d, ", tree[nodeIndex].levelIndex, invFile.size());
    // inserting image id into the inverted file
    if (id >= 0) {
      //printf("Leaf %d\n", nodeIndex);
#pragma omp critical
      {
      if (invFile.find(id) == invFile.end())
        invFile[id] = 1;
      else
        invFile[id]++;
      }
    }
    // accumulating image id's into possibleMatches
    else {
      // i don't like doing this serial, should find a better method
      //typedef std::unordered_map<uint64_t, uint32_t>::iterator it_type;
      //for (it_type iterator = invFile.begin(); iterator != invFile.end(); iterator++)
#pragma omp critical
      {
        possibleMatches.insert(tree[nodeIndex].levelIndex); //iterator->first);
      }
    }
  }
  // if inner node
  else {
    uint32_t maxChild = tree[nodeIndex].firstChildIndex;
    /*printf("Desc (%d): ", tree[maxChild].mean.dims);
    for (int i = 0; i < tree[maxChild].mean.cols; i++)
      printf("%f ", tree[maxChild].mean.at<float>(i));
    printf("\n\n");*/
    double max = tree[maxChild].mean.dims == 0 ? 0 : descriptor.dot(tree[maxChild].mean);
    //double max = descriptor.dot(tree[maxChild].mean);

    for (uint32_t i = 1; i < split; i++) {
      if (tree[nodeIndex].invertedFileLength == 0)
        continue;
      uint32_t childIndex = tree[nodeIndex].firstChildIndex + i;
    /*printf("Desc (%d): ", tree[nodeIndex].mean.dims);
    for (int i = 0; i < tree[nodeIndex].mean.cols; i++)
      printf("%f ", tree[nodeIndex].mean.at<float>(i));
    printf("\n\n");*/
      if (tree[childIndex].mean.dims == 0)
        continue;
      double dot = descriptor.dot(tree[childIndex].mean);

      if (dot>max) {
        max = dot;
        maxChild = childIndex;
      }
    }
    generateVectorHelper(maxChild, descriptor, counts, possibleMatches, id);
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

  cv::Mat descriptors, descriptorsf;
  if (!filesystem::load_cvmat(descriptors_location, descriptors)) return nullptr;

  std::unordered_set<uint32_t> possibleMatches;
  descriptors.convertTo(descriptorsf, CV_32FC1);

  // printf("--Generating vector...\n");
  std::vector<float> vec = generateVector(descriptorsf, true, possibleMatches);
  // printf("--Generated vector\n");

  typedef std::pair<uint64_t, float> matchPair;
  struct myComparer {
    bool operator() (matchPair a, matchPair b) { return a.second < b.second; };
  } comparer;

  std::unordered_set<uint64_t> possibleImages;
  for (uint32_t elem : possibleMatches) {
    std::unordered_map<uint64_t, uint32_t> & invFile = invertedFiles[elem];

    typedef std::unordered_map<uint64_t, uint32_t>::iterator it_type;
    for (it_type iterator = invFile.begin(); iterator != invFile.end(); iterator++)
    if (possibleImages.count(iterator->first) == 0)
      possibleImages.insert(iterator->first);
  }

  //std::set<matchPair, myComparer> values;
  std::vector<matchPair> values;
  for (uint64_t elem : possibleImages) {
    // compute L1 norm (based on paper eq 5)
    //float l1norm = 0;
    float score = 0;

    // load datavec from disk
    std::shared_ptr<Image> image = std::static_pointer_cast<Image>(dataset.image(elem));
    const std::string &datavec_location = dataset.location(image->feature_path("datavec"));

    if (!filesystem::file_exists(datavec_location)) continue;
    std::vector<float> dbVec(numberOfNodes);
    std::ifstream ifs(datavec_location.c_str(), std::ios::binary);
    ifs.read((char *)&dbVec[0], numberOfNodes*sizeof(float));
    if ((ifs.rdstate() & std::ifstream::failbit) != 0) continue;

    for (uint32_t i = 0; i < numberOfNodes; i++) {
      float t = vec[i] - dbVec[i];
      score += t*t;
      // std::cout << vec[i] << std::endl;
      //l1norm += abs(vec[i] * (databaseVectors[elem])[i]);
    }
    //values[elem] = l1norm;
    //values.insert(elem, l1norm));

    values.push_back(matchPair(elem, sqrt(score)));
  }

  std::sort(values.begin(), values.end(), comparer);
  if (values.size() > ii_params->amountToReturn) {
    std::vector<matchPair>::iterator it = values.begin();
    values.erase(it + ii_params->amountToReturn, values.end());
  }

  /*int rank, procs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &procs);
  if (rank != 0)
    values.clear();*/

  // aggregate everything into node 0
#if ENABLE_MULTITHREADING && ENABLE_MPI
  int rank, procs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &procs);

  if (rank == 0) {
    int totalCount = 0;
    int t;
    for (int i = 0; i<procs - 1; i++) {
      MPI_Recv(&t, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      totalCount += t;
    }

    MPI_Status status;
    float score;
    for (int i = 0; i<totalCount; i++) {
      MPI_Recv(&score, 1, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      values.push_back(matchPair(status.MPI_TAG-1, score));
    }

    // this may result in duplicate entries
    // will have to decide if that's a problem and if its worth fixing
    std::sort(values.begin(), values.end(), comparer);
    values.erase(values.begin() + ii_params->amountToReturn, values.end());
  }
  else {
    int tmpCount = values.size();
    MPI_Send(&tmpCount, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

    for (matchPair p : values)
      MPI_Send(&p.second, 1, MPI_FLOAT, 0, p.first+1, MPI_COMM_WORLD);

    values.clear();
  }
#endif

  // printf("%d matches\n", values.size());
  // add all images in order or match
  for (matchPair m : values){
    match_result->matches.push_back(m.first);
    match_result->tfidf_scores.push_back(m.second);
    // std::cout << m.second << std::endl;
  }

  return (std::shared_ptr<MatchResultsBase>)match_result;
}

std::vector< std::shared_ptr<MatchResultsBase> > VocabTree::search(Dataset &dataset, const std::shared_ptr<SearchParamsBase> &params,
  const std::vector< std::shared_ptr<const Image > > &examples) {

  std::vector< std::shared_ptr<MatchResultsBase> > results(examples.size());
  /*
  int numThreads = 1;
#if ENABLE_MULTITHREADING && ENABLE_OPENMP
//#pragma omp parallel for
  numThreads = omp_get_max_threads();
#endif
  */
#if ENABLE_MULTITHREADING && ENABLE_OPENMP
  //#pragma omp parallel for
#endif
  for (int i = 0; i < examples.size(); i++) {
    std::shared_ptr<MatchResultsBase> imResults = search(dataset, params, examples[i]);
    results[i] = imResults;
  }

  return results;
}
  
uint32_t VocabTree::tree_splits() const {
	return split;
}

uint32_t VocabTree::tree_depth() const {
	return maxLevel;
}
