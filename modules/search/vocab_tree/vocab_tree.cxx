#include "vocab_tree.hpp"
#include <utils/filesystem.hpp>
#include <utils/vision.hpp>
#include <iostream>
#include <fstream>
#include <memory>
#include <math.h> // for pow
#include <utility> // std::pair

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

  return (ofs.rdstate() & std::ofstream::failbit) == 0;;
}

bool VocabTree::train(Dataset &dataset, const std::shared_ptr<const TrainParamsBase> &params,
  const std::vector< std::shared_ptr<const Image > > &examples) {

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
#if !(ENABLE_MULTITHREADING && ENABLE_MPI)
  std::random_shuffle(all_ids.begin(), all_ids.end());
#endif

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
      //all_descriptors.push_back(descriptors);
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
  float_t ratio = 0;

#if ENABLE_MULTITHREADING && ENABLE_MPI
  int rank = MPI::COMM_WORLD.Get_rank();
  uint32_t minNode, maxNode;
  int descriptorCount;
  std::vector<uint32_t> indices;
  ratio = ((float_t)merged_descriptor.rows) / MPI::COMM_WORLD.Get_size();\

#if ENABLE_MULTITHREADING && ENABLE_OPENMP
#pragma omp parallel for
#endif
  for(int i=0; i<numberOfNodes; i++)
    tree[i].index = -1;

  if (rank == 0) {
    minNode = 0;
    maxNode = MPI::COMM_WORLD.Get_size() - 1;
    descriptorCount = merged_descriptor.rows;
    indices.resize(descriptorCount);

#if ENABLE_MULTITHREADING && ENABLE_OPENMP
#pragma omp parallel for
#endif
    for(uint32_t i=0; i<descriptorCount; i++)
      indices[i] = i;
  }
  else {
    // get: which node to work on, number of descriptors, descriptors (indices), current level, maxNode (that can send work to), and levelIndex
    // write to startNode, startLevel, and levelIndex
    // read in mean
    Request requests[6];
    Request headerReq;

    requests[0] = Comm::Irecv(&startNode, 1, MPI_INT, MPI_ANY_SOURCE, index_tag);
    requests[1] = Comm::Irecv(&levelIndex, 1, MPI_INT, MPI_ANY_SOURCE, levelIndex_tag);

    // send mean_vector
    cvmat_header h;
    headerReq = Comm::Irecv(&h, sizeof(cvmat_header), MPI_BYTES, MPI_ANY_SOURCE, meanHeader_tag,);

    requests[2] = Comm::Irecv(&maxNode, 1, MPI_INT, MPI_ANY_SOURCE, maxNode_tag);
    requests[3] = Comm::Irecv(&startLevel, 1, MPI_INT, MPI_ANY_SOURCE, level_tag);

    int indicesCount;
    Comm::Recv(&indicesCount, 1, MPI_INT, MPI_ANY_SOURCE, indicesCount_tag, MPI_STATUSES_IGNORE);
    indices.resize(indicesCount);
    requests[4] = Comm::Irecv(&(indices[0]), indicesCount, MPI_INT, MPI_ANY_SOURCE, indices_tag);

    Request::Waitany(1, &headerReq, MPI_STATUSES_IGNORE);
    tree[startNode].mean.create(h.rows, h.cols, h.elem_type);
    requests[5] = Comm::Irecv((char *)tree[startNode].mean.ptr(), h.rows * h.cols * h.elem_size, MPI_BYTES, MPI_ANY_SOURCE, mean_tag);

    Request::Waitall(6, requests);
  }

#else
  std::vector<uint32_t> indices(0);
  uint32_t maxNode = 0;
#endif

  tree[startNode].levelIndex = levelIndex;
  tree[startNode].index = startNode;
  buildTreeRecursive(startNode, merged_descriptor, tc, attempts, cv::KMEANS_PP_CENTERS, startLevel, indices, maxNode, ratio);

  databaseVectors.reserve(all_ids.size());

  // for mpi: synchronize all trees
  // even tags are for the node structs, odds are for mean data
#if ENABLE_MULTITHREADING && ENABLE_MPI
  MPI::Comm::Barrier(); // all nodes should build their part of the tree

  int cols = merged_descriptor.cols;
  uint64_t elemSize = merged_descriptor.elemSize;
  int32_t elemType = merged_descriptor.type();
  int procs = MPI::COMM_WORLD.Get_size();

  std::vector<Request> sentRequests;
  std::vector<Request> recieveRequests;

  for (int i = 0; i < numberOfNodes; i++) {
    // if this compute node has data on this tree node then the index will be >=0
    if (tree[i].index >= 0) { // send data out
      for (int j = 0; j < procs; j++) {
        if (j == rank) continue;
        sentRequests.push_back(Comm::Isend(&tree[i], sizeof(TreeNode), MPI_BYTES, j, 2 * i);
        sentRequests.push_back(Comm::Isend((char *)tree[i].ptr(), cols*elemSize, MPI_BYTES, j, 2 * i + 1);
      }
    }
    else { // recieve data
      recieveRequests.push_back(Comm::Irecv(&tree[i], sizeof(TreeNode), MPI_BYTES, MPI_ANY_SOURCE, 2 * i));
    }
  }

  int recieved = 0; // keeps track of how many nodes have been read, including mean matrix data
  Status status;
  int index;
  while (recieved < recieveReqCount) {
    index = Request::Waitany(recieveRequests.size(), &recieveRequests[0], &status);
    if(status.MPI_TAG%2 == 0) { // make new request
      int t = status.MPI_TAG / 2;
      tree[t].mean.create(1, cols, elemType);
      recieveRequests[recieved] = Comm::Irecv((char *)tree[t].mean.ptr(), cols*elemSize, MPI_BYTES, MPI_ANY_SOURCE, status.MPI_TAG+1);
    }
    else
      recieved++;
  }

  Request::Waitall(sentRequests.sount(), &sentRequests[0]);
  MPI::Comm::Barrier(); 

#endif

  // generate data on the reference images - descriptors go down tree, add images to inverted lists at leaves, 
  //   and generate di vector for image
  // Also stores counts for how many images pass through each node to calculate weights
  std::vector<uint32_t> counts(numberOfNodes);
  for (size_t i = 0; i < numberOfNodes; i++)
    counts[i] = 0;


#if ENABLE_MULTITHREADING && ENABLE_MPI
  int imagesPerProc = ceil(((float)all_ids.size()) / procs;
#else
  int imagesPerProc = all_ids.size();
  int rank = 0;
#endif

#if ENABLE_MULTITHREADING && ENABLE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (size_t i = rank*imagesPerProc; i < std::min((int)all_ids.size(), (rank + 1)*imagesPerProc); i++) {
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
#if ENABLE_MULTITHREADING && ENABLE_MPI
  int c = 0;
  Request requests[procs-1];

  for (int i = 0; i < procs; i++) {
    if (rank == i) continue;
    requests[c] = Comm::Isend(&counts[0], counts.size(), MPI_INT, i, 0);
    c++;
  }

  // recieving
  std::vector<uint32_t> addOthers(numberOfNodes);
  for(int i=0; i<numberOfNodes; i++)
    addOthers[i]=0;
  std::vector<uint32_t> otherCounts(numberOfNodes);

  for (int i = 0; i < procs - 1; i++) {
    Comm::Recv(&otherCounts[0], counts.size(), MPI_INT, MPI_ANY_SOURCE, 0);
    for (int j = 0; j < counts.size(); j++)
      addOthers[j] += otherCounts[j];
  }
  Request::Waitall(c, &requests[0]);

  for (int j = 0; j < counts.size(); j++)
    counts[j] += addOthers[j];
#endif

  // create weights according to equation 4: w_i = ln(N / N_i)
#if ENABLE_MULTITHREADING && ENABLE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (size_t i = 0; i < numberOfNodes; i++) {
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
    for (size_t i = 0; i < numberOfNodes; i++) {
      (iterator->second)[i] *= weights[i];
      length += (float)pow((iterator->second)[i], 2.0);
    }
    // normalizing
    length = sqrt(length);
    for (size_t i = 0; i < numberOfNodes; i++) 
      (iterator->second)[i] /= length;
  }

  /*for (uint32_t i = 0; i < (uint32_t)pow(split, maxLevel - 1); i++) {
     //printf("Size of inv file %d: %d\n", i, invertedFiles[i].size());
  }
   printf("\n\n");
  uint32_t l = 0, inL = 0;
  for (uint32_t i = 0; i < numberOfNodes; i++) {
     printf("Node %d, ifl %d, count %d, weight %f || ", i, tree[i].invertedFileLength, counts[i], weights[i]);
    inL++;
    if (inL >= (uint32_t)pow(split, l)) {
      l++;
      inL = 0;
      printf("\n");
    }
  }*/


  struct Temp_pair {
    uint64_t key;
    uint32_t val;
  };

  // synchronize leaf and vector information, everything send to node 0
#if ENABLE_MULTITHREADING && ENABLE_MPI
  MPI::Comm::Barrier(); 
  // get leaf sets
  int leaves = invertedFiles.size();
  if (rank == 0) {
    int totalCount = 0;
    int t;
    for(int i=0; i<procs-1; i++) {
      Comm::Recv(&t, 1, MPI_INT, MPI_ANY_SOURCE, leaves);
      totalCount += t;
    }

    Temp_pair tmpPair;
    Status status;
    for(int i=0; i<totalCount; i++) {
      Comm::Recv(&tmpPair, sizeof(Temp_pair), MPI_BYTES, MPI_ANY_SOURCE, 1, &status);
      invertedFiles[status.MPI_TAG][tmpPair.key] = tmpPair.val;
    }
  }
  else {
    Request sizeRequests[leaves];
    Temp_pair tmpPair;
    for (int i = 0; i < leaves; i++) {
      int fileSize = invertedFiles[i].size();
      sizeRequests[i] = Comm::Isend(&fileSize, 1, MPI_INT, 0, leaves);

      for (auto & p : invertedFiles[i]) {
        tmpPair.key = p.first;
        tmpPair.val = p.second;
        Comm::Send(&tmpPair, sizeof(Temp_pair), MPI_BYTES, 0, 1);
      }
    }

    Request::Waitall(leaves, &sizeRequests[0]);
  }

  MPI::Comm::Barrier();

  //get data vectors
  if (rank == 0) {
    int totalCount = 0;
    int t;
    for(int i=0; i<procs-1; i++) {
      Comm::Recv(&t, 1, MPI_INT, MPI_ANY_SOURCE, 0);
      totalCount += t;
    }
    Status status;
    for(int i=0; i<totalCount; i++) {
      std::vector<float> newVec(numberOfNodes);
      Comm::Recv(&newVec[0], numberOfNodes, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, &status);
      databaseVectors[status.MPI_TAG-1] = newVec;
    }
  }
  else {
    int count = databaseVectors.size();
    Comm::Send(&count, 1, MPI_INT, 0, 0);
    for (auto & p : databaseVectors)
      Comm::Send(&(p.second[0]), numberOfNodes, MPI_FLOAT, 0, p.first + 1);
  }
#endif

  return true;
}


void VocabTree::buildTreeRecursive(uint32_t t, const cv::Mat &descriptors, cv::TermCriteria &tc,
  int attempts, int flags, int currLevel, std::vector<uint32_t> indices, uint32_t maxNode, float_t ratio) {

#if ENABLE_MULTITHREADING && ENABLE_MPI
  int rank = MPI::COMM_WORLD.Get_rank();
#endif

  tree[t].invertedFileLength = descriptors.rows;
  tree[t].level = currLevel;

  // handles the leaves
  if (currLevel == maxLevel - 1) {
    tree[t].firstChildIndex = 0;
    return;
  }

  cv::Mat labels;
  cv::Mat centers;

#if !(ENABLE_MULTITHREADING && ENABLE_MPI && rank!=maxNode)
  std::vector<cv::Mat> groups(split);
  std::vector< std::vector<cv::Mat> > unjoinedGroups(split);
  //for (uint32_t i = 0; i < split; i++)
    //groups[i] = cv::Mat();//cv::Mat(0, descriptors.cols, descriptors.type);
#else
  std::vector<std::vector<uint32_t> > groups(split);
#endif

  // printf("t: %d  rows: %d, counts: ", t, descriptors.rows);

  bool enoughToFill = true;
  if (descriptors.rows >= split) {
    int numIndices = indices.size();
    // gather desired descriptors
#if ENABLE_MULTITHREADING && ENABLE_MPI && (rank!=maxNode || numIndices>0)
  std::vector<cv::Mat> listDescriptors;
    for (uint32_t : indices)
      listDescriptors.push_back(descriptors.row(i));
    cv::Mat compiledDescriptors = vision::merge_descriptors(listDescriptors, true);

    // can we do this like bag_of_words does using mpi for the first node?
    cv::kmeans(compiledDescriptors, split, labels, tc, attempts, flags, centers);
#else
    cv::kmeans(descriptors, split, labels, tc, attempts, flags, centers);
#endif

    for (int i = 0; i < labels.rows; i++) {
      int index = labels.at<int>(i);
#if ENABLE_MULTITHREADING && ENABLE_MPI && rank!=maxNode
      groups[index].push_back(indices[index]);
#else
      //if (groups[index].cols != descriptors.cols)
        //printf("COL MISMATCH\n");
      //groups[index].push_back(descriptors.row(i));
      unjoinedGroups[index].push_back(descriptors.row(i));
#endif
    }
  }
  else {
    // *** THIS SHOULDN'T BE THE CASE, why is kmeans splitting poorly? ****
    enoughToFill = false;
    for (int i = 0; i < descriptors.rows; i++)
#if ENABLE_MULTITHREADING && ENABLE_MPI && rank!=maxNode
      groups[i].push_back(indices[i]);
#else
      //groups[i].push_back(descriptors.row(i));
      unjoinedGroups[i].push_back(descriptors.row(i));
#endif
  }

#if !(ENABLE_MULTITHREADING && ENABLE_MPI && rank!=maxNode)
 for (int i = 0; i<split; i++) {
    if (unjoinedGroups[i].size() > 0)
      /***** This will give memory problems and crash *****/
      groups[i] = vision::merge_descriptors(unjoinedGroups[i], false); 
  }
#endif

  if (indices.size() > 0)
    indices.clear(); // this is only important for the case where mpi has data in the indices, but won't be calling
      // more mpi nodes, so it will want indices to be clear


#if ENABLE_MULTITHREADING && ENABLE_OPENMP
  uint32_t totalChildren = pow(split, currLevel);
  uint32_t maxThreads = omp_get_num_threads();
#endif
  
  // only do omp parallel if not splitting the work to other children
#if ENABLE_MULTITHREADING && ENABLE_OPENMP && totalChildren<maxThreads && !(ENABLE_MULTITHREADING && ENABLE_MPI && rank!=maxNode)
#pragma omp parallel for schedule(dynamic)
#endif
  for (uint32_t i = 0; i < split; i++) {
    uint32_t childLevelIndex = tree[t].levelIndex*split + i;
    uint32_t childIndex = (uint32_t)((pow(split, tree[t].level + 1) - 1) / (split - 1)) + childLevelIndex;
    if (i == 0)
      tree[t].firstChildIndex = childIndex;

#if ENABLE_MULTITHREADING && ENABLE_MPI && rank!=maxNode
    // calculate destination, will give them all nodes to max
    // I'm assuming that the groups are ordered from highest to lowest, that's what the kmeans appears to do
    int numNodes = std::max(1,round(((float)groups[i].size())/ratio));
    int newNode = maxNode - numNodes + 1;
    // send out messages
    Request requests[8];
    requests[0] = Comm::Isend(&childIndex, 1, MPI_INT, newNode, index_tag);
    requests[1] = Comm::Isend(&childLevelIndex, 1, MPI_INT, newNode, levelIndex_tag);
    int indicesCount = groups[i].size();
    requests[2] = Comm::Isend(&indicesCount, 1, MPI_INT, newNode, indicesCount_tag);
    requests[3] = Comm::Isend(&(groups[i][0]), indicesCount, MPI_INT, newNode, indices_tag);

    // send mean_vector
    cvmat_header h;
    h.elem_size = centers.row(i).elemSize();
    h.elem_type = centers.row(i).type();
    h.rows = centers.row(i).rows;
    h.cols = centers.row(i).cols;
    requests[4] = Comm::Isend(&h, sizeof(cvmat_header), MPI_BYTES, newNode, meanHeader_tag);
    requests[5] = Comm::Isend((char *)centers.row(i).ptr(), h.rows * h.cols * h.elem_size, MPI_BYTES, newNode, mean_tag);

    requests[6] = Comm::Isend(&maxNode, 1, MPI_INT, newNode, maxNode_tag);
    int t = currLevel + 1;
    requests[7] = Comm::Isend(&t, 1, MPI_INT, newNode, level_tag);

    Request::Waitall(8, requests);

    // update maxNode
    maxNode = newNode - 1;
#else
    if (enoughToFill)
      tree[childIndex].mean = centers.row(i);
    tree[childIndex].levelIndex = childLevelIndex;
    tree[childIndex].index = childIndex;

    buildTreeRecursive(childIndex, groups[i], tc, attempts, flags, currLevel + 1, indices, maxNode, ratio);
#endif
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
  int rank = MPI::COMM_WORLD.Get_rank();
  int procs = MPI::COMM_WORLD.Get_size();

#if ENABLE_MULTITHREADING && ENABLE_OPENMP
#pragma omp parallel for
#endif
  for (int r = (id>=0? 0:rank); r < descriptors.rows; r+=(id>0?1:procs)) {
#else
#if ENABLE_MULTITHREADING && ENABLE_OPENMP
#pragma omp parallel for
#endif
  for (int r = 0; r < descriptors.rows; r++) {
#endif
    generateVectorHelper(0, descriptors.row(r), vec, possibleMatches, id);
  }

#if ENABLE_MULTITHREADING && ENABLE_MPI
  if(id<0) {
    for(int i=0; i<procs; i++)
    if (i != rank)
      Comm::Send(&vec[0], numberOfNodes, MPI_FLOAT, i, 0);

    std::vector<float> otherVec(numberOfNodes);
    for (int i = 0; i<procs; i++)
    if (i != rank) {
      Comm::Recv(&otherVec[0], numberOfNodes, MPI_FLOAT, MPI_ANY_SOURCE, 0);
      for (int j = 0; j < numberOfNodes; j++)
        vec[i] += otherVec[j];
    }
  }
#endif

  // if shouldWeight is true then weight all values in the vector and normalize
  if (shouldWeight) {
    float length = 0; // for normalizing
#if ENABLE_MULTITHREADING && ENABLE_OPENMP
#pragma omp parallel for
#endif
    for (uint32_t i = 0; i < numberOfNodes; i++) {
      vec[i] *= weights[i];
      length += vec[i] * vec[i];
    }
    length = sqrt(length);
#if ENABLE_MULTITHREADING && ENABLE_OPENMP
#pragma omp parallel for
#endif
    for (uint32_t i = 0; i < numberOfNodes; i++) {
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
    double max = tree[maxChild].mean.dims == 0 ? 0 : descriptor.dot(tree[maxChild].mean);
    //double max = descriptor.dot(tree[maxChild].mean);
    
    for (uint32_t i = 1; i < split; i++) {
      if (tree[nodeIndex].invertedFileLength == 0)
        continue;
      uint32_t childIndex = tree[nodeIndex].firstChildIndex + i;
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
  std::vector<float> vec = generateVector(descriptorsf, true, possibleMatches);

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
    for (uint32_t i = 0; i < numberOfNodes; i++) {
      float t = vec[i] - (databaseVectors[elem])[i];
      score += t*t;
      // std::cout << vec[i] << std::endl;
      //l1norm += abs(vec[i] * (databaseVectors[elem])[i]);
    }
    //values[elem] = l1norm;
    //values.insert(elem, l1norm));

    values.push_back(matchPair(elem, sqrt(score)));
  }

  int keep = 10;
  std::sort(values.begin(), values.end(), comparer);
  if (values.size() > keep) {
    std::vector<matchPair>::iterator it = values.begin();
    values.erase(it + keep, values.end());
  }

  // aggregate everything into node 0
#if ENABLE_MULTITHREADING && ENABLE_MPI
  int rank = MPI::COMM_WORLD.Get_rank();
  int procs = MPI::COMM_WORLD.Get_size();

  if (rank == 0) {
    int totalCount = 0;
    int t;
    for (int i = 0; i<procs - 1; i++) {
      Comm::Recv(&t, 1, MPI_INT, MPI_ANY_SOURCE, 0);
      totalCount += t;
    }

    Status status;
    float score;
    for (int i = 0; i<totalCount; i++) {
      Comm::Recv(&score, 1, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, &status);
      values.push_back(matchPair(status.MPI_TAG-1, score));
    }

    // this may result in duplicate entries
    // will have to decide if that's a problem and if its worth fixing
    std::sort(values.begin(), values.end(), comparer);
    it = values.begin();
    values.erase(it+keep, values.end());
  }
  else {
    Request sizeRequests[leaves];
    Temp_pair tmpPair;
    int tmpCount = values.size();
    sizeRequests[i] = Comm::Isend(&tmpCount, 1, MPI_INT, 0, 0);

    for (matchPair p : values)
      Comm::Send(&p.second, 1, MPI_FLOAT, 0, p.first+1);
    Request::Waitall(leaves, &sizeRequests[0]);
  }

#endif

  // printf("%d matches\n", values.size());
  // add all images in order or match
  for (matchPair m : values){
    match_result->matches.push_back(m.first);
    match_result->tfidf_scores.push_back(m.second);
    // std::cout << m.second << std::endl;
  }

  // add in matches, just do 2 for now
  //possibleMatches.size() / 10.0
  /*for (int i = 0; i < 1; i++) {
    std::set<matchPair>::iterator top = values.begin();
    match_result->matches.push_back(top->first);
    match_result->tfidf_scores.push_back(top->second);
  }*/
  //match_result->matches.push_back(0);

  return (std::shared_ptr<MatchResultsBase>)match_result;
}

uint32_t VocabTree::tree_splits() const {
	return split;
}

uint32_t VocabTree::tree_depth() const {
	return maxLevel;
}
