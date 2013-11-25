vocabtree
=========

Multinode, multicore large scale image search.


Build Instructions
===================

Requires a c++11 compatible compiler and CMake to build.

Supported OS

* Linux
* Windows

Required Dependencies

* OpenCV (https://github.com/Itseez/opencv)
* Boost (http://www.boost.org/)

Optional Dependencies

* OpenMP - required for multithreading support
* MPI - required for multinode support (ideally should be used with OpenMP)

Building
----------

Create a build directory and then run cmake from that directory pointing to the root source directory.

Ex: From the root source directory:

    mkdir build
    cd build
    cmake ..
    make

Binaries are located in your build directory under bin.

Unix Specific
----------
If on Unix, and yasm and MPI are available, there is an automatically enabled option to use the FASTANN and 
FASTCLUSTER libraries (http://www.robots.ox.ac.uk/~vgg/software/fastanncluster/).  These will be automatically 
downloaded and compiled and enables out of core multinode kmeans clustering.  Otherwise, the kmeans 
implementation bundled with OpenCV will be used, which may or may not be multithreaded (single node) 
depending on your OpenCV build settings.


Documentation
===================

Documentation can be built by running (requires doxygen)

    make doc

Documentation can also be found here:

http://psastras.github.io/vocabtree/


Sample Usage
===================

Compute Features
-----------------------

```c++
// Construct a simple dataset from data located in /home/foo/data/ and store the database at 
// /home/foo/data/database.bin.  A SimpleDataset is an implementation of a generic dataset which
// expects images to be found in /home/foo/data/images/.  It also expects sift descriptors to be 
// stored in the corresponding SimpleImage::feature_path("descriptor").
// If the dataset binary exists it will automatically be loaded.
// You can implement your own dataset class if necessary by implementing Dataset.

SimpleDataset simple_dataset("/home/foo/data/", "/home/foo/data/database.bin");
for (int64_t i = 0; i < simple_dataset.num_images(); i++) {
	std::shared_ptr<SimpleDataset::SimpleImage> image = std::static_pointer_cast<SimpleDataset::SimpleImage>(
		simple_dataset.image(i));
	if (image == nullptr) continue;
	
	const std::string &keypoints_location = simple_dataset.location(image->feature_path("keypoints"));
	const std::string &descriptors_location = simple_dataset.location(image->feature_path("descriptors"));
	if (filesystem::file_exists(keypoints_location) && filesystem::file_exists(descriptors_location)) continue;
	
	const std::string &image_location = simple_dataset.location(image->location());
	
	if (!filesystem::file_exists(image_location)) continue;
	
	cv::Mat im = cv::imread(image_location, cv::IMREAD_GRAYSCALE);
	
	cv::Mat keypoints, descriptors;
	if (!vision::compute_sparse_sift_feature(im, nullptr, keypoints, descriptors)) continue;
	
	filesystem::create_file_directory(keypoints_location);
	filesystem::create_file_directory(descriptors_location);
	
	filesystem::write_cvmat(keypoints_location, keypoints);
	filesystem::write_cvmat(descriptors_location, descriptors);
}
```

Train and Search a Tree
-----------------------

```c++
    SimpleDataset simple_dataset("/home/foo/data/", "/home/foo/data/database.bin");
    
    // Train a vocabulary tree on the dataset
    VocabTree vt;
    std::shared_ptr<VocabTree::TrainParams> train_params = std::make_shared<VocabTree::TrainParams>();
    train_params->depth = 4; // vocabulary tree with max depth of 4
    train_params->split = 4; // vocabulary tree with split factor of 4
    
    // train on a random sampling of 128 images from the dataset
    vt.train(simple_dataset, train_params, simple_dataset.random_images(128));
    
    // Search for image zero in the dataset
    std::shared_ptr<VocabTree::MatchResults> matches =
        std::static_pointer_cast<VocabTree::MatchResults>(vt.search(simple_dataset, nullptr, simple_dataset.image(0)));
    
    // Print out the matches to image zero
    for (uint64_t id : matches->matches)
        std::cout << id << " ";
    std::cout << std::endl;
```
