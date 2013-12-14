#pragma once

#include "image.hpp"
#include "cache.hpp"
#include "config.hpp"

#include <memory>
#include <boost/bimap.hpp>
#include <sstream>
#include <iomanip>

#if !(_MSC_VER && !__INTEL_COMPILER)
typedef bow_ring_priority_cache_t bow_feature_cache_t;
typedef vec_ring_priority_cache_t vec_feature_cache_t;
#endif

/// The Dataset class is an abstract wrapper describing a dataset.  A dataset consiste of the actual
/// data, plus a way to convert the images, or frames of a video into an integer index.  The dataset
/// should at minimum provide an easy way to map image paths to unique integers.  For a sample implementation
/// of a Dataset see the SimpleDataset class.
///
/// Combined with the Image class implementation, a Dataset + Image provides a way to find relevant paths
/// for features and images.  Note that the implementation of a Dataset or Image class should implement a relative
/// path to the Image data, with the absolute path being interchangebale.
class Dataset {

public:
	/// Constructs a dataset given a base location.  An example base location might be
	/// /c/data/.  Given this base location, an implementation of the Dataset should 
	/// find all the data and construct a mapping between the data and the id, for example 
	/// by searching through base_location + /images/.
	Dataset(const std::string &base_location);

	/// Loads a dataset from the db_data_location.  The base_location provides the absolute
	/// path of data.
	Dataset(const std::string &base_location, const std::string &db_data_location);

	virtual ~Dataset();

	/// Writes the dataset mapping to the input data location.  Returns true if successful, false
	/// otherwise.
	virtual bool write(const std::string &db_data_location) = 0;

	/// Reads the dataset mapping from the input data location.  Returns true if successful, false
	/// otherwise.
	virtual bool read (const std::string &db_data_location) = 0;

	/// Given a unique integer ID, returns an Image associated with that ID.
	virtual PTR_LIB::shared_ptr<Image> image(uint64_t id) const		= 0;

	/// Returns the number of images in the dataset.
	virtual uint64_t num_images() const = 0;

	/// Returns the absolute path of the data directory
	std::string location() const;

	/// Returns the absolute path of the file (appends the file path to the database path).
	std::string location(const std::string &relative_path) const;

	/// Adds the given image to the database, if there is an id collision, will not add the image and 
	/// return false, otherwise returns true.
	virtual bool add_image(const PTR_LIB::shared_ptr<const Image> &image) = 0 ;

	/// Returns a vector of all images in the dataset.
	std::vector<  PTR_LIB::shared_ptr< const Image> > all_images() const;

	/// Returns a vector of random images in the dataset of size count.
	std::vector<  PTR_LIB::shared_ptr< const Image> > random_images(size_t count) const;

	/// @TODO: Shards the dataset to the new input locations, and returns the sharded datasets
	std::vector<Dataset> shard(const std::vector<std::string> &new_locations);

	virtual numerics::sparse_vector_t load_bow_feature(uint64_t id) const = 0;
	virtual std::vector<float> load_vec_feature(uint64_t id) const = 0;

protected:
	std::string	data_directory;  /// Holds the absolute path of the data.
};

/// Prints out information about the dataset.
std::ostream& operator<< (std::ostream &out, const Dataset &dataset);

/// SimpleDataset is a sample implementation of a Dataset, where the data is stored as JPEG
/// images in a single folder called images/ and features are stored in a folder feats/<feat_name>.
/// For example, given a base absolute path of /c/data/.  Image data is found in /c/data/images and
/// sift features are found in /c/data/feats/sift/.
class SimpleDataset : public Dataset {

public:

	/// SimpleImage class used with the SimpleDataset class.  Features are stored in
	/// <data_dir>/<feats>/<feat_name>
	class SimpleImage : public Image {
		public:
			/// Constructs a SimpleImage given the Image relative path in the Dataset directory,
			/// and a corresponding unique image ID.
			SimpleImage(const std::string &path, uint64_t imageid);
			
			/// Returns the corresponding feature path given a feature name (ex. "sift").
			std::string feature_path(const std::string &feat_name) const;

			/// Returns the image location relative to the database data directory.
			std::string location() const;
		
		protected:
			std::string image_path; /// Stores the relative image path.
	};

	/// Creates a simple dataset from the images in base_location/images.  It is recommended
	/// to then call write(...) to save the dataset so that it does not have to traverse the HDD
	/// everytime we load the dataset.
	SimpleDataset(const std::string &base_location, size_t cache_size = 0);
	
	/// If a dataset file is location at db_data_location, will load that file from.  Otherwise,
	/// this will create the dataset from base_location/images and call write(db_data_location).
	SimpleDataset(const std::string &base_location, const std::string &db_data_location, size_t cache_size = 0);
	~SimpleDataset();

	/// Writes the SimpleDataset out to the specified file.  If the containing directory does not 
	/// exist, it will be automatically created.  The Dataset data is stored in a binary format
	/// with num_images() entries of the form uint64_t, uint16_t, char * corresponding to an
	/// image id, string length, and the image location string respectively.  Returns true if 
	/// success, fail otherwise (checks the ofstream error bit).
	bool write(const std::string &db_data_location);
	
	/// Reads the specified SimpleDataset.  See write(const std::string &db_data_location) for
	/// more information about the binary format.  Returns true if success, false otherwise. 
	/// (checks the ifstream error bit).
	bool read(const std::string &db_data_location);

	/// Given a unique integer ID, returns an Image associated with that ID.
	PTR_LIB::shared_ptr<Image> image(uint64_t id) const;

	/// Adds the given image to the database, if there is an id collision, will not add the image and 
	/// return false, otherwise returns true.
	bool add_image(const PTR_LIB::shared_ptr<const Image> &image);

	/// Returns the number of images in the dataset.
	uint64_t num_images() const;

	/// Returns the corresponding feature path given a feature name (ex. "sift").
	numerics::sparse_vector_t load_bow_feature(uint64_t id) const;
	std::vector<float> load_vec_feature(uint64_t id) const;

#if !(_MSC_VER && !__INTEL_COMPILER)
	PTR_LIB::shared_ptr<bow_feature_cache_t> cache();
#endif

private:
	
	/// Constructs the dataset an fills in the image id map.
	numerics::sparse_vector_t load_bow_feature_cache(uint64_t id) const;
	std::vector<float> load_vec_feature_cache(uint64_t id) const;

	void construct_dataset();

	boost::bimap<std::string, uint64_t> id_image_map; /// Map which holds the image path and id
#if !(_MSC_VER && !__INTEL_COMPILER)
	PTR_LIB::shared_ptr<bow_feature_cache_t> bow_feature_cache;
	PTR_LIB::shared_ptr<vec_feature_cache_t> vec_feature_cache;
#endif

};