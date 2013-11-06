#pragma once

#include "image.hpp"
#include <memory>

class Dataset {

public:
	Dataset(const std::string &base_location);
	Dataset(const std::string &base_location, const std::string &db_data_location);

	virtual ~Dataset();

	virtual bool write(const std::string &db_data_location) = 0;
	virtual bool read (const std::string &db_data_location) = 0;
	virtual std::shared_ptr<Image> image(uint64_t id) 		= 0;

protected:
	std::string	data_directory;
};

class SimpleDataset : public Dataset {

public:

	class SimpleImage : public Image {
		public:
			SimpleImage(const std::string &path, uint64_t imageid) : Image(imageid) { 
				image_path = path;
			}	

			std::string feature_path(const std::string &feat_name) const {
				return "/simple/feats/" + feat_name + "/"; 
			}

		protected:
			std::string image_path;
	};

	SimpleDataset(const std::string &base_location);
	SimpleDataset(const std::string &base_location, const std::string &db_data_location);
	~SimpleDataset();

	bool write(const std::string &db_data_location);
	std::shared_ptr<Image> image(uint64_t id);

private:
	
	void construct_dataset();

};