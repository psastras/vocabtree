#include "dataset.hpp"


Dataset::Dataset(const std::string &base_location) {
	data_directory = base_location;
}

Dataset::Dataset(const std::string &base_location, const std::string &db_data_location) {
	data_directory = base_location;
}

Dataset::~Dataset() { }


SimpleDataset::SimpleDataset(const std::string &base_location) : Dataset(base_location) { 
	construct_dataset();
}

SimpleDataset::SimpleDataset(const std::string &base_location, const std::string &db_data_location) 
	: Dataset(base_location, db_data_location) { }

SimpleDataset::~SimpleDataset() { }

std::shared_ptr<Image> SimpleDataset::image(uint64_t id) {
	return nullptr;
}

void SimpleDataset::construct_dataset() {

}