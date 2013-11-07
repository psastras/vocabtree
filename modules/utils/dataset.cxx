#include "dataset.hpp"
#include "filesystem.hpp"
#include "vision.hpp"

#include <fstream>

Dataset::Dataset(const std::string &base_location) {
	data_directory = base_location;
}

Dataset::Dataset(const std::string &base_location, const std::string &db_data_location) {
	data_directory = base_location;
}

Dataset::~Dataset() { }

std::string Dataset::location() const {
	return data_directory;
}

std::string Dataset::location(const std::string &relative_path) const {
	return data_directory + "/" + relative_path;
}


std::vector< std::shared_ptr<const Image> > Dataset::all_images() {
	std::vector< std::shared_ptr< const Image> > images(this->num_images());
	for (uint64_t i = 0; i < this->num_images(); i++) {
		images[i] = this->image(i);
	}
	return images;
}

std::ostream& operator<< (std::ostream &out, const Dataset &dataset) {
	out << "Dataset location: " << dataset.location() << ", number of images: " << dataset.num_images();
	return out;
}

SimpleDataset::SimpleDataset(const std::string &base_location) : Dataset(base_location) { 
	this->construct_dataset();
}

SimpleDataset::SimpleDataset(const std::string &base_location, const std::string &db_data_location) 
	: Dataset(base_location, db_data_location) {
		if (filesystem::file_exists(db_data_location)) {
			this->read(db_data_location);
		}
		else {
			this->construct_dataset();
			this->write(db_data_location);
		}
}

SimpleDataset::~SimpleDataset() { }

std::shared_ptr<Image> SimpleDataset::image(uint64_t id) {
	const std::string &image_path = id_image_map.right.at(id);

	std::shared_ptr<Image> current_image = std::make_shared<SimpleImage>(image_path, id);
	return current_image;
}

void SimpleDataset::construct_dataset() {
	const std::vector<std::string> &image_file_paths = filesystem::list_files(data_directory + "/images/", ".jpg");
	for (size_t i = 0; i < image_file_paths.size(); i++) {
		id_image_map.insert(boost::bimap<std::string, uint64_t>::value_type( filesystem::basename(image_file_paths[i], true), i));
	}
}

bool SimpleDataset::read(const std::string &db_data_location) {
	if (!filesystem::file_exists(db_data_location)) return false;
	std::ifstream ifs(db_data_location, std::ios::binary);
	
	uint64_t num_images;
	ifs.read((char *)&num_images, sizeof(uint64_t));

	for (uint64_t i = 0; i < num_images; i++) {
		
		uint64_t image_id;
		uint16_t length;
		ifs.read((char *)&image_id, sizeof(uint64_t));
		ifs.read((char *)&length, sizeof(uint16_t));

		std::string image_location;
		image_location.resize(length);
		
		ifs.read((char *)&image_location[0], sizeof(char)* length);
		std::shared_ptr<const SimpleImage> simage = std::make_shared<const SimpleImage>(image_location, image_id);
		this->add_image(simage);

	}
	return (ifs.rdstate() & std::ifstream::failbit) == 0;
}

bool SimpleDataset::write(const std::string &db_data_location) {
	filesystem::create_file_directory(db_data_location);
	
	std::ofstream ofs(db_data_location, std::ios::binary | std::ios::trunc);
	uint64_t num_images = this->num_images();
	ofs.write((const char *)&num_images, sizeof(uint64_t));

	for (uint64_t i = 0; i < this->num_images(); i++) {
		std::shared_ptr<SimpleDataset::SimpleImage> image = std::static_pointer_cast<SimpleDataset::SimpleImage>(this->image(i));
		const std::string &image_location = image->location();
		uint64_t image_id = image->id;
		uint16_t length = image_location.size();
		ofs.write((const char *)&image_id, sizeof(uint64_t));
		ofs.write((const char *)&length, sizeof(uint16_t));
		ofs.write((const char *)&image_location[0], sizeof(char)* length);
	}

	return (ofs.rdstate() & std::ofstream::failbit) == 0;
}

uint64_t SimpleDataset::num_images() const {
	return id_image_map.size();
}

SimpleDataset::SimpleImage::SimpleImage(const std::string &path, uint64_t imageid) : Image(imageid) {
	image_path = path;
}

std::string SimpleDataset::SimpleImage::feature_path(const std::string &feat_name) const {
	std::stringstream ss;
	ss << "/feats/" << feat_name << "/" << std::setw(6) << std::setfill('0') << id << "." << feat_name;
	return ss.str();
}

std::string SimpleDataset::SimpleImage::location() const {
	return image_path;
}

bool SimpleDataset::add_image(const std::shared_ptr<const Image> &image) {
	if (id_image_map.right.find(image->id) != id_image_map.right.end()) return false;
	const std::shared_ptr<const SimpleDataset::SimpleImage> simage = std::static_pointer_cast<const SimpleDataset::SimpleImage>(image);
	id_image_map.insert(boost::bimap<std::string, uint64_t>::value_type(simage->location(), simage->id));
	return true;
}