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


std::vector< PTR_LIB::shared_ptr<const Image> > Dataset::all_images() const {
	std::vector< PTR_LIB::shared_ptr< const Image> > images(this->num_images());
	for (uint64_t i = 0; i < this->num_images(); i++) {
		images[i] = this->image(i);
	}
	return images;
}

std::vector< PTR_LIB::shared_ptr<const Image> > Dataset::random_images(size_t count) const {
	std::vector< PTR_LIB::shared_ptr< const Image> > all = this->all_images();
	std::random_shuffle(all.begin(), all.end());
	std::vector< PTR_LIB::shared_ptr< const Image> > images(all.begin(), all.begin() + count);
	return images;
}

std::ostream& operator<< (std::ostream &out, const Dataset &dataset) {
	out << "Dataset location: " << dataset.location() << ", number of images: " << dataset.num_images();
	return out;
}

SimpleDataset::SimpleDataset(const std::string &base_location, size_t cache_size) : Dataset(base_location) { 
	this->construct_dataset();
#if !(_MSC_VER && !__INTEL_COMPILER)
	if(cache_size > 0) {
		bow_feature_cache = PTR_LIB::make_shared<bow_feature_cache_t>(
			std::function<numerics::sparse_vector_t(uint64_t)>(std::bind(&SimpleDataset::load_bow_feature_cache, this, std::placeholders::_1)),
			 cache_size);
	}
#endif
}

SimpleDataset::SimpleDataset(const std::string &base_location, const std::string &db_data_location, size_t cache_size) 
	: Dataset(base_location, db_data_location) {
	if (filesystem::file_exists(db_data_location)) {
		this->read(db_data_location);
	}
	else {
		this->construct_dataset();
		this->write(db_data_location);
	}
#if !(_MSC_VER && !__INTEL_COMPILER)
	if(cache_size > 0) {
		bow_feature_cache = PTR_LIB::make_shared<bow_feature_cache_t>(
			std::function<numerics::sparse_vector_t(uint64_t)>(std::bind(&SimpleDataset::load_bow_feature_cache, this, std::placeholders::_1)),
			 cache_size);
	}
#endif
}

SimpleDataset::~SimpleDataset() { }

PTR_LIB::shared_ptr<Image> SimpleDataset::image(uint64_t id) const {
	const std::string &image_path = id_image_map.right.at(id);

	PTR_LIB::shared_ptr<Image> current_image = PTR_LIB::make_shared<SimpleImage>(image_path, id);
	return current_image;
}

void SimpleDataset::construct_dataset() {
	const std::vector<std::string> &image_file_paths = filesystem::list_files(data_directory + "/images/", ".jpg");
	for (size_t i = 0; i < image_file_paths.size(); i++) {
		id_image_map.insert(boost::bimap<std::string, uint64_t>::value_type( image_file_paths[i].substr(data_directory.size(), image_file_paths[i].size() - data_directory.size()), i));
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
		PTR_LIB::shared_ptr<const SimpleImage> simage = PTR_LIB::make_shared<const SimpleImage>(image_location, image_id);
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
		PTR_LIB::shared_ptr<SimpleDataset::SimpleImage> image = std::static_pointer_cast<SimpleDataset::SimpleImage>(this->image(i));
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
	uint32_t level0 = id >> 20;
	uint32_t level1 = (id - (level0 << 20)) >> 10;

	std::stringstream ss;
	ss << "/feats/" << feat_name << "/" << 
		std::setw(4) << std::setfill('0') << level0 << "/" <<
		std::setw(4) << std::setfill('0') << level1 << "/" <<
		std::setw(9) << std::setfill('0') << id << "." << feat_name;
		
	return ss.str();
}

std::string SimpleDataset::SimpleImage::location() const {
	return image_path;
}

numerics::sparse_vector_t SimpleDataset::load_bow_feature(uint64_t id) const {
#if !(_MSC_VER && !__INTEL_COMPILER)
	if(bow_feature_cache) {
		return (*bow_feature_cache)(id);
	} else {
#endif
		return load_bow_feature_cache(id);
#if !(_MSC_VER && !__INTEL_COMPILER)
	}
#endif
}

std::vector<float> SimpleDataset::load_vec_feature(uint64_t id) const {
#if !(_MSC_VER && !__INTEL_COMPILER)
	if(vec_feature_cache) {
		return (*vec_feature_cache)(id);
	} else {
#endif
		return load_vec_feature_cache(id);
#if !(_MSC_VER && !__INTEL_COMPILER)
	}
#endif
}

numerics::sparse_vector_t SimpleDataset::load_bow_feature_cache(uint64_t id) const {
	numerics::sparse_vector_t bow_descriptors;
	uint32_t level0 = id >> 20;
	uint32_t level1 = (id - (level0 << 20)) >> 10;
	std::stringstream ss;
	ss <<  this->location() << "/feats/" << "bow_descriptors" << "/" << 
	std::setw(4) << std::setfill('0') << level0 << "/" <<
	std::setw(4) << std::setfill('0') << level1 << "/" <<
	std::setw(9) << std::setfill('0') << id << "." << "bow_descriptors";
	std::string location = ss.str();
	if (!filesystem::file_exists(location)) return bow_descriptors;	
	filesystem::load_sparse_vector(location, bow_descriptors);
	return bow_descriptors;
}

std::vector<float> SimpleDataset::load_vec_feature_cache(uint64_t id) const {
	std::vector<float> vec_feature;
	uint32_t level0 = id >> 20;
	uint32_t level1 = (id - (level0 << 20)) >> 10;
	std::stringstream ss;
	ss <<  this->location() << "/feats/" << "datavec" << "/" << 
	std::setw(4) << std::setfill('0') << level0 << "/" <<
	std::setw(4) << std::setfill('0') << level1 << "/" <<
	std::setw(9) << std::setfill('0') << id << "." << "datavec";
	std::string location = ss.str();
	if (!filesystem::file_exists(location)) return vec_feature;	
	filesystem::load_vector(location, vec_feature);
	return vec_feature;
}

bool SimpleDataset::add_image(const PTR_LIB::shared_ptr<const Image> &image) {
	if (id_image_map.right.find(image->id) != id_image_map.right.end()) return false;
	const PTR_LIB::shared_ptr<const SimpleDataset::SimpleImage> simage = std::static_pointer_cast<const SimpleDataset::SimpleImage>(image);
	id_image_map.insert(boost::bimap<std::string, uint64_t>::value_type(simage->location(), simage->id));
	return true;
}

#if !(_MSC_VER && !__INTEL_COMPILER)
PTR_LIB::shared_ptr<bow_feature_cache_t> SimpleDataset::cache() {
	return bow_feature_cache;
}
#endif

// std::vector<char> Dataset::load_data(const std::string &filename) {
// 	std::ifstream input(filename, std::ios::binary);
//     // copies all data into buffer
//     std::vector<char> buffer((
//             std::istreambuf_iterator<char>(input)), 
//             (std::istreambuf_iterator<char>()));
//     return buffer;
// }