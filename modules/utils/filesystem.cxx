#include "filesystem.hpp"

#include <sys/stat.h>
#include <fcntl.h>
#include <fstream>
#include <iomanip>

#include <boost/filesystem.hpp>

namespace filesystem {

	bool file_exists(const std::string& name) {
	  struct stat buffer;
	  return (stat (name.c_str(), &buffer) == 0);
	}

	void create_file_directory(const std::string &absfilepath) {
		boost::filesystem::path p(absfilepath.c_str());
		boost::filesystem::path d = p.parent_path();
		if(!boost::filesystem::exists(d)) {
			boost::filesystem::create_directories(d);
		}
	}
	struct cvmat_header {
		uint64_t elem_size;
		int32_t elem_type;
		uint32_t rows, cols;
	};

	bool write_cvmat(const std::string &fname, const cv::Mat &data) {
		std::ofstream ofs(fname.c_str(), std::ios::binary | std::ios::trunc);
		cvmat_header h;
		h.elem_size = data.elemSize();
		h.elem_type = data.type();
		h.rows = data.rows;
		h.cols = data.cols;
		ofs.write((char *)&h, sizeof(cvmat_header));
		ofs.write((char *)data.ptr(), h.rows * h.cols * h.elem_size);
		return (ofs.rdstate() & std::ofstream::failbit) == 0;
	}

	bool load_cvmat(const std::string &fname, cv::Mat &data) {
		if(!file_exists(fname)) return false;
		std::ifstream ifs(fname.c_str(), std::ios::binary);
		cvmat_header h;
		ifs.read((char *)&h, sizeof(cvmat_header));
		if (h.rows == 0 || h.cols == 0) return false;
		data.create(h.rows, h.cols, h.elem_type);
		ifs.read((char *)data.ptr(), h.rows * h.cols * h.elem_size);
		return (ifs.rdstate() & std::ifstream::failbit) == 0;
	}

	bool write_sparse_vector(const std::string &fname, const std::vector<std::pair<uint32_t, float > > &data) {
		std::ofstream ofs(fname.c_str(), std::ios::binary | std::ios::trunc);
		uint32_t dim0 = data.size();
		ofs.write((char *)&dim0, sizeof(uint32_t));
		ofs.write((char *)&data[0], sizeof(std::pair<uint32_t, float >) * dim0);
		return (ofs.rdstate() & std::ofstream::failbit) == 0;
	}

	bool load_sparse_vector(const std::string &fname, std::vector<std::pair<uint32_t, float > > &data) {
		if(!file_exists(fname)) return false;
		std::ifstream ifs(fname.c_str(), std::ios::binary);
		uint32_t dim0;
		ifs.read((char *)&dim0, sizeof(uint32_t));
		data.resize(dim0);
		ifs.read((char *)&data[0], sizeof(std::pair<uint32_t, float >) * dim0);
		return (ifs.rdstate() & std::ifstream::failbit) == 0;
	}

	std::vector<std::string> list_files(const std::string &path, const std::string &ext, bool recursive) {
		boost::filesystem::path input_path(path);
		std::vector<std::string> file_list;
		if (boost::filesystem::exists(input_path) && boost::filesystem::is_directory(input_path)) {
			if(recursive) {
				boost::filesystem::recursive_directory_iterator end;
				for (boost::filesystem::recursive_directory_iterator it(input_path); it != end; ++it) {
					if (!boost::filesystem::is_directory(*it)) {
						if ((ext.length() > 0 && boost::filesystem::extension(*it) == boost::filesystem::path(ext)) ||
							ext.length() == 0) {
								file_list.push_back(it->path().string());
						}
					}
				}
			} else {
				boost::filesystem::directory_iterator end;
				for (boost::filesystem::directory_iterator it(input_path); it != end; ++it) {
					if (!boost::filesystem::is_directory(*it)) {
						if ((ext.length() > 0 && boost::filesystem::extension(*it) == boost::filesystem::path(ext)) ||
							ext.length() == 0) {
								file_list.push_back(it->path().string());
						}
					}
				}
			}
		}
		return file_list;
	}

	std::string basename(const std::string &path, bool include_extension) {
		if (!include_extension) return boost::filesystem::basename(boost::filesystem::path(path));

		return boost::filesystem::basename(boost::filesystem::path(path)) + boost::filesystem::extension(path);
	}

	bool write_text(const std::string &fname, const std::string &text) {
		std::ofstream ofs(fname, std::ios::trunc);
		ofs.write(text.c_str(), text.size());
		return (ofs.rdstate() & std::ofstream::failbit) == 0;
	}

	bool write_vector(const std::string &fname, const std::vector<float> &data) {
    create_file_directory(fname);

    std::ofstream ofs(fname.c_str(), std::ios::binary | std::ios::trunc);
    ofs.write((char *)&data[0], data.size()*sizeof(float));
    return (ofs.rdstate() & std::ofstream::failbit) == 0;
	}

  bool load_vector(const std::string &fname, std::vector<float> &data) {
    if (!file_exists(fname)) return false;

    std::ifstream ifs(fname.c_str(), std::ios::binary);
    ifs.read((char *)&data[0], data.size()*sizeof(float));
    return (ifs.rdstate() & std::ifstream::failbit) == 0;
	}
}