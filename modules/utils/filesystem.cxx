#include "filesystem.hpp"

#include <sys/stat.h>
#include <fcntl.h>
#include <fstream>
#include <iomanip>

#include <boost/filesystem.hpp>

namespace filesystem {
		
	size_t filesize(const std::string &filename) {
		struct stat sb;
	    int fd = open(filename.c_str(), O_RDWR);
		fstat(fd, &sb);
		size_t max_off = sb.st_size;
		close(fd);
		return max_off;
	}

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
		uint64_t elem_size, elem_type;
		uint32_t rows, cols;
	};

	void write_cvmat(const std::string &fname, const cv::Mat &data) {
		std::ofstream ofs(fname.c_str(), std::ios::binary | std::ios::trunc);
		cvmat_header h;
		h.elem_size = data.elemSize();
		h.elem_type = data.type();
		h.rows = data.rows;
		h.cols = data.cols;
		ofs.write((char *)&h, sizeof(cvmat_header));
		ofs.write((char *)data.ptr(), h.rows * h.cols * h.elem_size);

	}

	bool load_cvmat(const std::string &fname, cv::Mat &data) {
		if(!file_exists(fname)) return false;
		std::ifstream ifs(fname.c_str(), std::ios::binary);
		cvmat_header h;
		ifs.read((char *)&h, sizeof(cvmat_header));
		data.create(h.rows, h.cols, h.elem_type);
		ifs.read((char *)data.ptr(), h.rows * h.cols * h.elem_size);
		return true;
	}

	void write_bow(const std::string &fname, const std::vector<std::pair<uint32_t, float > > &data) {
		std::ofstream ofs(fname.c_str(), std::ios::binary | std::ios::trunc);
		uint32_t dim0 = data.size();
		ofs.write((char *)&dim0, sizeof(uint32_t));
		ofs.write((char *)&data[0], sizeof(std::pair<uint32_t, float >) * dim0);
	}

	bool load_bow(const std::string &fname, std::vector<std::pair<uint32_t, float > > &data) {
		if(!file_exists(fname)) return false;
		std::ifstream ifs(fname.c_str(), std::ios::binary);
		uint32_t dim0;
		ifs.read((char *)&dim0, sizeof(uint32_t));
		data.resize(dim0);
		ifs.read((char *)&data[0], sizeof(std::pair<uint32_t, float >) * dim0);
		return true;
	}
}