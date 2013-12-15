#include <utils/filesystem.hpp>
#include <sstream>
#include <iomanip>

// converts a video into a bunch of images
int main(int argc, char *argv[]) {
	std::string input_video_dir(argv[1]);
	std::string output_data_dir(argv[2]);

	uint64_t skip = 10;

	const std::vector<std::string> &videos = filesystem::list_files(input_video_dir, ".mp4");
	size_t num_videos = MIN(videos.size(), 16);

	#pragma omp parallel for schedule(dynamic)
	for(size_t i=0; i<num_videos; i++) {
		std::cout << videos[i] << std::endl;
		cv::VideoCapture vc(videos[i]);
		uint32_t num_frames = vc.get(cv::CAP_PROP_FRAME_COUNT);
		std::string video_name = filesystem::basename(videos[i], false);
		for(uint32_t i=0; i<num_frames; i+=skip) {
			std::stringstream ss;
			ss << output_data_dir << "/images/" << video_name << "/" << std::setw(6) << std::setfill('0') << i / skip<< ".jpg";
			if(filesystem::file_exists(ss.str())) continue;
			filesystem::create_file_directory(ss.str());
			cv::Mat frame;
			vc.set(cv::CAP_PROP_POS_FRAMES, i);
			vc >> frame;
		
			cv::imwrite(ss.str(), frame);
		}
	}

	return 0;
}