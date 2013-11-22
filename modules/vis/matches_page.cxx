#include "matches_page.hpp"

#include <utils/filesystem.hpp>

#include <sstream>
#include <iomanip>

const static std::string s_stylesheet_name = "style.css";

MatchesPage::MatchesPage(uint32_t max_matches_per_page, uint32_t max_images_per_match) {
	max_matches_per_page_ = max_matches_per_page;
	max_images_per_match_ = max_images_per_match;
}

MatchesPage::~MatchesPage() {

}

void MatchesPage::add_match(uint32_t query_id, std::vector<uint64_t> &match_ids, const Dataset &dataset) {
	std::stringstream html_string;
	html_string << "<table><tr>";
	html_string << "<td><img src='" << dataset.location(dataset.image(query_id)->location()) << "' /></td><td> </td>";
	for(size_t i=0; i< MIN(match_ids.size(), max_images_per_match_); i++) {
		std::shared_ptr<Image> image =	dataset.image(match_ids[i]);
		const std::string &impath = dataset.location(image->location());
		html_string << "<td><img src='" << impath << "' /></td>";
	}
	html_string << "</tr></table>";
	html_strings.push_back(html_string.str());
}

void MatchesPage::write(const std::string &folder) const {
	if(!filesystem::file_exists(folder)) {
		filesystem::create_file_directory(folder + "/index.html");
	}

	// style sheet
	const std::string stylesheet_string = this->stylesheet();
	filesystem::write_text(folder + "/" + s_stylesheet_name, stylesheet_string);

	for(size_t i=0; i<html_strings.size(); i+=max_matches_per_page_) {
		uint32_t cur_page = i / max_images_per_match_;
		std::stringstream current_page_str;
		current_page_str << header();
		current_page_str << navbar(cur_page, html_strings.size() / max_matches_per_page_ + 1);

		for(size_t j=i; j<MIN(html_strings.size(), i+max_matches_per_page_); j++) {
			current_page_str << html_strings[j];
		}
		current_page_str << footer();
		
		filesystem::write_text(folder + "/" + this->pagename(cur_page), current_page_str.str());
	}
}

std::string MatchesPage::pagename(uint32_t cur_page) const {
	std::stringstream current_page_name;
	current_page_name << "matches_" << std::setw(5) << std::setfill('0') << cur_page << ".html";
	return current_page_name.str();
}

std::string MatchesPage::stylesheet() const {
	std::string stylesheet_str = R"( 

		body {
			margin: 0px;
			color : #fcfcfc;
			background: #111;
			font-size: 14px;
			font-family: sans-serif;
		}

		table {
			margin: 5px;
			border: 2px solid #fcfcfc;
			border-spacing: 0;
   		    border-collapse: collapse;
		}

		a { 
			color: #fff;
			text-decoration: none;
			font-weight: bold;
		}

		img {
			height: 120px;
		}

		)";
	return stylesheet_str;
}

std::string MatchesPage::header() const {
	std::string header_str = R"( 
			<html>
				<head>
					<link rel='stylesheet' type='text/css' href=')" + s_stylesheet_name + R"(' />
				</head>
				<body>
		)";
	return header_str;
}

std::string MatchesPage::footer() const {
	std::string footer_str = R"( 
				</body>
			</html>
		)";
	return footer_str;
}

std::string MatchesPage::navbar(uint32_t cur_page, uint32_t max_pages) const {
	std::stringstream navbar_str;
	navbar_str << "<table><tr>";
	for(uint32_t i=0; i<max_pages; i++) {
		navbar_str << "<td><a href='" << pagename(i) << "'>" << i << "</a></td>";
	}
	navbar_str << "</tr></table>";
	return navbar_str.str();
}