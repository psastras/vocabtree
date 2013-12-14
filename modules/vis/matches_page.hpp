#pragma once

#include <utils/dataset.hpp>
#include <string>

/// MatchesPage class keeps track of query images and their matches, and outputs an html 
/// page containing these matches for visualization purposes.
class MatchesPage {
public: 
	/// Ctor, max_matches_per_page specified the max number of rows per page, if this is set too
	/// high, the web browser might have problems loading the page.  max_images_per_match specifies
	/// the max number of images in each row.  If this is set too high, the web browser might have 
	/// problems loading the page.
	MatchesPage(uint32_t max_matches_per_page = 16, uint32_t max_images_per_match = 16);
	~MatchesPage();

	/// Adds a match to the html_strings variable which will be written as html on write().  
	/// The query_id is the id of the image query.  match_ids are the ids of the matches. 
	/// Finally, dataset is used to figure out the image paths.  If validated vector is provided,
	/// the web page will highlight validated matches, values should be zero if unvalidated,
	/// > 0 if validated, and < 0 if failed validation.  The validation vector can be smaller
	/// than the size of match_ids, in which case it is assumed to correspond to beginning of
	/// match_ids.
	void add_match(uint32_t query_id, std::vector<uint64_t> &match_ids, const Dataset &dataset,
		PTR_LIB::shared_ptr< std::vector<int> > validated = nullptr);

	/// Writes out all the html match mages to the input specified folder.  The first page 
	/// will look something like folder/matches_00000.html.
	void write(const std::string &folder) const;

protected:

	std::string stylesheet() const;  /// Returns a string containing css stylesheet
	std::string header() const; /// Returns a string containing the html header
	std::string footer() const; /// Returns a string containing the html footer
	std::string navbar(uint32_t cur_page, uint32_t max_pages) const; /// Returns a string containing the navbar needed for pagination
	std::string pagename(uint32_t cur_page) const; /// Returns a string containing the pagename (ex. matches_00001.html)

	std::vector<std::string> html_strings; /// Holds the html strings for each match passed into add_match

	uint32_t max_matches_per_page_, max_images_per_match_;

};