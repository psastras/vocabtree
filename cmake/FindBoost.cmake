
find_path( BOOST_INCLUDE_PATH boost/config.hpp
	/code/local/include
	/usr/include
	/usr/local/include
	/sw/include
	/afs/cs.cmu.edu/user/psastras/ladoga/install_warp/include
	/opt/local/include
	3rdparty/install/${CMAKE_SYSTEM_NAME}/include
	/usr/local/Cellar/boost/1.53.0/include
	DOC "The directory where boost/config.hpp resides")
set(BOOST_SEARCH_PATHS /code/local/lib
	/usr/lib64 
	/usr/lib 
	/usr/local/lib64 
	/usr/local/lib 
	/sw/lib 
	/afs/cs.cmu.edu/user/psastras/ladoga/install_warp/lib 
	3rdparty/install/${CMAKE_SYSTEM_NAME}/lib
	/usr/local/Cellar/boost/1.53.0/lib
	/opt/local/lib)
find_library(BOOST_FILESYSTEM NAMES boost_filesystem-mt boost_filesystem libboost_filesystem-vc120-mt-1_55 PATHS ${BOOST_SEARCH_PATHS} DOC "The boost 
filesystem library")
find_library(BOOST_FILESYSTEM_DEBUG NAMES libboost_filesystem-vc120-mt-gd-1_55  boost_filesystem boost_filesystem-mt PATHS ${BOOST_SEARCH_PATHS} DOC "The boost filesystem library")
find_library(BOOST_SYSTEM NAMES boost_system-mt boost_system libboost_system-vc120-mt-1_55 PATHS ${BOOST_SEARCH_PATHS} DOC "The boost system library")
find_library(BOOST_SYSTEM_DEBUG NAMES libboost_system-vc120-mt-gd-1_55 boost_system boost_system-mt libboost_system PATHS ${BOOST_SEARCH_PATHS} DOC "The boost system library")
find_library(BOOST_TIMERS NAMES boost_timer boost_timer-mt libboost_timer libboost_timer-vc120-mt-1_55 PATHS ${BOOST_SEARCH_PATHS} DOC "The boost timer library")
find_library(BOOST_TIMERS_DEBUG NAMES libboost_timer-vc120-mt-gd-1_55 boost_timer boost_timer-mt libboost_timer PATHS ${BOOST_SEARCH_PATHS} DOC "The boost timer library")
#set(BOOST_LIBRARIES optimized ${BOOST_FILESYSTEM} optimized ${BOOST_SYSTEM} optimized ${BOOST_TIMERS})
set(BOOST_LIBRARIES optimized ${BOOST_FILESYSTEM} optimized ${BOOST_SYSTEM} optimized ${BOOST_TIMERS}
	debug ${BOOST_FILESYSTEM_DEBUG} debug ${BOOST_SYSTEM_DEBUG} debug ${BOOST_TIMERS_DEBUG})
if(BOOST_INCLUDE_PATH)
	set(BOOST_FOUND 1)
else(BOOST_INCLUDE_PATH)
	set(BOOST_FOUND 0)
	message(FATAL_ERROR "Boost not found.  Either set BOOST_INCLUDE_PATH and BOOST_LIBRARIES or make sure Boost is included in your PATH")
endif(BOOST_INCLUDE_PATH)
MARK_AS_ADVANCED(BOOST_FOUND)
