
find_path( BOOST_INCLUDE_PATH boost/config.hpp
	/code/local/include
	/usr/include
	/usr/local/include
	/sw/include
	/afs/cs.cmu.edu/user/psastras/ladoga/install_warp/include
	/opt/local/include
	/Users/kayvonf/src/extern/boost_1_53_0/include
	DOC "The directory where boost/config.hpp resides")
set(BOOST_SEARCH_PATHS /code/local/lib
	/usr/lib64
	/usr/lib
	/usr/local/lib64
	/usr/local/lib
	/sw/lib
	/afs/cs.cmu.edu/user/psastras/ladoga/install_warp/lib
	/Users/kayvonf/src/extern/boost_1_53_0/lib
	/opt/local/lib)
find_library(BOOST_SERIALIZATION NAMES boost_serialization-mt boost_serialization PATHS	${BOOST_SEARCH_PATHS} DOC "The boost serialization library")
find_library(BOOST_FILESYSTEM NAMES boost_filesystem-mt boost_filesystem PATHS ${BOOST_SEARCH_PATHS} DOC "The boost filesystem library")
find_library(BOOST_SYSTEM NAMES boost_system-mt boost_system PATHS ${BOOST_SEARCH_PATHS} DOC "The boost system library")
find_library(BOOST_TIMERS NAMES boost_timer  PATHS ${BOOST_SEARCH_PATHS} DOC "The boost timer library")
set(BOOST_LIBRARIES ${BOOST_SERIALIZATION} ${BOOST_FILESYSTEM} ${BOOST_SYSTEM} ${BOOST_TIMERS})
if(BOOST_INCLUDE_PATH)
	set(BOOST_FOUND 1)
else(BOOST_INCLUDE_PATH)
	set(BOOST_FOUND 0)
	message(FATAL_ERROR "Boost not found.  Either set BOOST_INCLUDE_PATH and BOOST_LIBRARIES or make sure Boost is included in your PATH")
endif(BOOST_INCLUDE_PATH)

MARK_AS_ADVANCED(BOOST_FOUND)
