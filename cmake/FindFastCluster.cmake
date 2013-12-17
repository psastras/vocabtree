find_path( FASTCLUSTER_INCLUDE_PATH fastcluster/kmeans.h
	/code/local/include
	/usr/include
	/usr/local/include
	/sw/include
	/afs/cs.cmu.edu/user/psastras/ladoga/install_warp/include
	/opt/local/include
	3rdparty/install/${CMAKE_SYSTEM_NAME}/include
	DOC "The directory where fastcluster/kmeans.h resides")
set(FAST_CLUSTER_SEARCH_PATHS /code/local/lib
	/usr/lib64 
	/usr/lib 
	/usr/local/lib64 
	/usr/local/lib 
	/sw/lib 
	/afs/cs.cmu.edu/user/psastras/ladoga/install_warp/lib 
	3rdparty/install/${CMAKE_SYSTEM_NAME}/lib
	/opt/local/lib)
find_library(FASTANN NAMES fastann  PATHS	${FAST_CLUSTER_SEARCH_PATHS} DOC "The FastANN library")
find_library(FASTCLUSTER NAMES fastcluster  PATHS	${FAST_CLUSTER_SEARCH_PATHS} DOC "The FastCluster library")
set(FASTCLUSTER_LIBRARIES ${FASTANN} ${FASTCLUSTER})
if(FASTCLUSTER_INCLUDE_PATH)
	set(FASTCLUSTER_FOUND 1)
else(FASTCLUSTER_INCLUDE_PATH)
	set(FASTCLUSTER_FOUND 0)
endif(FASTCLUSTER_INCLUDE_PATH)
MARK_AS_ADVANCED(FASTCLUSTER_FOUND)
