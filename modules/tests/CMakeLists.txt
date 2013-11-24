
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/tests_config.hpp.cmake
			   ${CMAKE_CURRENT_SOURCE_DIR}/tests_config.hpp)

ADD_EXECUTABLE(compute_index_simple compute_index_simple.cxx)
INCLUDE_DIRECTORIES(compute_index_simple ${VOCAB_TREE_INCLUDE} ${OPENCV_INCLUDE_PATH})
TARGET_LINK_LIBRARIES(compute_index_simple search utils)
IF(ENABLE_MPI)
	 TARGET_LINK_LIBRARIES(compute_index_simple ${MPI_LIBRARIES})
ENDIF()

ADD_EXECUTABLE(vocab_tree_simple vocab_tree_simple.cxx)
INCLUDE_DIRECTORIES(vocab_tree_simple ${VOCAB_TREE_INCLUDE} ${OPENCV_INCLUDE_PATH})
TARGET_LINK_LIBRARIES(vocab_tree_simple search utils)

ADD_EXECUTABLE(compute_sift_simple compute_sift_simple.cxx)
INCLUDE_DIRECTORIES(compute_sift_simple ${VOCAB_TREE_INCLUDE} ${OPENCV_INCLUDE_PATH})
TARGET_LINK_LIBRARIES(compute_sift_simple search utils)
IF(ENABLE_MPI)
	 TARGET_LINK_LIBRARIES(compute_sift_simple ${MPI_LIBRARIES})
ENDIF()

ADD_EXECUTABLE(compute_bow_simple compute_bow_simple.cxx)
INCLUDE_DIRECTORIES(compute_bow_simple ${VOCAB_TREE_INCLUDE} ${OPENCV_INCLUDE_PATH})
TARGET_LINK_LIBRARIES(compute_bow_simple search utils)
IF(ENABLE_MPI)
	 TARGET_LINK_LIBRARIES(compute_bow_simple ${MPI_LIBRARIES})
ENDIF()

ADD_EXECUTABLE(compute_search_simple compute_search_simple.cxx)
INCLUDE_DIRECTORIES(compute_search_simple ${VOCAB_TREE_INCLUDE} ${OPENCV_INCLUDE_PATH})
TARGET_LINK_LIBRARIES(compute_search_simple search utils)
IF(ENABLE_MPI)
	 TARGET_LINK_LIBRARIES(compute_search_simple ${MPI_LIBRARIES})
ENDIF()