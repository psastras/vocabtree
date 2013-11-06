
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/tests_config.hpp.cmake
			   ${CMAKE_CURRENT_SOURCE_DIR}/tests_config.hpp)

ADD_EXECUTABLE(inverted_index_simple inverted_index_simple.cxx)
INCLUDE_DIRECTORIES(inverted_index_simple ${VOCAB_TREE_INCLUDE} ${OPENCV_INCLUDE_PATH})
TARGET_LINK_LIBRARIES(inverted_index_simple search utils)


ADD_EXECUTABLE(vocab_tree_simple vocab_tree_simple.cxx)
INCLUDE_DIRECTORIES(vocab_tree_simple ${VOCAB_TREE_INCLUDE} ${OPENCV_INCLUDE_PATH})
TARGET_LINK_LIBRARIES(vocab_tree_simple search utils)