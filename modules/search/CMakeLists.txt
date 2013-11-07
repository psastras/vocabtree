SET(search_base_SRCS search_base/search_base.cxx)

SET(inverted_index_SRCS inverted_index/inverted_index.cxx)

SET(bag_of_words_SRCS bag_of_words/bag_of_words.cxx)

SET(vocab_tree_SRCS vocab_tree/vocab_tree.cxx)


ADD_LIBRARY(search ${search_base_SRCS} ${inverted_index_SRCS} ${vocab_tree_SRCS} ${bag_of_words_SRCS})
INCLUDE_DIRECTORIES(search ${VOCAB_TREE_INCLUDE} ${OPENCV_INCLUDE_PATH} ${BOOST_INCLUDE_PATH})
TARGET_LINK_LIBRARIES(search ${OPENCV_LIBRARIES} ${BOOST_LIBRARIES})

# Need -fPIC only if posix position independent libraries
IF(CMAKE_COMPILER_IS_GNUCC)
	SET_TARGET_PROPERTIES(search PROPERTIES COMPILE_FLAGS -fPIC)
ENDIF(CMAKE_COMPILER_IS_GNUCC)

