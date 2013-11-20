CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/bench_config.hpp.cmake
			   ${CMAKE_CURRENT_SOURCE_DIR}/bench_config.hpp)

ADD_EXECUTABLE(bench_geo_verify bench_geo_verify.cxx)
INCLUDE_DIRECTORIES(bench_geo_verify ${VOCAB_TREE_INCLUDE} ${OPENCV_INCLUDE_PATH})
TARGET_LINK_LIBRARIES(bench_geo_verify search utils)

ADD_EXECUTABLE(bench_compute_features bench_compute_features.cxx)
INCLUDE_DIRECTORIES(bench_compute_features ${VOCAB_TREE_INCLUDE} ${OPENCV_INCLUDE_PATH})
TARGET_LINK_LIBRARIES(bench_compute_features search utils)

ADD_EXECUTABLE(bench_index bench_index.cxx)
INCLUDE_DIRECTORIES(bench_index ${VOCAB_TREE_INCLUDE} ${OPENCV_INCLUDE_PATH})
TARGET_LINK_LIBRARIES(bench_index search utils)

ADD_EXECUTABLE(bench_bow bench_bow.cxx)
INCLUDE_DIRECTORIES(bench_bow ${VOCAB_TREE_INCLUDE} ${OPENCV_INCLUDE_PATH})
TARGET_LINK_LIBRARIES(bench_bow search utils)