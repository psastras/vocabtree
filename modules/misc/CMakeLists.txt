ADD_EXECUTABLE(prep_video_dataset prep_video_dataset.cxx)
INCLUDE_DIRECTORIES(prep_video_dataset ${VOCAB_TREE_INCLUDE} ${OPENCV_INCLUDE_PATH})
TARGET_LINK_LIBRARIES(prep_video_dataset search utils)
