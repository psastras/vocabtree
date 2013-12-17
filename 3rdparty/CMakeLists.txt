IF(MPI_FOUND AND YASM_EXECUTABLE)
	INCLUDE(ExternalProject)
	ExternalProject_Add(FastANN
		SOURCE_DIR     ${VOCAB_TREE_3RDPARTY_DIR}/philbinj-fastann/
		PREFIX		   ${VOCAB_TREE_3RDPARTY_DIR}/
		INSTALL_DIR    ${VOCAB_TREE_3RDPARTY_DIR}/install/${CMAKE_SYSTEM_NAME}
		CMAKE_ARGS     -DCMAKE_INSTALL_PREFIX:PATH=${VOCAB_TREE_3RDPARTY_DIR}/install/${CMAKE_SYSTEM_NAME}
		BUILD_IN_SOURCE 1
		PATCH_COMMAND   ${CMAKE_COMMAND} -E copy_directory ${VOCAB_TREE_3RDPARTY_DIR}/patches/philbinj-fastann/ ${VOCAB_TREE_3RDPARTY_DIR}/philbinj-fastann/
		URL http://www.robots.ox.ac.uk/~vgg/software/fastanncluster/fastann/philbinj-fastann-cbf02be.tar.gz
	)

	ExternalProject_Add(FastCluster
		DEPENDS FastANN
		SOURCE_DIR     ${VOCAB_TREE_3RDPARTY_DIR}/philbinj-fastcluster/
		PREFIX		   ${VOCAB_TREE_3RDPARTY_DIR}/
		INSTALL_DIR    ${VOCAB_TREE_3RDPARTY_DIR}/install/${CMAKE_SYSTEM_NAME}
		CMAKE_ARGS     -DCMAKE_INSTALL_PREFIX:PATH=${VOCAB_TREE_3RDPARTY_DIR}/install/${CMAKE_SYSTEM_NAME}
		BUILD_IN_SOURCE 1
		PATCH_COMMAND   ${CMAKE_COMMAND} -E copy_directory ${VOCAB_TREE_3RDPARTY_DIR}/patches/philbinj-fastcluster/ ${VOCAB_TREE_3RDPARTY_DIR}/philbinj-fastcluster/
		URL http://www.robots.ox.ac.uk/~vgg/software/fastanncluster/fastcluster/philbinj-fastcluster-bf3d361.tar.gz
	)
ENDIF()