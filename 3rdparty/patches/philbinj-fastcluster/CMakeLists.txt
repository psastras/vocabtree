cmake_minimum_required(VERSION 2.6)
project(fastcluster)

SET(CMAKE_BUILD_TYPE Release)

IF(WIN32)
    MESSAGE("Windows not currently supported" FATAL_ERROR)
ENDIF(WIN32)

IF(UNIX)
    SET(CMAKE_CXX_COMPILER "mpic++")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2 -g3 -msse2 -march=native")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O2 -g3 -msse2 -march=native")
    MESSAGE("${CMAKE_CXX_FLAGS}")
ENDIF(UNIX)

INCLUDE_DIRECTORIES(fastcluster ../install/${CMAKE_SYSTEM_NAME}/include)
LINK_DIRECTORIES(fastcluster ../install/${CMAKE_SYSTEM_NAME}/lib)
ADD_LIBRARY(fastcluster SHARED mpi_queue.cpp kmeans.cpp randomkit.c whetstone.cpp mpi_utils.cpp)
TARGET_LINK_LIBRARIES(fastcluster fastann)

INCLUDE_DIRECTORIES(test_mpi_queue ../install/${CMAKE_SYSTEM_NAME}/include)
LINK_DIRECTORIES(test_mpi_queue ../install/${CMAKE_SYSTEM_NAME}/lib)
ADD_EXECUTABLE(test_mpi_queue test_mpi_queue.cpp)
TARGET_LINK_LIBRARIES(test_mpi_queue fastcluster)

ADD_CUSTOM_COMMAND(
    OUTPUT dummy_python_cmd
    COMMAND python setup.py install
    DEPENDS fastcluster)
ADD_CUSTOM_TARGET(python_cmd DEPENDS dummy_python_cmd)

INSTALL(TARGETS fastcluster
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
       )
INSTALL(FILES kmeans.h kmeans.hpp mpi_queue.hpp randomkit.h DESTINATION include/fastcluster)
INSTALL(CODE "execute_process(COMMAND python setup.py install)")
