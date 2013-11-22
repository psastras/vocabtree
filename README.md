vocabtree
=========

Multinode, multicore large scale image search.


Build Instructions
===================

Supported OS

* Linux
* Windows

Required Dependencies

* OpenCV
* Boost

Optional Dependencies

* OpenMP - required for multithreading support
* MPI - required for multinode support (ideally should be used with OpenMP)

Building
----------

Create a build directory and then run cmake from that directory pointing to the root source directory.

Ex: From the root source directory:

    mkdir build
    cd build
    cmake ..
    make

Binaries are located in your build directory under bin.

Unix Specific
----------
If on Unix, and yasm and MPI are available, there is an automatically enabled option to use the FASTANN and 
FASTCLUSTER libraries (http://www.robots.ox.ac.uk/~vgg/software/fastanncluster/).  These will be automatically 
downloaded and compiled and enables out of core multinode kmeans clustering.  Otherwise, the kmeans 
implementation bundled with OpenCV will be used, which may or may not be multithreaded (single node) 
depending on your OpenCV build settings.


Documentation
===================

Documentation can be built by running (requires doxygen)

    make doc

Documentation can also be found here:

http://psastras.github.io/vocabtree/
