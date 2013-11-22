message(STATUS "downloading...
     src='http://www.robots.ox.ac.uk/~vgg/software/fastanncluster/fastann/philbinj-fastann-cbf02be.tar.gz'
     dst='/home/psastras/vocabtree/3rdparty///src/philbinj-fastann-cbf02be.tar.gz'
     timeout='none'")




file(DOWNLOAD
  "http://www.robots.ox.ac.uk/~vgg/software/fastanncluster/fastann/philbinj-fastann-cbf02be.tar.gz"
  "/home/psastras/vocabtree/3rdparty///src/philbinj-fastann-cbf02be.tar.gz"
  SHOW_PROGRESS
  # no EXPECTED_HASH
  # no TIMEOUT
  STATUS status
  LOG log)

list(GET status 0 status_code)
list(GET status 1 status_string)

if(NOT status_code EQUAL 0)
  message(FATAL_ERROR "error: downloading 'http://www.robots.ox.ac.uk/~vgg/software/fastanncluster/fastann/philbinj-fastann-cbf02be.tar.gz' failed
  status_code: ${status_code}
  status_string: ${status_string}
  log: ${log}
")
endif()

message(STATUS "downloading... done")
