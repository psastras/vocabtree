find_path( OPENCV_INCLUDE_PATH opencv2/opencv.hpp
	/usr/include
	/code/local/include
	/usr/local/include
	/sw/include
	/opt/local/lib/cmake
	/opt/local/include
	/afs/cs.cmu.edu/user/psastras/ladoga/install_warp/include
	/home/psastras/ladoga/install_warp/include
	3rdparty/install/${CMAKE_SYSTEM_NAME}/include
	DOC "The directory where opencv2/opencv.hpp resides")
find_library(OPENCV_LIB_PATH opencv_core
	/usr/lib64
	/usr/lib
	/usr/local/lib64
	/usr/local/lib
	/code/local/lib
	/usr/lib64
	/sw/lib
	/opt/local/lib/cmake
	/opt/local/lib
	/afs/cs.cmu.edu/user/psastras/ladoga/install_warp/lib
	/home/psastras/ladoga/install_warp/lib
	3rdparty/install/${CMAKE_SYSTEM_NAME}/lib
	)
get_filename_component(OPENCV_LIB_PATH "${OPENCV_LIB_PATH}"
					PATH)
find_library(OPENCV_CORE NAMES opencv_core PATHS	${OPENCV_LIB_PATH} DOC "The OpenCV core library")
find_library(OPENCV_HIGHGUI NAMES opencv_highgui PATHS	${OPENCV_LIB_PATH} DOC "The OpenCV highgui library")
find_library(OPENCV_IMGPROC NAMES opencv_imgproc PATHS	${OPENCV_LIB_PATH} DOC "The OpenCV imgproc library")
find_library(OPENCV_VIDEO NAMES opencv_video PATHS	${OPENCV_LIB_PATH} DOC "The OpenCV video library")
find_library(OPENCV_NONFREE NAMES opencv_nonfree PATHS	${OPENCV_LIB_PATH} DOC "The OpenCV nonfree library")
find_library(OPENCV_FEATURES2D NAMES opencv_features2d PATHS	${OPENCV_LIB_PATH} DOC "The OpenCV features 2d library")
find_library(OPENCV_FLANN NAMES opencv_flann PATHS	${OPENCV_LIB_PATH} DOC "The OpenCV flann library")
find_library(OPENCV_CALIB3D NAMES opencv_calib3d PATHS	${OPENCV_LIB_PATH} DOC "The OpenCV calib 3d library")
find_library(OPENCV_CONTRIB NAMES opencv_contrib PATHS	${OPENCV_LIB_PATH} DOC "The OpenCV contrib library")
find_library(OPENCV_STITCHING NAMES opencv_stitching PATHS	${OPENCV_LIB_PATH} DOC "The OpenCV stitching library")
set(OPENCV_LIBRARIES ${OPENCV_CORE} ${OPENCV_STITCHING} ${OPENCV_HIGHGUI} ${OPENCV_VIDEO} ${OPENCV_IMGPROC} ${OPENCV_NONFREE} ${OPENCV_CONTRIB} ${OPENCV_FEATURES2D} ${OPENCV_FLANN} ${OPENCV_CALIB3D})
#just going to assume noone has an insane opencv install thus check for the include path is sufficient
if(OPENCV_INCLUDE_PATH)
	set(OPENCV_FOUND 1)
else(OPENCV_INCLUDE_PATH)
	set(OPENCV_FOUND 0)
	message(FATAL_ERROR "OpenCV not found.  Either set OPENCV_INCLUDE_PATH and OPENCV_LIBRARIES or make sure OpenCV is included in your PATH")
endif(OPENCV_INCLUDE_PATH)
MARK_AS_ADVANCED(OPENCV_FOUND)
