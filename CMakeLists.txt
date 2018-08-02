CMAKE_MINIMUM_REQUIRED(VERSION 2.8.6)

set (CMAKE_CXX_STANDARD 11)
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -msse4.1" )

find_package( OpenCV REQUIRED )
ADD_EXECUTABLE(3drs_real_time 3drs_real_time.cpp)
TARGET_LINK_LIBRARIES( 3drs_real_time pthread)
TARGET_LINK_LIBRARIES(3drs_real_time ${OpenCV_LIBS} )

