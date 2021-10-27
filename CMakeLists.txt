cmake_minimum_required(VERSION 3.12)

project(ssd_mobilenet_detection C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)

if(NOT CMAKE_BUILD_TYPE)
	message(STATUS "Setting build type to Debug, for debug builds use"
		"'-DCMAKE_BUILD_TYPE=Debug'.")
	set(CMAKE_BUILD_TYPE "Release")
endif()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(ssd_mobilenet_detection "")

target_include_directories(ssd_mobilenet_detection
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/.
  ${CMAKE_CURRENT_LIST_DIR}/../tensorflow
  ${CMAKE_CURRENT_LIST_DIR}/../tensorflow/tensorflow/lite/build/flatbuffers/include/
  ${CMAKE_CURRENT_LIST_DIR}/tinyhttpd
)

target_sources(ssd_mobilenet_detection
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/main.cpp
  ${CMAKE_CURRENT_LIST_DIR}/util.cpp
  ${CMAKE_CURRENT_LIST_DIR}/nms.h
  ${CMAKE_CURRENT_LIST_DIR}/nms.cpp
  ${CMAKE_CURRENT_LIST_DIR}/tinyhttpd/threadqueue.c
  ${CMAKE_CURRENT_LIST_DIR}/tinyhttpd/sockstream.cpp
  ${CMAKE_CURRENT_LIST_DIR}/tinyhttpd/mycgi.cpp
  ${CMAKE_CURRENT_LIST_DIR}/tinyhttpd/tinyhttpd.cpp
  ${CMAKE_CURRENT_LIST_DIR}/tinyhttpd/httpserver.cpp
  ${CMAKE_CURRENT_LIST_DIR}/tinyhttpd/http_util.cpp
)

target_link_directories(
  ssd_mobilenet_detection
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/libs
)

target_link_libraries(
  ssd_mobilenet_detection
  tensorflow-lite
  dl
  pthread
  flatbuffers
  XNNPACK
  pthreadpool
  cpuinfo
  clog
  fft2d_fftsg
  fft2d_fftsg2d
  ruy
  farmhash
  cgicc
  ${OpenCV_LIBS}
)
