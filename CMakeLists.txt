cmake_minimum_required(VERSION 3.12)

project(ssd_mobilenet_detection C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

if(NOT CMAKE_BUILD_TYPE)
	message(STATUS "Setting build type to Debug, for debug builds use"
		"'-DCMAKE_BUILD_TYPE=Release'.")
	set(CMAKE_BUILD_TYPE "Debug")
endif()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(ssd_mobilenet_detection "")

target_include_directories(ssd_mobilenet_detection
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/.
  ${CMAKE_CURRENT_LIST_DIR}/../tensorflow
)

target_sources(ssd_mobilenet_detection
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/main.cpp
  ${CMAKE_CURRENT_LIST_DIR}/capture.cpp
  ${CMAKE_CURRENT_LIST_DIR}/main_functions.h
  ${CMAKE_CURRENT_LIST_DIR}/main_functions.cpp
  ${CMAKE_CURRENT_LIST_DIR}/model_settings.h
  ${CMAKE_CURRENT_LIST_DIR}/model_settings.cpp
  ${CMAKE_CURRENT_LIST_DIR}/mstick.cpp
  ${CMAKE_CURRENT_LIST_DIR}/nms.h
  ${CMAKE_CURRENT_LIST_DIR}/nms.cpp
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
  ${OpenCV_LIBS}
)
