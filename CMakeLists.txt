cmake_minimum_required(VERSION 3.12)

project(ssd_mobilenet_detection C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to Release. Use -DCMAKE_BUILD_TYPE=Debug for debug builds.")
    set(CMAKE_BUILD_TYPE "Release")
endif()

option(BUILD_SERVER "Build the HTTP detection server" ON)

add_library(ssd_core STATIC
    nms.cpp
    util.cpp
    request_params.cpp
)
target_include_directories(ssd_core
    PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
)

add_executable(gentest gentest.cpp)

if(BUILD_SERVER)
    set(TINYHTTPD_DIR ${CMAKE_CURRENT_LIST_DIR}/tinyhttpd)
    if(NOT EXISTS ${TINYHTTPD_DIR}/threadqueue.c)
        message(FATAL_ERROR
            "Missing tinyhttpd sources at ${TINYHTTPD_DIR}. "
            "Either restore that directory or configure with -DBUILD_SERVER=OFF "
            "to build/tests without the HTTP service binary.")
    endif()

    find_package(OpenCV REQUIRED)
    find_package(Boost REQUIRED)

    add_executable(ssd_mobilenet_detection)
    target_include_directories(ssd_mobilenet_detection
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/../tensorflow
        ${CMAKE_CURRENT_LIST_DIR}/../tensorflow/tensorflow/lite/build/flatbuffers/include/
        ${TINYHTTPD_DIR}
        ${OpenCV_INCLUDE_DIRS}
    )
    target_sources(ssd_mobilenet_detection
        PRIVATE
        main.cpp
        ${TINYHTTPD_DIR}/threadqueue.c
        ${TINYHTTPD_DIR}/sockstream.cpp
        ${TINYHTTPD_DIR}/mycgi.cpp
        ${TINYHTTPD_DIR}/tinyhttpd.cpp
        ${TINYHTTPD_DIR}/httpserver.cpp
        ${TINYHTTPD_DIR}/http_util.cpp
    )
    target_link_directories(ssd_mobilenet_detection
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/libs
    )
    target_link_libraries(ssd_mobilenet_detection
        PRIVATE
        ssd_core
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
        ${Boost_PROGRAM_OPTIONS_LIBRARY}
    )
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(crop_params_test
        tests/crop_params_test.cpp
        request_params.cpp
    )
    target_include_directories(crop_params_test
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
    )
    add_test(NAME crop_params_test COMMAND crop_params_test)

    add_executable(nms_golden_test
        tests/nms_golden_test.cpp
        nms.cpp
    )
    target_include_directories(nms_golden_test
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
    )
    add_test(
        NAME nms_golden_test
        COMMAND nms_golden_test ${CMAKE_CURRENT_LIST_DIR}/tests/golden
    )
endif()
