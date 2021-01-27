
cmake_minimum_required(VERSION 3.15)
project(dnn)

set(CMAKE_CXX_STANDARD 14)
add_definitions(-std=c++11)

# Add OpenCV headers location to your include paths
find_package(OpenCV  REQUIRED)
message(${OpenCV_INCLUDE_DIRS})
message(${OpenCV_LIBS})

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(dnn main.cpp)

target_link_libraries(dnn ${OpenCV_LIBS})