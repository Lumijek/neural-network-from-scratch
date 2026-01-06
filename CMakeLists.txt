cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

project(main VERSION 1.0)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -I/opt/local/include -Wall  -O3 -mcpu=apple-m1 ")
set(CMAKE_BUILD_TYPE Debug)
add_executable(main main.c matrix.c neural_network.c idx_loader.c)

include_directories("/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Headers/")
include_directories("data/")
target_link_libraries(main "-framework Accelerate")

if(NOT APPLE)
  target_link_libraries(main m)
endif()
