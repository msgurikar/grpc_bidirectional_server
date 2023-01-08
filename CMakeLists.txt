cmake_minimum_required(VERSION 3.16.3)

project(SampleRepo)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall --std=c++17 -O3 -fPIC" )
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3 -fPIC" )

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/API/gen)

add_subdirectory(ProtoApi)
add_subdirectory(SampleService)
