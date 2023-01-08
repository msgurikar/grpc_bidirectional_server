cmake_minimum_required(VERSION 3.16.3)

project(
  SampleService
  VERSION 1.0.0
  LANGUAGES CXX)

find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)

set(PROJECT_INCLUDES				
		inc/SampleServiceImpl.h
		)

set(PROJECT_SOURCES		
		src/main.cpp
		src/SampleServiceImpl.cpp)

set(PROJECT_FILES ${PROJECT_INCLUDES} ${PROJECT_SOURCES})

add_executable(${PROJECT_NAME} ${PROJECT_FILES})

target_include_directories(${PROJECT_NAME} 
  PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(${PROJECT_NAME}
		 PRIVATE
			SampleProtoApi 
			gRPC::grpc++_reflection
			gRPC::grpc++
			gRPC::grpc
			protobuf::libprotobuf)


