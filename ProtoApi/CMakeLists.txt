cmake_minimum_required(VERSION 3.16.3)

project(SampleProtoApi)

find_package(Protobuf CONFIG REQUIRED)
set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)

find_package(gRPC CONFIG REQUIRED)
find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin
             PATHS "${VCPKG_DIR}/installed/${TRIPLET_DEFAULT}/tools/grpc")

set(API_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../ProtoApi")
set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

file(MAKE_DIRECTORY ${API_DIR}/gen)

set(PROTO_FILES ${CMAKE_CURRENT_SOURCE_DIR}/protobuf/Sample.proto)

set(GRPC_FILES
${CMAKE_CURRENT_SOURCE_DIR}/SampleService.proto)

# generate and include protobuf and grpc
protobuf_generate(
  OUT_VAR PROTO_CPP
  PROTOS ${PROTO_FILES}
  LANGUAGE cpp
  IMPORT_DIRS ${API_DIR}
  PROTOC_OUT_DIR ${API_DIR}/gen/ProtoApi)

protobuf_generate(
  OUT_VAR GRPC_CPP
  PROTOS ${GRPC_FILES}
  LANGUAGE cpp
  IMPORT_DIRS ${API_DIR}
  PROTOC_OUT_DIR ${API_DIR}/gen/ProtoApi)

protobuf_generate(
  OUT_VAR GRPC_GEN_CPP
  PROTOS ${GRPC_FILES} 
  LANGUAGE grpc
  GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc
  PLUGIN "protoc-gen-grpc=${_GRPC_CPP_PLUGIN_EXECUTABLE}"
  IMPORT_DIRS ${API_DIR}
  PROTOC_OUT_DIR ${API_DIR}/gen/ProtoApi)

add_library(${PROJECT_NAME} STATIC ${PROTO_CPP} ${GRPC_CPP} ${GRPC_GEN_CPP})

target_include_directories(${PROJECT_NAME}
			PUBLIC 
        ${API_DIR}/gen/ProtoApi)


target_link_libraries(${PROJECT_NAME}
			PRIVATE 
        gRPC::grpc++
				gRPC::grpc
				protobuf::libprotobuf
				)
