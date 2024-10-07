cmake_minimum_required(VERSION 3.28)

project(mlir-toy)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(MLIR REQUIRED CONFIG)
include_directories(${MLIR_INCLUDE_DIRS})

add_subdirectory(third-party)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third-party/AST/include)


llvm_map_components_to_libnames(llvm_libs support core irreader)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

add_subdirectory(lib)
add_subdirectory(include)
