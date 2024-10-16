cmake_minimum_required(VERSION 3.28)

project(mlir-toy)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_compile_options(-fno-rtti)

find_package(MLIR REQUIRED CONFIG)
include_directories(${MLIR_INCLUDE_DIRS})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddMLIR)

add_subdirectory(third-party)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third-party/AST/include)

find_program(LLVM_TABLEGEN_EXE "llvm-tblgen" ${LLVM_TOOLS_BINARY_DIR})
string(TOLOWER "${LLVM_NATIVE_ARCH}" NATIVE_ARCH_LOWER)
llvm_map_components_to_libnames(LLVM_TARGET_SPECIFIC_LIBS 
  "${NATIVE_ARCH_LOWER}codegen" 
  "${NATIVE_ARCH_LOWER}desc"
  "${NATIVE_ARCH_LOWER}info"
  "${NATIVE_ARCH_LOWER}utils"
)
message(STATUS "LLVM_TARGET_SPECIFIC_LIBS: ${LLVM_TARGET_SPECIFIC_LIBS}")

llvm_map_components_to_libnames(llvm_libs support core irreader)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

add_subdirectory(lib)
add_subdirectory(include)
add_subdirectory(tools)
add_subdirectory(test)
