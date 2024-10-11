file(GLOB_RECURSE MLIRGEN_TEST_SOURCES "*.cpp")

add_executable(MLIRGenTests
  ${MLIRGEN_TEST_SOURCES}
)

target_link_libraries(MLIRGenTests
  PRIVATE
  ToyParser 
  ToyAST
  ToyContext
  ToyMLIRGen
  MLIRIR
  doctest
  ${llvm_libs}
)

