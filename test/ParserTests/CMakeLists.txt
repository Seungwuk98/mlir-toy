file(GLOB_RECURSE PARSER_TESTS_SOURCES "*.cpp")

add_executable(ParserTests
  ${PARSER_TESTS_SOURCES}
)

target_link_libraries(ParserTests
  PRIVATE
  ToyParser 
  ToyAST
  ToyContext
  ToyMLIRGen
  MLIRIR
  doctest
  ${llvm_libs}
)

