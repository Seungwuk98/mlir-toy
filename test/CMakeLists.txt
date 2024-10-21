include_directories(TestUtils)

add_subdirectory(ParserTests)
add_subdirectory(MLIRGenTests)

if (${ENABLE_LIT_TESTS}) 
  add_subdirectory(LitTests)
endif()
