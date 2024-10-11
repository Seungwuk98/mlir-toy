add_library(
  ToyMLIRGen
  STATIC 
  IRGenerator.cpp
)


target_link_libraries(
  ToyMLIRGen 
  PRIVATE

  ToyDialect
  ToyAST
)
