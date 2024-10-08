add_library(
  ToyContext
  STATIC
  ToyContext.cpp
)

target_link_libraries(
  ToyContext
  PRIVATE 
  ToyAST
)


