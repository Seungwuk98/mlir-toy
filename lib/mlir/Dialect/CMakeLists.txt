add_library(
  ToyDialect 
  STATIC
  ToyDialect.cpp
  ToyOp.cpp
)

add_dependencies(ToyDialect ToyDialectIncGen ToyOpIncGen)

target_link_libraries(ToyDialect
  PRIVATE 
  MLIRIR
)
