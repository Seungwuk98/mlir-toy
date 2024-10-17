add_library(
  ToyDialect 
  STATIC
  ToyDialect.cpp
  ToyOp.cpp
  ToyInterface.cpp
  ToyType.cpp
)

add_dependencies(ToyDialect ToyDialectIncGen ToyOpIncGen ToyInterfaceIncGen ToyTypeIncGen)

target_link_libraries(ToyDialect
  PRIVATE 
  MLIRIR
  MLIRFunctionInterfaces
  MLIRCastInterfaces
  ToyMLIRPass
)
