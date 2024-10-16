add_executable(
  toyc
  toyc.cpp
)

target_link_libraries(
  toyc 
  PRIVATE
 
  ToyAST 
  ToyContext 
  ToyDialect
  ToyLLVMDumper
  ToyMLIRGen
  ToyMLIRPass
)
