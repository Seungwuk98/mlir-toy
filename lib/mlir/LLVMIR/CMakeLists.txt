
add_library(
  ToyLLVMDumper
  
  STATIC 
  LLVMDumper.cpp
)

target_link_libraries(
  ToyLLVMDumper

  PRIVATE 
  MLIRTargetLLVMIRExport
  MLIRExecutionEngine
)
