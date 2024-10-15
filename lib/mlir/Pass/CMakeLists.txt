add_library(
  ToyMLIRPass
  STATIC
  ToyCombine.cpp
  ShapeInference.cpp
  ToyToAffineLoweringPass.cpp
  ToLLVMLoweringPass.cpp
) 

add_dependencies(ToyMLIRPass ToyCombineIncGen)

target_link_libraries(
  ToyMLIRPass 
  PRIVATE 
  MLIRTransforms
  MLIRArithDialect
  MLIRAffineDialect
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRArithToLLVM
  MLIRSCFToControlFlow
  MLIRControlFlowToLLVM
  MLIRAffineToStandard
  MLIRFuncToLLVM
  MLIRMemRefToLLVM
)
