add_library(
  ToyMLIRPass
  STATIC
  ToyCombine.cpp
  ShapeInference.cpp
) 

add_dependencies(ToyMLIRPass ToyCombineIncGen)

target_link_libraries(
  ToyMLIRPass 
  PRIVATE 
  MLIRTransforms
)
