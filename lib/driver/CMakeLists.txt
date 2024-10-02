add_executable(toyc toyc.cpp)

target_link_libraries(
  toyc 
  PRIVATE
  AST
  ${llvm_libs}
  MLIRIR
)

