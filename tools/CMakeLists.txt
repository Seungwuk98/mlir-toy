set(TOY_LINK_LIBS
  ToyAST 
  ToyContext 
  ToyDialect
  ToyLLVMDumper
  ToyMLIRGen
  ToyMLIRPass
  ToyParser
)

add_executable(
  toyc
  toyc.cpp
)

target_link_libraries(
  toyc 
  PRIVATE

  ${TOY_LINK_LIBS}
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_executable(
  toy-opt
  toy-opt.cpp
)

target_link_libraries(
  toy-opt 
  PRIVATE

  ${TOY_LINK_LIBS}
  MLIROptLib
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
)

set(TOY_TOOLS_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR} CACHE INTERNAL "")

install(TARGETS toyc DESTINATION bin)
install(TARGETS toy-opt DESTINATION bin)
