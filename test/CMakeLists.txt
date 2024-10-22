message(STATUS "Configuring toy lit test suite")

configure_lit_site_cfg(
${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
MAIN_CONFIG
${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(TOY_TEST_DEPENDS
  toyc
  toy-opt
)

find_program(FILE_CHECK_TOOL FileCheck
  REQUIRED
  HINTS ${LLVM_TOOLS_BINARY_DIR}
  NO_DEFAULT_PATH)

message(STATUS "FileCheck found: ${FILE_CHECK_TOOL}")


add_lit_testsuite(check-toy "Running Toy regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TOY_TEST_DEPENDS}
  )
