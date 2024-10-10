
# tablegen Dialect
set(LLVM_TARGET_DEFINITIONS ToyDialect.td)
mlir_tablegen(ToyDialect.h.inc -gen-dialect-decls)
mlir_tablegen(ToyDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(ToyDialectIncGen)

#tablegen Ops 
set(LLVM_TARGET_DEFINITIONS ToyOp.td)
mlir_tablegen(ToyOp.h.inc -gen-op-decls)
mlir_tablegen(ToyOp.cpp.inc -gen-op-defs)
add_public_tablegen_target(ToyOpIncGen)


