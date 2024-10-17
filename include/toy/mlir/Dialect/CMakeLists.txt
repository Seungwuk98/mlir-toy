
# tablegen Dialect
set(LLVM_TARGET_DEFINITIONS ToyDialect.td)
mlir_tablegen(ToyDialect.h.inc -gen-dialect-decls)
mlir_tablegen(ToyDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(ToyDialectIncGen)

# tablegen Ops 
set(LLVM_TARGET_DEFINITIONS ToyOp.td)
mlir_tablegen(ToyOp.h.inc -gen-op-decls)
mlir_tablegen(ToyOp.cpp.inc -gen-op-defs)
add_public_tablegen_target(ToyOpIncGen)

# tablegen Types
set(LLVM_TARGET_DEFINITIONS ToyType.td)
mlir_tablegen(ToyType.h.inc -gen-typedef-decls)
mlir_tablegen(ToyType.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(ToyTypeIncGen)

# tablegen Interfaces
set(LLVM_TARGET_DEFINITIONS ToyInterface.td)
mlir_tablegen(ToyInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(ToyInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(ToyInterfaceIncGen)
