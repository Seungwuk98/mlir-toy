set(LLVM_TARGET_DEFINITIONS ToyExpr.td)
ast_tablegen(ToyExpr.hpp.inc --ast-decl-gen)
ast_tablegen(ToyExpr.cpp.inc --ast-def-gen)
add_public_tablegen_target(ToyExprGen)

set(LLVM_TARGET_DEFINITIONS ToyStmt.td)
ast_tablegen(ToyStmt.hpp.inc --ast-decl-gen)
ast_tablegen(ToyStmt.cpp.inc --ast-def-gen)
add_public_tablegen_target(ToyStmtGen)

