#include "toy/ast/ToyStmt.h"

namespace toy {

bool Stmt::classof(AST ast) {
  return ast.isa<
#define AST_TABLEGEN_ID_COMMA
#include "toy/ast/ToyStmt.hpp.inc"
      >();
}

} // namespace toy

#define AST_TABLEGEN_DEF
#include "toy/ast/ToyExpr.cpp.inc"
