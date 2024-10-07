#include "toy/ast/ToyExpr.h"

namespace toy {

bool Expr::classof(AST ast) {
  return ast.isa<
#define AST_TABLEGEN_ID_COMMA
#include "toy/ast/ToyExpr.hpp.inc"
      >();
}

} // namespace toy

#define AST_TABLEGEN_DEF
#include "toy/ast/ToyExpr.cpp.inc"
