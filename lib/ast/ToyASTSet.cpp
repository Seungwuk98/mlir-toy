#include "toy/ast/ToyASTSet.h"
#include "ast/ASTBuilder.h"
#include "ast/ASTTypeID.h"
#include "toy/ast/ToyExpr.h"
#include "toy/ast/ToyStmt.h"

DEFINE_TYPE_ID(::toy::ToyASTSet)

namespace toy {

void ToyASTSet::RegisterSet() {
  ast::ASTBuilder::registerAST<
#define AST_TABLEGEN_ID_COMMA
#include "toy/ast/ToyExpr.hpp.inc"
      >(getContext());

  ast::ASTBuilder::registerAST<
#define AST_TABLEGEN_ID_COMMA
#include "toy/ast/ToyStmt.hpp.inc"
      >(getContext());
}

} // namespace toy
