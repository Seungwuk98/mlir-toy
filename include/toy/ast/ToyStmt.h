#ifndef TOY_STMT_H
#define TOY_STMT_H

#include "ast/AST.h"
#include "toy/ast/ToyAST.h"
#include "toy/ast/ToyExpr.h"
#include "llvm/ADT/ArrayRef.h"
#include <optional>

namespace toy {

class Stmt : public ToyAST {
public:
  using ToyAST::ToyAST;
  static bool classof(AST ast);
};

} // namespace toy

#define AST_TABLEGEN_DECL
#include "toy/ast/ToyStmt.hpp.inc"

#endif // TOY_STMT_H
