#ifndef TOY_EXPR_H
#define TOY_EXPR_H

#include "ast/AST.h"
#include "ast/ASTDataHandler.h"
#include "toy/ast/ToyAST.h"
#include "llvm/ADT/ArrayRef.h"

namespace toy {

class Expr : public ToyAST {
public:
  using ToyAST::ToyAST;
  static bool classof(AST ast);
};

enum class BinaryOpKind {
  Add,
  Sub,
  Mul,
  Div,
};

} // namespace toy

namespace ast {
template <> struct detail::ASTDataHandler<toy::BinaryOpKind> {
  static bool isEqual(toy::BinaryOpKind lhs, toy::BinaryOpKind rhs) {
    return lhs == rhs;
  }

  static void walk(toy::BinaryOpKind data, std::function<void(AST)>) {}
};
} // namespace ast

#define AST_TABLEGEN_DECL
#include "toy/ast/ToyExpr.hpp.inc"

#endif // TOY_EXPR_H
