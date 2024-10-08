#include "toy/ast/ToyExpr.h"

namespace toy {

bool Expr::classof(AST ast) {
  return ast.isa<
#define AST_TABLEGEN_ID_COMMA
#include "toy/ast/ToyExpr.hpp.inc"
      >();
}

void Number::print(toy::Number ast, ::ast::ASTPrinter &printer) {
  printer.OS() << ast.getValue();
}

void Tensor::print(toy::Tensor ast, ::ast::ASTPrinter &printer) {
  printer.OS() << '[';
  for (auto I = ast.getElements().begin(), E = ast.getElements().end(); I != E;
       ++I) {
    if (I != ast.getElements().begin())
      printer.OS() << ", ";
    I->print(printer);
  }
  printer.OS() << ']';
}

void BinaryOp::print(toy::BinaryOp ast, ::ast::ASTPrinter &printer) {
  ast.getLhs().print(printer);

  switch (ast.getOpKind()) {
  case BinaryOpKind::Add:
    printer.OS() << " + ";
    break;
  case BinaryOpKind::Sub:
    printer.OS() << " - ";
    break;
  case BinaryOpKind::Mul:
    printer.OS() << " * ";
    break;
  case BinaryOpKind::Div:
    printer.OS() << "/";
    break;
  }
  ast.getRhs().print(printer);
}

void FunctionCall::print(toy::FunctionCall ast, ::ast::ASTPrinter &printer) {
  printer.OS() << ast.getCallee() << '(';
  for (auto I = ast.getArgs().begin(), E = ast.getArgs().end(); I != E; ++I) {
    if (I != ast.getArgs().begin())
      printer.OS() << ", ";
    I->print(printer);
  }
  printer.OS() << ')';
}

void Identifier::print(toy::Identifier ast, ::ast::ASTPrinter &printer) {
  printer.OS() << ast.getName();
}

void Transpose::print(toy::Transpose ast, ::ast::ASTPrinter &printer) {
  printer.OS() << "transpose(";
  ast.getTarget().print(printer);
  printer.OS() << ')';
}

void Print::print(toy::Print ast, ::ast::ASTPrinter &printer) {
  printer.OS() << "print(";
  ast.getTarget().print(printer);
  printer.OS() << ')';
}

} // namespace toy

#define AST_TABLEGEN_DEF
#include "toy/ast/ToyExpr.cpp.inc"
