#include "toy/ast/ToyStmt.h"
#include "ast/ASTPrinter.h"

namespace toy {

bool Stmt::classof(AST ast) {
  return ast.isa<
#define AST_TABLEGEN_ID_COMMA
#include "toy/ast/ToyStmt.h.inc"
      >();
}

void Module::print(toy::Module ast, ::ast::ASTPrinter &printer) {
  for (auto stmt : ast.getStmts()) {
    stmt.print(printer);
    printer.Line();
  }
}

void BlockStmt::print(toy::BlockStmt ast, ::ast::ASTPrinter &printer) {
  printer.OS() << '{';
  {
    ast::ASTPrinter::AddIndentScope indentScope(printer, 2);
    for (auto stmt : ast.getStmts()) {
      stmt.print(printer.PrintLine());
    }
  }
  printer.Line() << '}';
}

void FuncDecl::print(toy::FuncDecl ast, ::ast::ASTPrinter &printer) {
  printer.OS() << "def " << ast.getName() << '(';
  for (auto I = ast.getParams().begin(), E = ast.getParams().end(); I != E;
       ++I) {
    if (I != ast.getParams().begin())
      printer.OS() << ", ";
    printer.OS() << *I;
  }
  printer.OS() << ") ";
  BlockStmt::print(ast.getBody(), printer);
}

void StructDecl::print(toy::StructDecl ast, ::ast::ASTPrinter &printer) {
  printer.OS() << "struct " << ast.getName() << " {";
  {
    ast::ASTPrinter::AddIndentScope indentScope(printer, 2);
    for (auto field : ast.getFields()) {
      printer.Line() << "var " << field << ';';
    }
  }
  printer.Line() << '}';
}

void VarDecl::print(toy::VarDecl ast, ::ast::ASTPrinter &printer) {
  printer.OS() << "var " << ast.getName();
  if (ast.getShape()) {
    printer.OS() << '<';
    for (auto I = ast.getShape()->begin(), E = ast.getShape()->end(); I != E;
         ++I) {
      if (I != ast.getShape()->begin())
        printer.OS() << ", ";
      printer.OS() << *I;
    }
    printer.OS() << '>';
  }
  printer.OS() << " = ";
  ast.getInit().print(printer);
  printer.OS() << ';';
}

void StructVarDecl::print(toy::StructVarDecl ast, ::ast::ASTPrinter &printer) {
  printer.OS() << ast.getStructName() << ' ' << ast.getName() << " = ";
  printer.OS() << '{';
  for (auto I = ast.getInit().begin(), E = ast.getInit().end(); I != E; ++I) {
    if (I != ast.getInit().begin())
      printer.OS() << ", ";
    I->print(printer);
  }
  printer.OS() << "};";
}

void ExprStmt::print(toy::ExprStmt ast, ::ast::ASTPrinter &printer) {
  ast.getExpr().print(printer);
  printer.OS() << ';';
}

void ReturnStmt::print(toy::ReturnStmt ast, ::ast::ASTPrinter &printer) {
  printer.OS() << "return";
  if (ast.getExpr()) {
    printer.OS() << ' ';
    ast.getExpr()->print(printer);
  }
  printer.OS() << ';';
}
} // namespace toy

#define AST_TABLEGEN_DEF
#include "toy/ast/ToyStmt.cpp.inc"
