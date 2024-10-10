#ifndef TOY_AST_YAML_H
#define TOY_AST_YAML_H

#include "ast/ASTVisitor.h"
#include "toy/ast/ToyAST.h"
#include "toy/ast/ToyExpr.h"
#include "toy/ast/ToyStmt.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

namespace toy {

class YamlPrinter : public ast::VisitorBase<YamlPrinter
#define AST_TABLEGEN_ID(ID) , ID
#include "toy/ast/ToyExpr.hpp.inc"

#define AST_TABLEGEN_ID(ID) , ID
#include "toy/ast/ToyStmt.hpp.inc"
                                            > {
public:
  YamlPrinter(llvm::yaml::Output &Out) : Out(Out) {}

#define AST_TABLEGEN_ID(ID) void visit(ID ast);
#include "toy/ast/ToyExpr.hpp.inc"

#define AST_TABLEGEN_ID(ID) void visit(ID ast);
#include "toy/ast/ToyStmt.hpp.inc"

private:
  llvm::yaml::Output &Out;
};

class YamlOutput {
public:
  YamlOutput(llvm::raw_ostream &OS) : OS(OS), Out(OS) {}

  llvm::yaml::Output &getOutput() { return Out; }

private:
  llvm::raw_ostream &OS;
  llvm::yaml::Output Out;
};

} // namespace toy

#endif // TOY_AST_YAML_H
