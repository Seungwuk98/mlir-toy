#ifndef TOY_MLIR_IRGENERATOR_H
#define TOY_MLIR_IRGENERATOR_H

#include "ast/ASTVisitor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OwningOpRef.h"
#include "toy/ast/ToyExpr.h"
#include "toy/ast/ToyStmt.h"
#include "toy/context/ToyContext.h"
#include "toy/mlir/Dialect/ToyOp.h"
#include "toy/reporter/DiagnosticReporter.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SourceMgr.h"

namespace toy {

struct ToyValue {

  ToyValue() = default;
  ToyValue(mlir::Value V, StructDecl D) : V(V), D(D) {}

  mlir::Value V;
  StructDecl D;
};

class IRGenerator : public ast::VisitorBase<IRGenerator
#define AST_TABLEGEN_ID(ID) , ID
#include "toy/ast/ToyExpr.h.inc"

#define AST_TABLEGEN_ID(ID) , ID
#include "toy/ast/ToyStmt.h.inc"

                                            > {
public:
  IRGenerator(ToyContext *ctx, llvm::SourceMgr &srcMgr,
              DiagnosticReporter &reporter);

#define AST_TABLEGEN_ID(ID) void visit(ID expr);
#include "toy/ast/ToyExpr.h.inc"

#define AST_TABLEGEN_ID(ID) void visit(ID stmt);
#include "toy/ast/ToyStmt.h.inc"

  mlir::OwningOpRef<mlir::ModuleOp> getModuleOp() { return module.release(); }

  struct Reporter {
    enum Diag {
#define DIAG(ID, ...) ID,
#include "toy/mlir/MLIRGen/IRGeneratorDiagnostic.def"
    };

    static llvm::SourceMgr::DiagKind getDiagKind(Diag diag);
    static llvm::StringLiteral getDiagMsg(Diag diag);
  };

  template <typename... Args>
  void Report(llvm::SMRange loc, Reporter::Diag diag, Args &&...args) {
    auto kind = Reporter::getDiagKind(diag);
    auto msg = Reporter::getDiagMsg(diag);
    auto evaledMsg = llvm::formatv(msg.data(), std::forward<Args>(args)...);
    reporter.Report(loc, kind, evaledMsg.str());
  }

private:
  mlir::Location getLoc(llvm::SMLoc loc) {
    auto bufID = srcMgr.FindBufferContainingLoc(loc);
    auto lineAndCol = srcMgr.getLineAndColumn(loc, bufID);
    auto bufIdentifier = srcMgr.getMemoryBuffer(bufID)->getBufferIdentifier();
    return mlir::FileLineColLoc::get(ctx, bufIdentifier, lineAndCol.first,
                                     lineAndCol.second);
  }

  mlir::Location getLoc(AST ast) {
    auto loc = ast.getLoc().Start;
    return getLoc(loc);
  }

  ToyContext *ctx;
  llvm::SourceMgr &srcMgr;
  mlir::OpBuilder builder;
  DiagnosticReporter &reporter;
  mlir::OwningOpRef<mlir::ModuleOp> module;

  llvm::ScopedHashTable<llvm::StringRef, ToyValue> symbolTable;
  llvm::DenseMap<llvm::StringRef, mlir::FunctionType> functionDeclarations;
  ToyValue result;

  using ScopeTy = llvm::ScopedHashTable<llvm::StringRef, ToyValue>::ScopeTy;
};

} // namespace toy

#endif // TOY_MLIR_IRGENERATOR_H
