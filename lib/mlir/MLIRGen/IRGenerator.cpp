#include "toy/mlir/MLIRGen/IRGenerator.h"
#include "ast/ASTWalker.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "toy/ast/ToyExpr.h"
#include "toy/ast/ToyStmt.h"
#include "toy/mlir/Dialect/ToyOp.h"
#include "toy/reporter/DiagnosticReporter.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/ErrorHandling.h"
#include <functional>
#include <numeric>

namespace toy {

IRGenerator::IRGenerator(ToyContext *ctx, llvm::SourceMgr &srcMgr,
                         DiagnosticReporter &reporter)
    : ctx(ctx), srcMgr(srcMgr), builder(ctx), reporter(reporter) {}

void IRGenerator::visit(Number expr) {
  llvm::APFloat value(llvm::APFloat::IEEEdouble(), expr.getValue());
  result = builder.create<mlir::toy::ConstantOp>(getLoc(expr),
                                                 value.convertToDouble());
}

void IRGenerator::visit(Tensor expr) {
  auto shape = expr.getShapeTag();
  auto elementSize = std::accumulate(shape.begin(), shape.end(), 1,
                                     std::multiplies<std::size_t>());
  llvm::SmallVector<mlir::Attribute> elements;
  elements.reserve(elementSize);

  expr.walk<ast::WalkOrder::PreOrder>([&elements, this](AST ast) {
    if (auto number = ast.dyn_cast<Number>()) {
      llvm::APFloat value(llvm::APFloat::IEEEdouble(), number.getValue());
      elements.emplace_back(builder.getF64FloatAttr(value.convertToDouble()));
    }
    return ast::WalkResult::success();
  });

  assert(elements.size() == elementSize && "invalid shape and element size");

  auto tensorAttr = mlir::DenseElementsAttr::get(
      mlir::RankedTensorType::get(shape, builder.getF64Type()), elements);

  result = builder.create<mlir::toy::ConstantOp>(getLoc(expr), tensorAttr);
}

void IRGenerator::visit(BinaryOp expr) {
  auto lhs = expr.getLhs();
  auto rhs = expr.getRhs();

  lhs.accept(*this);
  if (reporter.getErrorCount())
    return;
  auto lhsV = result;

  rhs.accept(*this);
  if (reporter.getErrorCount())
    return;
  auto rhsV = result;

  switch (expr.getOpKind()) {
  case BinaryOpKind::Add:
    result = builder.create<mlir::toy::AddOp>(getLoc(expr), lhsV, rhsV);
    break;
  case BinaryOpKind::Mul:
    result = builder.create<mlir::toy::MulOp>(getLoc(expr), lhsV, rhsV);
    break;
  default:
    llvm_unreachable("Not implemented");
  }
}

void IRGenerator::visit(FunctionCall expr) {
  auto callee = expr.getCallee();
  auto args = expr.getArgs();

  llvm::SmallVector<mlir::Value> argValues;
  argValues.reserve(args.size());
  for (auto arg : args) {
    arg.accept(*this);
    if (reporter.getErrorCount())
      return;
    argValues.emplace_back(result);
  }

  result =
      builder.create<mlir::toy::GenericCallOp>(getLoc(expr), callee, argValues);
}

void IRGenerator::visit(Identifier expr) {
  if (symbolTable.count(expr.getName())) {
    result = symbolTable.lookup(expr.getName());
    return;
  }
  /// TODO report
}

void IRGenerator::visit(Transpose expr) {
  auto target = expr.getTarget();
  target.accept(*this);
  if (reporter.getErrorCount())
    return;
  auto targetV = result;

  result = builder.create<mlir::toy::TransposeOp>(getLoc(expr), targetV);
}

void IRGenerator::visit(Print expr) {
  auto target = expr.getTarget();
  target.accept(*this);
  if (reporter.getErrorCount())
    return;
  auto targetV = result;

  builder.create<mlir::toy::PrintOp>(getLoc(expr), targetV);
}

void IRGenerator::visit(Module moduleStmt) {
  auto moduleOp = builder.create<mlir::ModuleOp>(getLoc(moduleStmt));
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(moduleOp.getBody());
  ScopeTy scope(symbolTable);
  for (auto stmt : moduleStmt.getStmts()) {
    stmt.accept(*this);
    if (reporter.getErrorCount())
      return;
  }

  module = moduleOp;
}

void IRGenerator::visit(BlockStmt stmt) {
  ScopeTy scope(symbolTable);
  for (auto stmt : stmt.getStmts()) {
    stmt.accept(*this);
    if (reporter.getErrorCount())
      return;
  }
}

void IRGenerator::visit(FuncDecl stmt) {
  auto functionName = stmt.getName();

  auto f64TensorType = mlir::UnrankedTensorType::get(builder.getF64Type());

  mlir::Type returnType;
  bool hasReturn = false;
  stmt.getBody().walk([&](AST ast) {
    if (auto stmt = ast.dyn_cast<ReturnStmt>()) {
      hasReturn = true;
      if (stmt.getExpr())
        returnType = f64TensorType;
      return ast::WalkResult::interrupt();
    }
    return ast::WalkResult::success();
  });

  llvm::SmallVector<mlir::Type> argTypes(stmt.getParams().size(),
                                         f64TensorType);
  auto funcType = builder.getFunctionType(
      argTypes, returnType ? returnType : mlir::TypeRange{});

  auto funcOp =
      builder.create<mlir::toy::FuncOp>(getLoc(stmt), functionName, funcType);

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&funcOp.getBlocks().front());

  ScopeTy scope(symbolTable);
  for (auto [idx, param] : llvm::enumerate(stmt.getParams())) {
    if (symbolTable.count(param)) {
      /// TODO report
      return;
    }
    symbolTable.insert(param, funcOp.getArgument(idx));
  }

  stmt.getBody().accept(*this);
  if (!hasReturn)
    builder.create<mlir::toy::ReturnOp>(getLoc(stmt.getLoc().End),
                                        mlir::ValueRange{});
}

void IRGenerator::visit(VarDecl stmt) {
  auto varName = stmt.getName();
  stmt.getInit().accept(*this);
  if (reporter.getErrorCount())
    return;
  if (stmt.getShape()) {
    auto reshapedType =
        mlir::RankedTensorType::get(*stmt.getShape(), builder.getF64Type());
    result = builder.create<mlir::toy::ReshapeOp>(getLoc(stmt), reshapedType,
                                                  result);
  }
  symbolTable.insert(varName, result);
}

void IRGenerator::visit(ExprStmt stmt) { stmt.getExpr().accept(*this); }

void IRGenerator::visit(ReturnStmt stmt) {
  if (stmt.getExpr()) {
    stmt.getExpr()->accept(*this);
    if (reporter.getErrorCount())
      return;
    builder.create<mlir::toy::ReturnOp>(getLoc(stmt), result);
    return;
  }
  builder.create<mlir::toy::ReturnOp>(getLoc(stmt), mlir::ValueRange{});
}

} // namespace toy
