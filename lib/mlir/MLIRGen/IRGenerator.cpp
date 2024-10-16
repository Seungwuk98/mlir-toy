#include "toy/mlir/MLIRGen/IRGenerator.h"
#include "ast/ASTWalker.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Transforms/FoldUtils.h"
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
  result = {builder.create<mlir::toy::ConstantOp>(getLoc(expr),
                                                  value.convertToDouble()),
            nullptr};
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

  result = {builder.create<mlir::toy::ConstantOp>(getLoc(expr), tensorAttr),
            nullptr};
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

  mlir::Value resultV;
  switch (expr.getOpKind()) {
  case BinaryOpKind::Add:
    resultV = builder.create<mlir::toy::AddOp>(getLoc(expr), lhsV.V, rhsV.V);
    break;
  case BinaryOpKind::Mul:
    resultV = builder.create<mlir::toy::MulOp>(getLoc(expr), lhsV.V, rhsV.V);
    break;
  default:
    llvm_unreachable("Not implemented");
  }

  result = {resultV, nullptr};
}

void IRGenerator::visit(FunctionCall expr) {
  auto callee = expr.getCallee();
  mlir::FunctionType funcType;
  if (auto iter = functionDeclarations.find(callee);
      iter != functionDeclarations.end()) {
    funcType = iter->second;
  } else {
    Report(expr.getLoc(), Reporter::Diag::err_undeclared_function, callee);
    return;
  }

  auto args = expr.getArgs();
  if (args.size() != funcType.getNumInputs()) {
    Report(expr.getLoc(), Reporter::Diag::err_unmatched_function_args_count,
           funcType.getNumInputs(), args.size());
    return;
  }

  llvm::SmallVector<mlir::Value> argValues;
  argValues.reserve(args.size());
  for (auto arg : args) {
    arg.accept(*this);
    if (reporter.getErrorCount())
      return;
    argValues.emplace_back(result.V);
  }

  mlir::Value resultV;
  if (funcType.getNumResults())
    resultV = builder.create<mlir::toy::GenericCallOp>(getLoc(expr), callee,
                                                       argValues);
  else
    builder.create<mlir::toy::GenericCallOp>(getLoc(expr), mlir::TypeRange{},
                                             callee, argValues);

  result = {resultV, nullptr};
}

void IRGenerator::visit(Identifier expr) {
  if (symbolTable.count(expr.getName())) {
    result = symbolTable.lookup(expr.getName());
    return;
  }

  Report(expr.getLoc(), Reporter::Diag::err_undeclared_variable,
         expr.getName());
}

void IRGenerator::visit(Transpose expr) {
  auto target = expr.getTarget();
  target.accept(*this);
  if (reporter.getErrorCount())
    return;
  auto targetV = result.V;

  result = {builder.create<mlir::toy::TransposeOp>(getLoc(expr), targetV),
            nullptr};
}

void IRGenerator::visit(Print expr) {
  auto target = expr.getTarget();
  target.accept(*this);
  if (reporter.getErrorCount())
    return;
  auto targetV = result.V;

  builder.create<mlir::toy::PrintOp>(getLoc(expr), targetV);
}

void IRGenerator::visit(StructAccess expr) {
  auto target = expr.getLeft();
  target.accept(*this);
  if (reporter.getErrorCount())
    return;
  auto targetV = result.V;

  if (!targetV.getType().isa<mlir::toy::StructType>()) {
    Report(expr.getLoc(), Reporter::Diag::err_expected_struct_type,
           targetV.getType());
    return;
  }

  auto structDecl = result.D;
  auto field = expr.getField();

  auto fieldIter = llvm::find(structDecl.getFields(), field);
  if (fieldIter == structDecl.getFields().end()) {
    Report(expr.getLoc(), Reporter::Diag::err_undefined_field_name, field,
           structDecl.getName());
    Report(structDecl.getLoc(), Reporter::Diag::note_previous_declared_struct);
    return;
  }

  auto fieldIdx = std::distance(structDecl.getFields().begin(), fieldIter);

  result = {builder.create<mlir::toy::StructAccessOp>(getLoc(expr), targetV,
                                                      fieldIdx),
            nullptr};
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

  llvm::SmallVector<mlir::Type> argTypes;
  argTypes.reserve(stmt.getParams().size());

  for (auto idx = 0u; idx < stmt.getParams().size(); ++idx) {
    if (auto iter = stmt.getParamStructDeclsTag().find(idx);
        iter != stmt.getParamStructDeclsTag().end()) {
      llvm::SmallVector<mlir::Type> elementTypes(
          iter->second.getFields().size(), f64TensorType);
      argTypes.emplace_back(mlir::toy::StructType::get(elementTypes));
    } else {
      argTypes.emplace_back(f64TensorType);
    }
  }

  auto funcType = builder.getFunctionType(
      argTypes, returnType ? returnType : mlir::TypeRange{});
  auto inserted = functionDeclarations.try_emplace(functionName, funcType);
  if (!inserted.second) {
    Report(stmt.getLoc(), Reporter::Diag::err_duplicated_function,
           functionName);
    return;
  }

  auto funcOp =
      builder.create<mlir::toy::FuncOp>(getLoc(stmt), functionName, funcType);
  if (functionName != "main")
    funcOp.setPrivate();

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&funcOp.getBlocks().front());

  ScopeTy scope(symbolTable);
  for (auto [idx, param] : llvm::enumerate(stmt.getParams())) {
    const auto &[structOpt, name] = param;

    if (symbolTable.count(name)) {
      Report(stmt.getLoc(), Reporter::Diag::err_duplicated_function_parameter,
             name);
      return;
    }

    StructDecl structDecl = nullptr;
    if (structOpt)
      structDecl = stmt.getParamStructDeclsTag().lookup(idx);

    symbolTable.insert(name, {funcOp.getArgument(idx), structDecl});
  }

  stmt.getBody().accept(*this);
  if (!hasReturn)
    builder.create<mlir::toy::ReturnOp>(getLoc(stmt.getLoc().End),
                                        mlir::ValueRange{});
}

void IRGenerator::visit(StructDecl stmt) {
  // do nothing
}

void IRGenerator::visit(VarDecl stmt) {
  auto varName = stmt.getName();
  if (symbolTable.count(varName)) {
    Report(stmt.getLoc(), Reporter::Diag::err_already_declared_varible,
           varName);
    return;
  }

  stmt.getInit().accept(*this);
  if (reporter.getErrorCount())
    return;
  if (stmt.getShape()) {
    auto reshapedType =
        mlir::RankedTensorType::get(*stmt.getShape(), builder.getF64Type());
    result.V = builder.create<mlir::toy::ReshapeOp>(getLoc(stmt), reshapedType,
                                                    result.V);
  }
  symbolTable.insert(varName, {result.V, nullptr});
}

void IRGenerator::visit(StructVarDecl stmt) {
  auto structDecl = stmt.getStructDeclTag();
  auto varName = stmt.getName();
  if (symbolTable.count(varName)) {
    Report(stmt.getLoc(), Reporter::Diag::err_already_declared_varible,
           varName);
    return;
  }

  auto initValues = stmt.getInit();

  llvm::SmallVector<mlir::Attribute> elements;
  llvm::SmallVector<mlir::Type> elementTypes;
  elements.reserve(initValues.size());
  elementTypes.reserve(initValues.size());

  /// Initialize values are tensor constants. This constraints are ensured in
  /// Parser.
  for (auto init : initValues) {
    init.accept(*this);
    if (reporter.getErrorCount())
      return;
    auto value = result.V;

    llvm::SmallVector<mlir::OpFoldResult> structElements;
    auto foldSuccess = value.getDefiningOp()->fold(structElements);
    assert(mlir::succeeded(foldSuccess) && "expected constant value");
    assert(structElements.size() == 1 &&
           llvm::isa<mlir::Attribute>(structElements[0]) &&
           "expected constant value");
    elements.emplace_back(llvm::cast<mlir::Attribute>(structElements[0]));
    elementTypes.emplace_back(value.getType());
  }

  auto structType = mlir::toy::StructType::get(elementTypes);
  auto arrayAttr = mlir::ArrayAttr::get(ctx, elements);
  result.V = builder.create<mlir::toy::StructConstantOp>(getLoc(stmt),
                                                         structType, arrayAttr);
  result.D = structDecl;
  symbolTable.insert(varName, result);
}

void IRGenerator::visit(ExprStmt stmt) { stmt.getExpr().accept(*this); }

void IRGenerator::visit(ReturnStmt stmt) {
  if (stmt.getExpr()) {
    stmt.getExpr()->accept(*this);
    if (reporter.getErrorCount())
      return;
    builder.create<mlir::toy::ReturnOp>(getLoc(stmt), result.V);
    return;
  }
  builder.create<mlir::toy::ReturnOp>(getLoc(stmt), mlir::ValueRange{});
}

static llvm::SourceMgr::DiagKind irGenDiagKinds[] = {
#define DIAG(ID, Kind, Msg) llvm::SourceMgr::DK_##Kind,
#include "toy/mlir/MLIRGen/IRGeneratorDiagnostic.def"
};

static llvm::StringLiteral irGenDiagMsg[] = {
#define DIAG(ID, Kind, Msg) Msg,
#include "toy/mlir/MLIRGen/IRGeneratorDiagnostic.def"
};

llvm::SourceMgr::DiagKind IRGenerator::Reporter::getDiagKind(Diag diag) {
  return irGenDiagKinds[diag];
}

llvm::StringLiteral IRGenerator::Reporter::getDiagMsg(Diag diag) {
  return irGenDiagMsg[diag];
}

} // namespace toy
