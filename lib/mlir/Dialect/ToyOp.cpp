#include "toy/mlir/Dialect/ToyOp.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "toy/mlir/Dialect/ToyType.h"
#include <ranges>

#define GET_OP_CLASSES
#include "toy/mlir/Dialect/ToyOp.cpp.inc"

namespace mlir::toy {

static void printBinaryOp(OpAsmPrinter &printer, Operation *op) {
  printer << ' ' << op->getOperands();
  printer.printOptionalAttrDict(op->getAttrs());
  printer << " : ";

  auto resultType = *op->result_type_begin();
  if (llvm::all_of(op->getOperandTypes(),
                   [resultType](Type type) { return type == resultType; })) {
    printer << resultType;
    return;
  }

  printer.printFunctionalType(op->getOperandTypes(), op->getResultTypes());
}

static ParseResult parseBinaryOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 2> operands;
  SMLoc operandsLoc = parser.getCurrentLocation();
  Type type;
  if (parser.parseOperandList(operands, 2) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type))
    return failure();

  if (FunctionType funcType = type.dyn_cast<FunctionType>()) {
    if (parser.resolveOperands(operands, funcType.getInputs(), operandsLoc,
                               result.operands))
      return failure();
    result.addTypes(funcType.getResults());
    return success();
  }

  if (parser.resolveOperands(operands, type, result.operands))
    return failure();
  result.addTypes(type);
  return success();
}

//===----------------------------------------------------------------------===//
/// ConstantOp
//===----------------------------------------------------------------------===//

void ConstantOp::build(OpBuilder &builder, OperationState &result,
                       double value) {
  auto f64T = builder.getF64Type();
  auto f64TensorType = RankedTensorType::get({}, f64T);
  auto denseAttribute = DenseElementsAttr::get(f64TensorType, value);
  build(builder, result, f64TensorType, denseAttribute);
}

ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  mlir::DenseElementsAttr value;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(value, "value", result.attributes))
    return failure();

  result.addTypes(value.getType());
  return success();
}

void ConstantOp::print(OpAsmPrinter &printer) {
  printer << ' ';
  printer.printOptionalAttrDict((*this)->getAttrs(), {"value"});
  printer << getValue();
}

LogicalResult ConstantOp::verify() {
  auto resultType = getResult().getType();
  if (!resultType.isa<RankedTensorType>())
    return success();

  auto rankedType = resultType.cast<RankedTensorType>();
  auto attrType = getValue().getType().cast<RankedTensorType>();
  if (rankedType.getRank() != attrType.getRank())
    return emitOpError(
               "return type must match the one of the attached value attribute")
           << attrType.getRank() << " != " << rankedType.getRank();

  for (auto dim = 0u; dim < rankedType.getRank(); ++dim) {
    if (rankedType.getShape()[dim] != attrType.getShape()[dim])

      return emitOpError(
                 "return type shape mismatches its attribute at dimension ")
             << dim << ": " << attrType.getShape()[dim]
             << " != " << rankedType.getShape()[dim];
  }

  return success();
}

//===----------------------------------------------------------------------===//
/// AddOp
//===----------------------------------------------------------------------===//

void AddOp::build(OpBuilder &builder, OperationState &odsState, Value lhs,
                  Value rhs) {
  odsState.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  odsState.addOperands({lhs, rhs});
}

ParseResult AddOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseBinaryOp(parser, result);
}

void AddOp::print(OpAsmPrinter &printer) { printBinaryOp(printer, *this); }

void AddOp::inferShapes() { getResult().setType(getLhs().getType()); }

//===----------------------------------------------------------------------===//
/// MulOp
//===----------------------------------------------------------------------===//

void MulOp::build(OpBuilder &builder, OperationState &odsState, Value lhs,
                  Value rhs) {
  odsState.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  odsState.addOperands({lhs, rhs});
}

ParseResult MulOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseBinaryOp(parser, result);
}

void MulOp::print(OpAsmPrinter &printer) { printBinaryOp(printer, *this); }

void MulOp::inferShapes() { getResult().setType(getLhs().getType()); }

//===----------------------------------------------------------------------===//
/// FuncOp
//===----------------------------------------------------------------------===//

void FuncOp::build(OpBuilder &builder, OperationState &odsState, StringRef name,
                   FunctionType type, ArrayRef<NamedAttribute> attrs) {
  buildWithEntryBlock(builder, odsState, name, type, attrs, type.getInputs());
}

void FuncOp::print(OpAsmPrinter &printer) {
  return function_interface_impl::printFunctionOp(
      printer, *this, false, getFunctionTypeAttrName(), getArgAttrsAttrName(),
      getResAttrsAttrName());
}

ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto builder = [](mlir::Builder &builder, ArrayRef<Type> argTypes,
                    ArrayRef<Type> resultTypes,
                    function_interface_impl::VariadicFlag, std::string &) {
    return builder.getFunctionType(argTypes, resultTypes);
  };
  return function_interface_impl::parseFunctionOp(
      parser, result, false, getFunctionTypeAttrName(result.name), builder,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

//===----------------------------------------------------------------------===//
/// GenericCallOp
//===----------------------------------------------------------------------===//

void GenericCallOp::build(OpBuilder &builder, OperationState &odsState,
                          StringRef callee, ArrayRef<Value> arguments) {
  odsState.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  odsState.addOperands(arguments);
  odsState.addAttribute("callee",
                        SymbolRefAttr::get(builder.getContext(), callee));
}

CallInterfaceCallable GenericCallOp::getCallableForCallee() {
  return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

void GenericCallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
}

Operation::operand_range GenericCallOp::getArgOperands() { return getInputs(); }

MutableOperandRange GenericCallOp::getArgOperandsMutable() {
  return getInputsMutable();
}

//===----------------------------------------------------------------------===//
/// CastOp
//===----------------------------------------------------------------------===//

bool CastOp::areCastCompatible(::mlir::TypeRange inputs,
                               ::mlir::TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;

  auto inputType = inputs.front().dyn_cast<TensorType>();
  auto outputType = outputs.front().dyn_cast<TensorType>();
  if (!inputType || !outputType ||
      inputType.getElementType() != outputType.getElementType())
    return false;

  return !inputType.hasRank() || !outputType.hasRank() ||
         inputType == outputType;
}

void CastOp::inferShapes() { getResult().setType(getInput().getType()); }

//===----------------------------------------------------------------------===//
/// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  auto funcOp = getParentOp();

  if (getNumOperands() > 1)
    return emitOpError("expected at most 1 return operand");

  auto results = funcOp.getFunctionType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError("Unmatched return operands size(")
           << getNumOperands() << ") and function results size("
           << results.size() << ")";

  if (!hasOperand())
    return success();

  auto inputType = *operand_type_begin();
  auto resultType = results.front();

  if (inputType == resultType || inputType.isa<UnrankedTensorType>() ||
      resultType.isa<UnrankedTensorType>())
    return success();

  return emitOpError() << "type of return operand (" << inputType << ")"
                       << "doesn't match function result type (" << resultType
                       << ")";
}

//===----------------------------------------------------------------------===//
/// TransposeOp
//===----------------------------------------------------------------------===//

void TransposeOp::build(OpBuilder &builder, OperationState &odsState,
                        Value input) {
  odsState.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  odsState.addOperands(input);
}

LogicalResult TransposeOp::verify() {
  auto inputType = getOperand().getType().dyn_cast<RankedTensorType>();
  auto resultType = getResult().getType().dyn_cast<RankedTensorType>();
  if (!inputType || !resultType)
    return success();

  auto inputShape = inputType.getShape();
  auto resultShape = resultType.getShape();
  if (!std::ranges::equal(inputShape, std::ranges::reverse_view(resultShape)))
    return emitOpError(
        "expected result shape to be the transpose of the input");
  return success();
}

void TransposeOp::inferShapes() {
  auto inputType = getOperand().getType().cast<RankedTensorType>();
  auto resultType = RankedTensorType::get(
      llvm::SmallVector<std::int64_t>{inputType.getShape().rbegin(),
                                      inputType.getShape().rend()},
      inputType.getElementType());
  getResult().setType(resultType);
}

//===----------------------------------------------------------------------===//
/// StructConstantOp
//===----------------------------------------------------------------------===//

LogicalResult StructConstantOp::verify() {
  auto structType = getType();
  auto arrayAttr = getValue();
  if (!arrayAttr || arrayAttr.size() != structType.size())
    return emitOpError("attribute type mismatch. expected : ")
           << structType.size() << ", got : " << arrayAttr.size();
  return success();
}

//===----------------------------------------------------------------------===//
/// Struct AccessOp
//===----------------------------------------------------------------------===//

void StructAccessOp::build(OpBuilder &builder, OperationState &odsState,
                           Value structInst, std::size_t index) {
  auto structType = structInst.getType().cast<StructType>();
  odsState.addTypes(structType.getType(index));
  odsState.addOperands(structInst);
  odsState.addAttribute("index", builder.getI64IntegerAttr(index));
}

LogicalResult StructAccessOp::verify() {
  auto structType = getInput().getType().cast<StructType>();

  auto index = getIndex();
  if (index >= structType.size())
    return emitOpError("index out of range. size : ")
           << structType.size() << ", got : " << index;

  auto resultType = getOutput().getType();
  if (resultType != structType.getType(index))
    return emitOpError("result type mismatch. expected : ")
           << structType.getType(index) << ", got : " << resultType;
  return success();
}

} // namespace mlir::toy
