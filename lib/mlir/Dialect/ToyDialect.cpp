#include "toy/mlir/Dialect/ToyDialect.h"
#include "mlir/IR/Region.h"
#include "mlir/Transforms/InliningUtils.h"
#include "toy/mlir/Dialect/ToyOp.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

/// tablegen generated .inc file
#include "toy/mlir/Dialect/ToyDialect.cpp.inc"
namespace mlir::toy {

struct ToyInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }

  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &mapping) const final {
    return true;
  }

  void handleTerminator(Operation *op, ValueRange valuesToRepl) const final {
    auto returnOp = cast<ReturnOp>(op);

    assert(returnOp.getNumOperands() == valuesToRepl.size() &&
           "mismatch in number of return operands");
    for (const auto &[idx, operand] : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[idx].replaceAllUsesWith(operand);
  }

  Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                       Type resultType,
                                       Location loc) const final {
    return builder.create<CastOp>(loc, resultType, input);
  }
};

void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "toy/mlir/Dialect/ToyOp.cpp.inc"
      >();

  addInterfaces<ToyInlinerInterface>();
  addTypes<
#define GET_TYPEDEF_LIST
#include "toy/mlir/Dialect/ToyType.cpp.inc"
      >();
}

void ToyDialect::printType(Type type, DialectAsmPrinter &os) const {
  llvm::TypeSwitch<Type, void>(type)
      .Case([&](StructType structType) {
        os << "struct<";
        interleaveComma(structType.getElementTypes(), os);
        os << '>';
      })
      .Default([](Type) { llvm_unreachable("unknown 'toy' type"); });
}

Type ToyDialect::parseType(DialectAsmParser &parser) const {
  if (parser.parseKeyword("struct") || parser.parseLess())
    return nullptr;

  SmallVector<Type, 1> elementTypes;
  do {
    SMLoc typeLoc = parser.getCurrentLocation();
    Type elementType;
    if (parser.parseType(elementType))
      return nullptr;

    if (!elementType.isa<TensorType>()) {
      parser.emitError(typeLoc, "expected tensor type");
      return nullptr;
    }
    elementTypes.emplace_back(elementType);
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseGreater())
    return nullptr;
  return StructType::get(elementTypes);
}

Operation *ToyDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                           Type type, Location loc) {
  if (type.isa<StructType>())
    return builder.create<StructConstantOp>(loc, type, value.cast<ArrayAttr>());

  return builder.create<ConstantOp>(loc, type, value.cast<DenseElementsAttr>());
}

} // namespace mlir::toy
