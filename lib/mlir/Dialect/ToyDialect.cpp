#include "toy/mlir/Dialect/ToyDialect.h"
#include "mlir/IR/Region.h"
#include "mlir/Transforms/InliningUtils.h"
#include "toy/mlir/Dialect/ToyOp.h"

/// tablegen generated .inc file
#include "toy/mlir/Dialect/ToyDialect.cpp.inc"
#include "llvm/ADT/ArrayRef.h"

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
#define GEN_TYPEDEF_LIST
#include "toy/mlir/Dialect/ToyType.cpp.inc"
      >();
}

} // namespace mlir::toy
