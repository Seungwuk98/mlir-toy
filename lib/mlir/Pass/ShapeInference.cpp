#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "toy/mlir/Dialect/ToyOp.h"
#include <llvm-18/llvm/ADT/SmallPtrSet.h>

namespace mlir::toy {
#include "toy/mlir/Dialect/ToyInterface.cpp.inc"
}

namespace mlir::toy {

class ShapeInferencePass
    : public PassWrapper<ShapeInferencePass, OperationPass<FuncOp>> {
public:
  void runOnOperation() override {
    FuncOp func = getOperation();

    llvm::SmallPtrSet<Operation *, 4> ops;
    func.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
      if (mustBeInferred(op))
        ops.insert(op);
      return WalkResult::advance();
    });

    while (!ops.empty()) {

      auto iter = llvm::find_if(
          ops, [this](Operation *op) { return areAllShapesKnown(op); });
      if (iter == ops.end())
        break;

      auto *op = *iter;
      ops.erase(op);
      if (auto shapeInference = llvm::dyn_cast<ShapeInference>(op))
        shapeInference.inferShapes();
      else {
        op->emitError("unable to infer shape of operation without shape "
                      "inference interface");
        return signalPassFailure();
      }
    }

    if (!ops.empty()) {
      func->emitError("Shape inference failed");
      signalPassFailure();
    }
  }

  llvm::StringRef getArgument() const override final {
    return "toy-shape-inference";
  }

private:
  static bool areAllShapesKnown(Operation *op) {
    return llvm::all_of(op->getOperands(), [](Value operand) {
      if (operand.getType().isa<RankedTensorType>())
        return true;
      if (auto structType = operand.getType().dyn_cast<StructType>())
        return areAllShapeKnown(structType);
      return false;
    });
  }

  static bool areAllShapeKnown(StructType type) {
    return llvm::all_of(type.getElementTypes(),
                        [](Type type) { return type.isa<RankedTensorType>(); });
  }

  static bool mustBeInferred(Operation *op) {
    return llvm::any_of(op->getResults(), [](Value result) {
      if (result.getType().isa<UnrankedTensorType>())
        return true;
      if (auto structType = result.getType().dyn_cast<StructType>())
        return mustBeInferred(structType);
      return false;
    });
  }

  static bool mustBeInferred(StructType type) {
    return llvm::any_of(type.getElementTypes(), [](Type type) {
      return type.isa<UnrankedTensorType>();
    });
  }
};

std::unique_ptr<Pass> createShapeInferencePass() {
  return std::make_unique<ShapeInferencePass>();
}

void registerShapeInferencePass() { registerPass(createShapeInferencePass); }

} // namespace mlir::toy
