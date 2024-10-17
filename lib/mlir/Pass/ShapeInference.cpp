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
      return operand.getType().isa<RankedTensorType>();
    });
  }

  static bool mustBeInferred(Operation *op) {
    return llvm::any_of(op->getResults(), [](Value result) {
      return result.getType().isa<UnrankedTensorType>();
    });
  }
};

std::unique_ptr<Pass> createShapeInferencePass() {
  return std::make_unique<ShapeInferencePass>();
}

void registerShapeInferencePass() { registerPass(createShapeInferencePass); }

} // namespace mlir::toy
