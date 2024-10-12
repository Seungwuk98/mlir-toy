#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "toy/mlir/Dialect/ToyOp.h"

namespace mlir::toy {

struct SimplifyRedundantTranspose : public OpRewritePattern<TransposeOp> {

  SimplifyRedundantTranspose(MLIRContext *context)
      : OpRewritePattern(context, 1) {}

  LogicalResult matchAndRewrite(TransposeOp op,
                                PatternRewriter &rewriter) const override {
    if (auto nestedTranspose = op.getOperand().getDefiningOp<TransposeOp>()) {
      rewriter.replaceOp(op, nestedTranspose.getOperand());
      return success();
    }

    return failure();
  }
};

void TransposeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<SimplifyRedundantTranspose>(context);
}

} // namespace mlir::toy
