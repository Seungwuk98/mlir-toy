
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/DialectConversion.h"
#include "toy/mlir/Dialect/ToyDialect.h"
#include "toy/mlir/Dialect/ToyOp.h"
#include "toy/mlir/Pass/Passes.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir::toy {

class ToyToAffineLoweringPass
    : public PassWrapper<ToyToAffineLoweringPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyToAffineLoweringPass);

  void runOnOperation() override;

private:
};

static MemRefType tensor2MemRef(RankedTensorType type) {
  return MemRefType::get(type.getShape(), type.getElementType());
}

static Value insertAllocDealloc(Location loc, OpBuilder &builder,
                                MemRefType type) {
  OpBuilder::InsertionGuard guard(builder);
  auto *currBlock = builder.getBlock();

  auto alloc = builder.create<memref::AllocOp>(loc, type);
  alloc->moveBefore(&currBlock->front());

  auto dealloc = builder.create<memref::DeallocOp>(loc, alloc);
  alloc->moveBefore(&currBlock->back());

  return alloc;
}

template <typename Fn>
static void lowerOpToLoops(Operation *op, ValueRange operands,
                           PatternRewriter &rewriter, Fn &&innerBlockFn) {
  auto tensorType = op->getResult(0).getType().cast<RankedTensorType>();
  auto shape = tensorType.getShape();

  auto alloc =
      insertAllocDealloc(op->getLoc(), rewriter, tensor2MemRef(tensorType));

  SmallVector<int64_t, 4> fromArray(shape.size(), 0);
  SmallVector<int64_t, 4> stepArray(shape.size(), 1);
  affine::buildAffineLoopNest(
      rewriter, op->getLoc(), fromArray, tensorType.getShape(), stepArray,
      [&](OpBuilder &builder, Location loc, ValueRange ivs) {
        auto toStore = innerBlockFn(builder, ivs);
        builder.create<memref::StoreOp>(loc, toStore, alloc, ivs);
      });

  rewriter.replaceOp(op, alloc);
}

template <typename OpAdaptor, typename BinaryOp>
static void binaryOpToLoops(Operation *op, ValueRange operands,
                            PatternRewriter &rewriter) {
  lowerOpToLoops(op, operands, rewriter,
                 [&](OpBuilder &builder, ValueRange ivs) -> Value {
                   OpAdaptor adaptor(operands);
                   auto lhs = adaptor.getLhs();
                   auto rhs = adaptor.getRhs();
                   return builder.create<BinaryOp>(op->getLoc(), lhs, rhs);
                 });
}

struct TransposeOpLowering : public ConversionPattern {
  TransposeOpLowering(MLIRContext *context)
      : ConversionPattern(TransposeOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    lowerOpToLoops(op, operands, rewriter,
                   [&](OpBuilder &builder, ValueRange ivs) -> Value {
                     TransposeOpAdaptor adaptor(operands);
                     auto input = adaptor.getInput();

                     SmallVector<Value, 4> indices(llvm::reverse(ivs));
                     return builder.create<memref::LoadOp>(loc, input, indices);
                   });
    return success();
  }
};

struct AddOpLowering : public ConversionPattern {
  AddOpLowering(MLIRContext *context)
      : ConversionPattern(AddOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    binaryOpToLoops<AddOpAdaptor, arith::AddFOp>(op, operands, rewriter);
    return success();
  }
};

struct MulOpLowering : public ConversionPattern {
  MulOpLowering(MLIRContext *context)
      : ConversionPattern(MulOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    binaryOpToLoops<MulOpAdaptor, arith::MulFOp>(op, operands, rewriter);
    return success();
  }
};

void ToyToAffineLoweringPass::runOnOperation() {
  mlir::ConversionTarget target(getContext());

  target.addLegalDialect<affine::AffineDialect, arith::ArithDialect,
                         memref::MemRefDialect, func::FuncDialect>();

  target.addIllegalDialect<ToyDialect>();
  target.addDynamicallyLegalOp<PrintOp>([](PrintOp op) {
    return llvm::none_of(op->getOperandTypes(),
                         [](Type type) { return type.isa<TensorType>(); });
  });

  RewritePatternSet patterns(&getContext());
  patterns.add<TransposeOpLowering, AddOpLowering, MulOpLowering>(
      &getContext());

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> createToyToAffineLoweringPass() {
  return std::make_unique<ToyToAffineLoweringPass>();
}

} // namespace mlir::toy
