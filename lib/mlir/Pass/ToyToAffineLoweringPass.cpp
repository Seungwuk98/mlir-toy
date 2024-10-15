
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
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

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, func::FuncDialect,
                    memref::MemRefDialect>();
  }
  void runOnOperation() override;

private:
};

static MemRefType tensor2MemRef(RankedTensorType type) {
  return MemRefType::get(type.getShape(), type.getElementType());
}

static Value insertAllocDealloc(Location loc, OpBuilder &builder,
                                MemRefType type) {
  auto currBlock = builder.getBlock();

  auto alloc = builder.create<memref::AllocOp>(loc, type);
  alloc->moveBefore(&currBlock->front());

  auto dealloc = builder.create<memref::DeallocOp>(loc, alloc);
  dealloc->moveBefore(&currBlock->back());

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
                     return builder.create<affine::AffineLoadOp>(loc, input,
                                                                 indices);
                   });
    return success();
  }
};

template <typename Op, typename BinaryOp>
struct BinaryOpLowering : public ConversionPattern {
  BinaryOpLowering(MLIRContext *context)
      : ConversionPattern(Op::getOperationName(), 1, context) {}

  using OpAdaptor = typename Op::Adaptor;

  LogicalResult
  matchAndRewrite(Operation *op, llvm::ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    lowerOpToLoops(
        op, operands, rewriter,
        [&](OpBuilder &builder, ValueRange ivs) -> Value {
          OpAdaptor adaptor(operands);
          auto lhs = adaptor.getLhs();
          auto rhs = adaptor.getRhs();
          auto lhsElement =
              builder.create<affine::AffineLoadOp>(op->getLoc(), lhs, ivs);
          auto rhsElement =
              builder.create<affine::AffineLoadOp>(op->getLoc(), rhs, ivs);
          return builder.create<BinaryOp>(op->getLoc(), lhsElement, rhsElement);
        });
    return success();
  }
};

using AddOpLowering = BinaryOpLowering<AddOp, arith::AddFOp>;
using MulOpLowering = BinaryOpLowering<MulOp, arith::MulFOp>;

struct ConstantOpLowering : public OpRewritePattern<ConstantOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConstantOp op,
                                PatternRewriter &rewriter) const final {
    auto alloc = insertAllocDealloc(
        op->getLoc(), rewriter,
        tensor2MemRef(op->getResult(0).getType().cast<RankedTensorType>()));

    auto tensorElements = op.getValue();
    auto shape = tensorElements.getType().cast<RankedTensorType>().getShape();
    auto maxShapeValue =
        shape.empty() ? 1 : *std::max_element(shape.begin(), shape.end());

    SmallVector<Value> indexValues;
    indexValues.reserve(maxShapeValue);

    for (int64_t idx = 0; idx < maxShapeValue; ++idx)
      indexValues.emplace_back(
          rewriter.create<arith::ConstantIndexOp>(op->getLoc(), idx));

    llvm::SmallVector<Value> values;
    values.reserve(tensorElements.size());
    for (auto element : tensorElements.getValues<FloatAttr>())
      values.emplace_back(
          rewriter.create<arith::ConstantOp>(op->getLoc(), element));

    int64_t offset = 0;
    SmallVector<Value> indices(shape.size());
    createStore(values, indexValues, shape, indices, 0, offset,
                [&](Value value, ArrayRef<Value> indices) {
                  rewriter.create<affine::AffineStoreOp>(op->getLoc(), value,
                                                         alloc, indices);
                });
    rewriter.replaceOp(op, alloc);
    return success();
  }

  static void createStore(ArrayRef<Value> values, ArrayRef<Value> indexValues,
                          ArrayRef<int64_t> shape,
                          SmallVectorImpl<Value> &indices, int64_t dim,
                          int64_t &offset,
                          function_ref<void(Value, ArrayRef<Value>)> fn) {
    if (dim == shape.size()) {
      fn(values[offset++], indices);
      return;
    }

    auto ub = shape[dim];
    for (int64_t i = 0; i < ub; ++i) {
      indices[dim] = indexValues[i];
      createStore(values, indexValues, shape, indices, dim + 1, offset, fn);
    }
  }
};

struct ReturnOpLowering : public OpRewritePattern<ReturnOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReturnOp op,
                                PatternRewriter &rewriter) const final {
    if (op.hasOperand()) {
      op->emitError() << "main function must not return a value";
      return failure();
    }

    rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
    return success();
  }
};

struct PrintOpLowering : public OpConversionPattern<PrintOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.modifyOpInPlace(op,
                             [&] { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

struct FuncOpLowering : public OpConversionPattern<FuncOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto funcName = op.getName();
    if (funcName != "main") {
      op->emitError("All functions are inlined except for 'main'");
      return failure();
    }

    auto operands = adaptor.getOperands();
    if (operands.size() != 0) {
      op->emitError("main function must not have arguments");
      return failure();
    }

    auto returnType = op.getResultTypes();
    if (!returnType.empty()) {
      op->emitError("main function must not return a value");
      return failure();
    }

    auto funcType = rewriter.getFunctionType({}, {});

    auto newFunc = rewriter.create<func::FuncOp>(op.getLoc(), funcName,
                                                 op.getFunctionType());

    rewriter.inlineRegionBefore(op.getRegion(), newFunc.getRegion(),
                                newFunc.end());
    rewriter.eraseOp(op);
    return success();
  }
};

void ToyToAffineLoweringPass::runOnOperation() {
  mlir::ConversionTarget target(getContext());

  target.addLegalDialect<BuiltinDialect, affine::AffineDialect,
                         arith::ArithDialect, memref::MemRefDialect,
                         func::FuncDialect>();

  target.addIllegalDialect<ToyDialect>();
  target.addDynamicallyLegalOp<PrintOp>([](PrintOp op) {
    return llvm::none_of(op->getOperandTypes(),
                         [](Type type) { return type.isa<TensorType>(); });
  });

  RewritePatternSet patterns(&getContext());
  patterns.add<TransposeOpLowering, AddOpLowering, MulOpLowering,
               ConstantOpLowering, ReturnOpLowering, PrintOpLowering,
               FuncOpLowering, AddOpLowering, MulOpLowering>(&getContext());

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> createToyToAffineLoweringPass() {
  return std::make_unique<ToyToAffineLoweringPass>();
}

} // namespace mlir::toy
