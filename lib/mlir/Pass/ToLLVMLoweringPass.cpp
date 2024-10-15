#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/DialectConversion.h"
#include "toy/mlir/Dialect/ToyOp.h"

#define FLOAT_NEWLINE "%f\n\0"
#define FLOAT_SPACE "%f \0"
#define NEWLINE "\n\0"

namespace mlir::toy {

class ToLLVMLoweringPass
    : public PassWrapper<ToLLVMLoweringPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToLLVMLoweringPass);

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<LLVM::LLVMDialect, scf::SCFDialect, cf::ControlFlowDialect>();
  }

  void runOnOperation() override final;
};

struct PrintOpToLLVMLowering : public OpConversionPattern<PrintOp> {
  using OpConversionPattern<PrintOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto module = op->getParentOfType<ModuleOp>();
    auto printfType = getPrintfType(rewriter);
    auto printf = getOrInsertPrintf(rewriter, module);

    /// need to use original operand
    auto input = op.getInput();
    auto type = input.getType().cast<MemRefType>();
    auto shape = type.getShape();

    DenseMap<int64_t, Value> indexToValue;
    std::ranges::for_each(shape, [&](int64_t idx) {
      auto [iter, inserted] = indexToValue.try_emplace(idx, nullptr);
      if (inserted)
        iter->second =
            rewriter.create<arith::ConstantIndexOp>(op->getLoc(), idx);
    });

    auto constant0 = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 0);
    auto [iter, inserted] = indexToValue.try_emplace(1, nullptr);
    if (inserted)
      iter->second = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 1);
    auto constant1 = iter->second;

    auto floatNewlineFmt =
        getOrInsertGlobalString(rewriter, module, op->getLoc(), FLOAT_NEWLINE);
    auto floatSpaceFmt =
        getOrInsertGlobalString(rewriter, module, op->getLoc(), FLOAT_SPACE);
    auto newline =
        getOrInsertGlobalString(rewriter, module, op->getLoc(), NEWLINE);

    llvm::SmallVector<Value> iterValues;
    iterValues.reserve(shape.size());

    for (auto [idx, elSize] : llvm::enumerate(shape)) {
      auto to = indexToValue[elSize];
      auto scfFor =
          rewriter.create<scf::ForOp>(op->getLoc(), constant0, to, constant1);
      for (auto &nested : *scfFor.getBody())
        rewriter.eraseOp(&nested);
      auto iv = scfFor.getInductionVar();
      iterValues.emplace_back(iv);

      rewriter.setInsertionPointToEnd(scfFor.getBody());

      auto toMinus1 =
          rewriter.create<arith::SubIOp>(op->getLoc(), to, constant1);
      auto ivEqualWithToMinus1 = rewriter.create<arith::CmpIOp>(
          op->getLoc(), arith::CmpIPredicate::eq, iv, toMinus1);

      /// TODO implement printf
      rewriter.create<scf::YieldOp>(op->getLoc());
      rewriter.setInsertionPointToStart(scfFor.getBody());
    }

    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult matchAndRewriteImpl(PrintOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
    auto module = op->getParentOfType<ModuleOp>();
    auto printfType = getPrintfType(rewriter);
    auto printf = getOrInsertPrintf(rewriter, module);

    /// need to use original operand
    auto input = op.getInput();
    auto type = input.getType().cast<MemRefType>();
    auto shape = type.getShape();

    DenseMap<int64_t, Value> indexToValue;
    std::ranges::for_each(shape, [&](int64_t idx) {
      auto [iter, inserted] = indexToValue.try_emplace(idx, nullptr);
      if (inserted)
        iter->second =
            rewriter.create<arith::ConstantIndexOp>(op->getLoc(), idx);
    });

    auto constant0 = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 0);
    auto [iter, inserted] = indexToValue.try_emplace(1, nullptr);
    if (inserted)
      iter->second = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 1);
    auto constant1 = iter->second;

    auto floatNewlineFmt =
        getOrInsertGlobalString(rewriter, module, op->getLoc(), FLOAT_NEWLINE);
    auto floatSpaceFmt =
        getOrInsertGlobalString(rewriter, module, op->getLoc(), FLOAT_SPACE);
    auto newline =
        getOrInsertGlobalString(rewriter, module, op->getLoc(), NEWLINE);

    llvm::SmallVector<Value> iterValues;
    iterValues.reserve(shape.size());

    for (auto [idx, elSize] : llvm::enumerate(shape)) {
      auto to = indexToValue[elSize];

      auto scfFor =
          rewriter.create<scf::ForOp>(op->getLoc(), constant0, to, constant1);
      for (auto &nested : *scfFor.getBody())
        rewriter.eraseOp(&nested);
      auto iv = scfFor.getInductionVar();
      iterValues.emplace_back(iv);

      rewriter.setInsertionPointToEnd(scfFor.getBody());

      auto toMinus1 =
          rewriter.create<arith::SubIOp>(op->getLoc(), to, constant1);
      auto ivEqualWithToMinus1 = rewriter.create<arith::CmpIOp>(
          op->getLoc(), arith::CmpIPredicate::eq, iv, toMinus1);

      if (idx != shape.size() - 1) {
        rewriter.create<scf::IfOp>(op->getLoc(), ivEqualWithToMinus1,
                                   [&](OpBuilder &builder, Location loc) {
                                     builder.create<LLVM::CallOp>(
                                         loc, printfType, printf,
                                         ValueRange{newline});
                                     builder.create<scf::YieldOp>(loc);
                                   });
      } else {
        auto load = rewriter.create<memref::LoadOp>(op->getLoc(), input);
        rewriter.create<scf::IfOp>(
            op->getLoc(), ivEqualWithToMinus1,
            [&](OpBuilder &builder, Location loc) {
              builder.create<LLVM::CallOp>(loc, printfType, printf,
                                           ValueRange{floatSpaceFmt, load});
              builder.create<scf::YieldOp>(loc);
            },
            [&](OpBuilder &builder, Location loc) {
              builder.create<LLVM::CallOp>(loc, printfType, printf,
                                           ValueRange{floatNewlineFmt, load});
              builder.create<scf::YieldOp>(loc);
            });
      }

      rewriter.create<scf::YieldOp>(op->getLoc());

      rewriter.setInsertionPointToStart(scfFor.getBody());
    }

    rewriter.eraseOp(op);
    return success();
  }

  static Value getOrInsertGlobalString(PatternRewriter &rewriter,
                                       ModuleOp module, Location loc,
                                       llvm::StringRef value) {
    auto *context = rewriter.getContext();
    LLVM::GlobalOp globalOp;
    if (!(globalOp = module.lookupSymbol<LLVM::GlobalOp>(value))) {
      auto llvmI8T = IntegerType::get(context, 8);
      auto llvmArrayT = LLVM::LLVMArrayType::get(llvmI8T, value.size());

      PatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());

      auto valueAttr = rewriter.getStringAttr(value);

      StringRef symbol;
      if (value == FLOAT_NEWLINE)
        symbol = "@printf.float.newline";
      else if (value == FLOAT_SPACE)
        symbol = "@printf.float.space";
      else if (value == NEWLINE)
        symbol = "@printf.newline";
      else
        llvm_unreachable("unhandled global string value");

      globalOp = rewriter.create<LLVM::GlobalOp>(
          module->getLoc(), llvmArrayT, /*isConstant=*/true,
          LLVM::Linkage::Internal, symbol, valueAttr);
    }

    auto getCharPtrPtr = rewriter.create<LLVM::AddressOfOp>(loc, globalOp);
    auto constant0 = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
    auto getCharPtr = rewriter.create<LLVM::GEPOp>(
        loc, LLVM::LLVMPointerType::get(module.getContext()),
        globalOp.getType(), getCharPtrPtr,
        ArrayRef<Value>{constant0, constant0});
    return getCharPtr;
  }

  static LLVM::LLVMFunctionType getPrintfType(PatternRewriter &rewriter) {
    auto context = rewriter.getContext();
    auto llvmI32T = IntegerType::get(context, 32);
    auto llvmI8PtrT = LLVM::LLVMPointerType::get(context);
    auto llvmFnT = LLVM::LLVMFunctionType::get(llvmI32T, llvmI8PtrT, true);
    return llvmFnT;
  }

  static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                             ModuleOp module) {
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
      return SymbolRefAttr::get(context, "printf");

    auto llvmFnT = getPrintfType(rewriter);

    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf", llvmFnT);
    return SymbolRefAttr::get(context, "printf");
  }
};

void ToLLVMLoweringPass::runOnOperation() {
  LLVMConversionTarget target(getContext());

  target.addLegalOp<ModuleOp>();

  LLVMTypeConverter typeConverter(&getContext());
  RewritePatternSet patterns(&getContext());

  populateAffineToStdConversionPatterns(patterns);
  populateSCFToControlFlowConversionPatterns(patterns);
  arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  populateFuncToLLVMConversionPatterns(typeConverter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
  populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);

  patterns.add<PrintOpToLLVMLowering>(&getContext());

  if (failed(
          applyFullConversion(getOperation(), target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> createToyToLLVMLoweringPass() {
  return std::make_unique<ToLLVMLoweringPass>();
}

} // namespace mlir::toy
