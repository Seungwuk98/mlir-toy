#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/DialectConversion.h"
#include "toy/mlir/Dialect/ToyDialect.h"
#include "toy/mlir/Dialect/ToyOp.h"
#include "toy/mlir/Pass/Passes.h"
#include "llvm/ADT/StringRef.h"
#include <algorithm>

#define FLOAT StringRef("%f\0", 3)
#define FLOAT_SPACE StringRef("%f \0", 4)
#define NEWLINE StringRef("\n\0", 2)

namespace mlir::toy {

class ToLLVMLoweringPass
    : public PassWrapper<ToLLVMLoweringPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToLLVMLoweringPass);

  void getDependentDialects(DialectRegistry &registry) const override final {
    registry
        .insert<LLVM::LLVMDialect, scf::SCFDialect, cf::ControlFlowDialect>();
  }

  void runOnOperation() override final;

  llvm::StringRef getArgument() const override final {
    return "toy-affine-to-llvm";
  }
};

class PrintOpLoweringPass
    : public PassWrapper<PrintOpLoweringPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrintOpLoweringPass);

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<LLVM::LLVMDialect, scf::SCFDialect, cf::ControlFlowDialect>();
  }

  llvm::StringRef getArgument() const override final {
    return "toy-print-lowering";
  }

  void runOnOperation() override final;
};

struct PrintOpToLLVMLowering : public OpConversionPattern<PrintOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override final {
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

    auto floatFmt =
        getOrInsertGlobalString(rewriter, module, op->getLoc(), FLOAT);

    if (shape.empty()) {
      auto load = rewriter.create<memref::LoadOp>(op->getLoc(), input);
      rewriter.create<LLVM::CallOp>(op->getLoc(), printfType, printf,
                                    ValueRange{floatFmt, load});
      rewriter.eraseOp(op);
      return success();
    }

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
      rewriter.setInsertionPointToStart(scfFor.getBody());

      auto iv = scfFor.getInductionVar();
      iterValues.emplace_back(iv);

      auto toMinus1 =
          rewriter.create<arith::SubIOp>(op->getLoc(), to, constant1);
      auto ivNequalWithToMinus1 = rewriter.create<arith::CmpIOp>(
          op->getLoc(), arith::CmpIPredicate::ne, iv, toMinus1);
      if (idx == shape.size() - 1) {
        auto load =
            rewriter.create<memref::LoadOp>(op->getLoc(), input, iterValues);
        rewriter.create<scf::IfOp>(
            op->getLoc(), ivNequalWithToMinus1,
            [&](OpBuilder &builder, Location loc) {
              builder.create<LLVM::CallOp>(loc, printfType, printf,
                                           ValueRange{floatSpaceFmt, load});
              builder.create<scf::YieldOp>(loc);
            },
            [&](OpBuilder &builder, Location loc) {
              builder.create<LLVM::CallOp>(loc, printfType, printf,
                                           ValueRange{floatFmt, load});
              builder.create<scf::YieldOp>(loc);
            });

      } else {
        rewriter.create<scf::IfOp>(op->getLoc(), ivNequalWithToMinus1,
                                   [&](OpBuilder &builder, Location loc) {
                                     builder.create<LLVM::CallOp>(
                                         loc, printfType, printf,
                                         ValueRange{newline});
                                     builder.create<scf::YieldOp>(loc);
                                   });
      }
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
      if (value == FLOAT)
        symbol = "@printf.float";
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
        globalOp.getType(), getCharPtrPtr, ArrayRef<Value>{constant0});
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

  target.addLegalOp<scf::YieldOp, ModuleOp>();

  RewritePatternSet patterns(&getContext());
  LLVMTypeConverter typeConverter(&getContext());

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

void PrintOpLoweringPass::runOnOperation() {
  LLVMConversionTarget target(getContext());

  target.addLegalDialect<LLVM::LLVMDialect, scf::SCFDialect,
                         cf::ControlFlowDialect, memref::MemRefDialect,
                         affine::AffineDialect, ToyDialect, arith::ArithDialect,
                         func::FuncDialect>();
  target.addLegalOp<scf::YieldOp, ModuleOp>();
  target.addIllegalOp<PrintOp>();

  RewritePatternSet patterns(&getContext());

  patterns.add<PrintOpToLLVMLowering>(&getContext());

  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> createAffineToLLVMLoweringPass() {
  return std::make_unique<ToLLVMLoweringPass>();
}

std::unique_ptr<Pass> createPrintOpLoweringPass() {
  return std::make_unique<PrintOpLoweringPass>();
}

void registerAffineToLLVMLoweringPass() {
  registerPass(createAffineToLLVMLoweringPass);
}

void registerPrintOpLoweringPass() { registerPass(createPrintOpLoweringPass); }

} // namespace mlir::toy
