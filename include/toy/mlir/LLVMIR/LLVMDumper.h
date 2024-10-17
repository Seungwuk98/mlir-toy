#ifndef MLIR_LLVM_IR_DUMPER_H
#define MLIR_LLVM_IR_DUMPER_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "toy/context/ToyContext.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include <cstdint>

namespace mlir::toy {

class LLVMDumper {
  using ToyContext = ::toy::ToyContext;

public:
  LLVMDumper(ToyContext *ctx, llvm::TargetMachine &targetMachine,
             std::uint8_t optLevel = 0, llvm::raw_ostream &os = llvm::outs(),
             llvm::raw_ostream &err = llvm::errs())
      : Ctx(ctx), TM(targetMachine), OS(os), Err(err), OptLevel(optLevel) {
    registerLLVMDialectTranslation(*ctx);
    registerBuiltinDialectTranslation(*ctx);
  }

  LogicalResult Dump(ModuleOp module);

  LogicalResult RunJIT(ModuleOp module);

private:
  ToyContext *Ctx;
  llvm::TargetMachine &TM;
  llvm::raw_ostream &OS;
  llvm::raw_ostream &Err;
  std::uint8_t OptLevel;
  llvm::LLVMContext LLVMCtx;
};

} // namespace mlir::toy

#endif // MLIR_LLVM_IR_DUMPER_H
