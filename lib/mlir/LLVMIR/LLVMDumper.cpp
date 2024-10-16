#include "toy/mlir/LLVMIR/LLVMDumper.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Export.h"

namespace mlir::toy {

LogicalResult LLVMDumper::Dump(ModuleOp module) {
  auto llvmModule = translateModuleToLLVMIR(module, LLVMCtx);
  if (!llvmModule) {
    Err << "Failed to translate MLIR to LLVM IR\n";
    return failure();
  }

  ExecutionEngine::setupTargetTripleAndDataLayout(llvmModule.get(), &TM);

  auto optPipeline = makeOptimizingTransformer(OptLevel, 0, &TM);
  if (auto err = optPipeline(llvmModule.get())) {
    Err << "Failed to optimize LLVM IR " << err << '\n';
    return failure();
  }

  OS << *llvmModule;
  return success();
}

LogicalResult LLVMDumper::RunJIT(ModuleOp module) {
  auto optPipeline = makeOptimizingTransformer(OptLevel, 0, &TM);

  ExecutionEngineOptions options;
  options.transformer = optPipeline;

  auto engineOpt = ExecutionEngine::create(module, options);
  assert(engineOpt && "Failed to construct a execution engine");
  auto *engine = engineOpt->get();

  auto invokationResult = engine->invoke("main");
  if (!invokationResult) {
    Err << "Failed to run the main function\n";
    return failure();
  }

  return success();
}

} // namespace mlir::toy
