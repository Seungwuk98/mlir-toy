#include "MLIRTestUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "toy/mlir/Dialect/ToyOp.h"
#include "toy/mlir/Pass/Passes.h"

namespace toy::test {

bool MLIRTest(llvm::StringRef Program,
              std::function<bool(mlir::ModuleOp)> CheckFn) {
  llvm::SourceMgr SM;
  ToyContext ctx;
  DiagnosticReporter Reporter(SM, llvm::errs());

  auto memBuf = llvm::MemoryBuffer::getMemBuffer(Program);
  auto bufID = SM.AddNewSourceBuffer(std::move(memBuf), {});
  auto bufferRef = SM.getMemoryBuffer(bufID);

  Lexer Lexer(bufferRef->getBuffer(), &ctx, SM, Reporter);
  Parser Parser(Lexer, SM);

  auto module = Parser.Parse();
  if (!module) {
    FAIL("Failed to parse the program");
    return false;
  }

  IRGenerator IRGen(&ctx, SM, Reporter);
  module.accept(IRGen);
  if (Reporter.getErrorCount()) {
    FAIL("Failed to generate MLIR");
    return false;
  }
  auto mlirModule = IRGen.getModuleOp();

  return CheckFn(mlirModule.get());
}

static void cmpModuleAndExpected(mlir::ModuleOp module,
                                 llvm::StringRef Expected) {
  std::string emittedProgram;
  llvm::raw_string_ostream ss(emittedProgram);

  module.print(ss);
  STR_EQ(Expected, emittedProgram);
}

bool MLIRGenTest(llvm::StringRef Program, llvm::StringRef Expected) {
  return MLIRTest(Program, [&](mlir::ModuleOp module) {
    cmpModuleAndExpected(module, Expected);
    return true;
  });
}

bool MLIRGenCanonicalizerPassTest(llvm::StringRef Program,
                                  llvm::StringRef Expected) {
  return MLIRTest(Program, [&](mlir::ModuleOp module) {
    mlir::PassManager pm(module->getName());
    pm.addNestedPass<mlir::toy::FuncOp>(mlir::createCanonicalizerPass());
    if (mlir::failed(pm.run(module))) {
      FAIL("Failed to run mlir pass");
      return false;
    }

    cmpModuleAndExpected(module, Expected);
    return true;
  });
}

bool MLIRGenInlinePassTest(llvm::StringRef Program, llvm::StringRef Expected) {
  return MLIRTest(Program, [&](mlir::ModuleOp module) {
    mlir::PassManager pm(module->getName());
    pm.addPass(mlir::createInlinerPass());
    pm.addNestedPass<mlir::toy::FuncOp>(mlir::createCanonicalizerPass());
    if (mlir::failed(pm.run(module))) {
      FAIL("Failed to run mlir pass");
      return false;
    }

    cmpModuleAndExpected(module, Expected);
    return true;
  });
}

bool MLIRShapeInferencePassTest(llvm::StringRef Program,
                                llvm::StringRef Expected) {
  return MLIRTest(Program, [&](mlir::ModuleOp module) {
    mlir::PassManager pm(module->getName());
    pm.addPass(mlir::createInlinerPass());

    auto &fnPm = pm.nest<mlir::toy::FuncOp>();
    fnPm.addPass(mlir::toy::createShapeInferencePass());
    fnPm.addPass(mlir::createCanonicalizerPass());

    if (mlir::failed(pm.run(module))) {
      FAIL("Failed to run mlir pass");
      return false;
    }

    cmpModuleAndExpected(module, Expected);
    return true;
  });
}

bool MLIRAffineLoweringTest(llvm::StringRef Program, llvm::StringRef Expected) {
  return MLIRTest(Program, [&](mlir::ModuleOp module) {
    mlir::PassManager pm(module->getName());
    pm.addPass(mlir::createInlinerPass());

    auto &fnPm = pm.nest<mlir::toy::FuncOp>();
    fnPm.addPass(mlir::toy::createShapeInferencePass());
    fnPm.addPass(mlir::createCanonicalizerPass());

    pm.addPass(mlir::toy::createToyToAffineLoweringPass());
    if (mlir::failed(pm.run(module))) {
      module.dump();
      FAIL("Failed to run mlir pass");
      return false;
    }

    cmpModuleAndExpected(module, Expected);
    return true;
  });
}

} // namespace toy::test
