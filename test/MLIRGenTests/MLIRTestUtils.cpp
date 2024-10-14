#include "MLIRTestUtils.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "toy/mlir/Dialect/ToyOp.h"

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

bool MLIRGenTest(llvm::StringRef Program, llvm::StringRef Expected) {
  return MLIRTest(Program, [&](mlir::ModuleOp module) {
    std::string emittedProgram;
    llvm::raw_string_ostream ss(emittedProgram);

    module.print(ss);
    STR_EQ(Expected, emittedProgram);

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

    std::string emittedProgram;
    llvm::raw_string_ostream ss(emittedProgram);

    module.print(ss);
    STR_EQ(Expected, emittedProgram);
    return true;
  });
}

} // namespace toy::test
