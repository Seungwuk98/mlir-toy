#include "mlir/IR/OperationSupport.h"
#include "toy/mlir/MLIRGen/IRGenerator.h"
#include "llvm/Support/raw_ostream.h"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "TestExtras.h"
#include "doctest/doctest.h"
#include "toy/parser/Parser.h"

namespace toy::test {

static void MLIRTest(llvm::StringRef Program, llvm::StringRef Expected) {
  llvm::SourceMgr SM;
  ToyContext ctx;
  ctx.shouldPrintOpOnDiagnostic();
  DiagnosticReporter Reporter(SM, llvm::errs());

  auto memBuf = llvm::MemoryBuffer::getMemBuffer(Program);
  auto bufID = SM.AddNewSourceBuffer(std::move(memBuf), {});
  auto bufferRef = SM.getMemoryBuffer(bufID);

  Lexer Lexer(bufferRef->getBuffer(), &ctx, SM, Reporter);
  Parser Parser(Lexer, SM);

  auto module = Parser.Parse();
  if (!module) {
    FAIL("Failed to parse the program");
    return;
  }

  IRGenerator IRGen(&ctx, SM, Reporter);
  module.accept(IRGen);
  if (Reporter.getErrorCount()) {
    FAIL("Failed to generate MLIR");
    return;
  }
  auto mlirModule = IRGen.getModuleOp();

  std::string emittedProgram;
  llvm::raw_string_ostream ss(emittedProgram);

  mlirModule->print(ss);
  STR_EQ(Expected, emittedProgram);
}

#define MLIR_GEN_TEST(TestName, Program, Expected)                             \
  SUBCASE(TestName) MLIRTest(Program, Expected)

TEST_SUITE("MLIR Generation Tests") {}

TEST_CASE("Gen Test" * doctest::test_suite("MLIR Generation Tests")) {
  MLIR_GEN_TEST("Toy Simple Program 1", R"(

def main() {
  var a = [1, 2, 3, 4, 5, 6]; 
  return;
}

)",
                R"(
module {
  toy.func @main() {
    %0 = toy.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>
    toy.return
  }
}
)");

  MLIR_GEN_TEST("Toy Simple Program 2", R"(

def transpose_transpose(a) {
  return transpose(transpose(a));
}

def main() {
  var a<2, 3> = [1, 2, 3, 4, 5, 6];
  transpose_transpose(a);
  return;
}
)",
                R"(
module {
  toy.func @transpose_transpose(%arg0: tensor<*xf64>) -> tensor<*xf64> {
    %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
    %1 = toy.transpose(%0 : tensor<*xf64>) to tensor<*xf64>
    toy.return %1 : tensor<*xf64>
  }
  toy.func @main() {
    %0 = toy.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>
    %1 = toy.reshape(%0 : tensor<6xf64>) to tensor<2x3xf64>
    %2 = toy.generic_call @transpose_transpose(%1) : (tensor<2x3xf64>) -> tensor<*xf64>
    toy.return
  }
}

)");
}

} // namespace toy::test
